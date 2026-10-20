# Copyright 2026 Aerodyne Research, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# opentof/runner.py

import os
import sys
import time
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

import pandas as pd
import matplotlib.pyplot as plt
import opentof as ot

# ==============================================================================
# Performance & CPU Threading Initialization
# Sets C-library thread limits to 1 to prevent thread-thrashing during fast 
# micro-optimization loops (e.g., mass calibration peak fitting).
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


@contextmanager
def headless_plotting(enabled: bool = True):
    """
    Context manager that temporarily disables interactive plotting
    without polluting the global Matplotlib state upon script completion.
    """
    was_interactive = plt.isinteractive()
    if enabled:
        plt.ioff()
    try:
        yield
    finally:
        plt.interactive(was_interactive)


def resolve_calibrant_dict(calibrant_list: list) -> Dict[str, float]:
    """Converts a list of chemical formulas or m/z values into exact mass dictionaries."""
    res = {}
    for comp in calibrant_list:
        if isinstance(comp, (int, float)):
            res[f"Mass_{comp:.4f}"] = float(comp)
        else:
            mass = ot.return_mass(comp)
            if mass is not None:
                res[comp] = mass
            else:
                logging.warning(f"Could not parse mass for calibrant formula: '{comp}'")
    return res

def elevate_process_priority():
    """Sets the current process to HIGH priority in-place without respawning."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            # On Linux/macOS, nice values range from -20 (highest) to 19 (lowest)
            # Setting negative nice values on Unix usually requires root/sudo
            p.nice(-10)
            
        # print("[OpenTof] Process priority successfully set to HIGH.")
    except Exception as e:
        print(f"Warning: Failed to set process priority ({e}). Continuing at default priority.")

# Call this at the very beginning of your runner or main entry point
elevate_process_priority()

class PipelineRunner:
    """
    Object-oriented execution runner for OpenTof workflows.
    Provides step callbacks, structured metadata reporting, and clean configuration handling.
    """

    def __init__(self, config: Dict[str, Any], callback: Optional[Callable[[str, float], None]] = None):
        self.cfg = config
        self.callback = callback
        self.deployment: Optional[ot.Deployment] = None
        self.run_summary: Dict[str, Any] = {
            "status": "initialized",
            "step_times": {},
            "export_paths": {},
            "errors": []
        }

    def _notify(self, step_name: str, progress: float):
        """Invoke step callback if attached (useful for GUI progress bars)."""
        logging.info(f"[{progress*100:3.0f}%] {step_name}")
        if self.callback:
            self.callback(step_name, progress)

    def run(self) -> ot.Deployment:
        start_time = time.time()
        inputs = self.cfg.get("data_inputs", {})
        outputs = self.cfg.get("data_outputs", {})
        wf = self.cfg.get("workflow_settings", {})

        plot_flag = outputs.get("save_plots", True)
        default_show_plot = outputs.get("show_plot_flag", False)

        with headless_plotting(enabled=not default_show_plot):
            try:
                # 1. Initialize Deployment
                self._notify("Initializing Deployment", 0.05)
                step_start = time.time()
                self.deployment = ot.Deployment.from_directory(
                    directory=inputs["directory_path"],
                    pattern=inputs.get("pattern", "*.h5"),
                    chunk_size=inputs.get("chunk_size", 1000),
                    plot_dir=outputs.get("plot_dir")
                )
                self.run_summary["step_times"]["init"] = time.time() - step_start

                # Resolve base output directory (uses deployment's resolved plot_dir + default_plot_subdir)
                base_output_dir = Path(
                    self.deployment.plot_dir if getattr(self.deployment, "plot_dir", None)
                    else ot.utils.get_default_plot_dir()
                ) / self.deployment.default_plot_subdir
                base_output_dir.mkdir(parents=True, exist_ok=True)

                # 2. Resolve Calibrants
                calibrant_dict = resolve_calibrant_dict(self.cfg.get("calibrants", []))

                # 3. First-Pass Mass Calibration
                self._notify("Running First-Pass Mass Calibration", 0.20)
                step_start = time.time()
                mc_cfg = wf.get("mass_calibration", {})
                show_mc = mc_cfg.get("show_plot_flag", default_show_plot)
                self.deployment.mass_calibration(
                    calibrants=calibrant_dict,
                    averaging_interval=mc_cfg.get("averaging_interval", 300),
                    auto_reject=mc_cfg.get("auto_reject", True),
                    auto_reject_threshold=mc_cfg.get("auto_reject_threshold", 25.0),
                    interval_auto_reject=mc_cfg.get("interval_auto_reject", False),
                    interval_auto_reject_threshold=mc_cfg.get("interval_auto_reject_threshold", 100.0),
                    interval_max_rejects=mc_cfg.get("interval_max_rejects", 5),
                    plot_flag=plot_flag,
                    show_plot_flag=show_mc
                )
                self.run_summary["step_times"]["mass_calibration_1"] = time.time() - step_start

                # 4. Reference Spectrum
                self._notify("Determining Reference Spectrum", 0.35)
                step_start = time.time()
                ref_cfg = wf.get("reference_spectrum", {})
                ref_mass = ref_cfg.get("reference_peak_mass")
                if isinstance(ref_mass, str):
                    ref_mass = ot.return_mass(ref_mass)
                show_ref = ref_cfg.get("show_plot_flag", default_show_plot)

                self.deployment.determine_reference_spectrum(
                    reference_peak_mass=ref_mass,
                    plt_y_max=ref_cfg.get("plt_y_max", 10),
                    window_seconds=ref_cfg.get("window_seconds", 600),
                    plot_flag=plot_flag,
                    show_plot_flag=show_ref
                )
                self.run_summary["step_times"]["reference_spectrum"] = time.time() - step_start

                # 5. Baseline Determination
                self._notify("Determining Baseline", 0.50)
                step_start = time.time()
                bsl_cfg = wf.get("baseline", {})
                show_bsl = bsl_cfg.get("show_plot_flag", default_show_plot)

                self.deployment.determine_baseline(
                    plt_y_max=bsl_cfg.get("plt_y_max", 10),
                    window_size=bsl_cfg.get("window_size", None),
                    plot_flag=plot_flag,
                    show_plot_flag=show_bsl
                )
                self.run_summary["step_times"]["baseline"] = time.time() - step_start

                # 6. Peak Width & Shape
                self._notify("Determining Peak Width & Shape", 0.65)
                step_start = time.time()
                pw_cfg = wf.get("peak_width", {})
                show_pw = pw_cfg.get("show_plot_flag", default_show_plot)
                self.deployment.determine_peak_width(
                    mode=pw_cfg.get("mode", "ransac"),
                    plot_flag=plot_flag,
                    show_plot_flag=show_pw
                )

                ps_cfg = wf.get("peak_shape", {})
                show_ps = ps_cfg.get("show_plot_flag", default_show_plot)
                self.deployment.determine_peak_shape(
                    tail_intensity_cutoff=ps_cfg.get("tail_intensity_cutoff", 0.03),
                    omega_r=ps_cfg.get("omega_r", 0.75),
                    omega_l=ps_cfg.get("omega_l", 0.65),
                    plot_flag=plot_flag,
                    show_plot_flag=show_ps
                )
                self.run_summary["step_times"]["peak_width_shape"] = time.time() - step_start

                # 7. Optional Second-Pass Mass Calibration
                if mc_cfg.get("second_pass_custom_shape", True):
                    self._notify("Running Second-Pass Mass Calibration", 0.75)
                    step_start = time.time()
                    active_calibrants = self.deployment.calibration['calibrants'].set_index('calibrants')['mass'].to_dict()
                    self.deployment.mass_calibration(
                        calibrants=active_calibrants,
                        averaging_interval=mc_cfg.get("averaging_interval", 300),
                        auto_reject=mc_cfg.get("auto_reject", True),
                        auto_reject_threshold=mc_cfg.get("second_pass_auto_reject_threshold", 10.0),
                        peak_type='custom',
                        custom_shape=self.deployment.custom_peak_shape,
                        plot_flag=plot_flag,
                        show_plot_flag=show_mc
                    )
                    self.run_summary["step_times"]["mass_calibration_2"] = time.time() - step_start

                # 8. Peak List Population & Fitting
                self._notify("Populating Peak List and Fitting", 0.85)
                step_start = time.time()
                pf_cfg = wf.get("peak_fitting", {})
                target_peak_list = pf_cfg.get("peak_list", [])
                self.deployment.populate_peak_list_and_isotopes(target_peak_list)

                fitting_mode = pf_cfg.get("fitting_mode", "constrained")
                peak_type = pf_cfg.get("peak_type", "custom")

                if fitting_mode == "constrained":
                    self.deployment.FFI_constrained(peak_type=peak_type)
                else:
                    self.deployment.FFI_unconstrained(peak_type=peak_type)
                self.run_summary["step_times"]["peak_fitting"] = time.time() - step_start

                # 9. Quantification & Export
                self._notify("Quantifying Signals and Exporting Data", 0.95)
                step_start = time.time()
                q_cfg = wf.get("quantification", {})
                tank_concs = q_cfg.get("calibration_tank_concentrations", {})
                default_conc = q_cfg.get("default_tank_conc", 1000.0)

                export_df = pd.DataFrame({
                    "timestamp": self.deployment.timestamps,
                    "std_acq_data": self.deployment.standard_acquisition_data.astype(int),
                    "cycling_table": self.deployment.cycling_status['type'].astype(int),
                })

                for peak in target_peak_list:
                    area_col = f"{peak}_area"
                    if area_col not in self.deployment.peak_data.columns:
                        logging.warning(f"Peak area column '{area_col}' missing from fit results. Skipping.")
                        continue

                    compound_ions_s = self.deployment.peak_data[area_col]
                    cal_conc = tank_concs.get(peak, default_conc)

                    ppb_bc, _, _, _ = ot.quantify_signal(
                        peak_area_array=compound_ions_s,
                        cycling_dict=self.deployment.cycling_status,
                        cal_tank_conc=cal_conc,
                        zero_flow=q_cfg.get("zero_flow", 250),
                        cal_flow=q_cfg.get("cal_flow", 5),
                        mode=q_cfg.get("interp_mode", "interp"),
                        plot_flag=False
                    )
                    export_df[f"{peak}_ppb"] = ppb_bc

                # --- EXPORT PATH RESOLUTION ---
                # 1. Resolve CSV Output Path
                csv_out_raw = outputs.get("export_csv_path")
                if csv_out_raw:
                    csv_out = Path(csv_out_raw)
                else:
                    csv_out = base_output_dir / "quantified_data.csv"
                
                csv_out.parent.mkdir(parents=True, exist_ok=True)
                export_df.to_csv(csv_out, index=False)
                self.run_summary["export_paths"]["csv"] = str(csv_out)
                logging.info(f"✓ Quantified CSV exported to: {csv_out}")

                # 2. Resolve OpenTof State HDF5 Output Path (.h5 using OT driver)
                # Enabled by default if set to True or if explicit path provided
                export_ot_h5 = outputs.get("export_ot_h5", True)  # Defaults to True
                ot_h5_raw = outputs.get("export_ot_h5_path")
                
                if export_ot_h5:
                    if ot_h5_raw:
                        ot_h5_out = Path(ot_h5_raw)
                    else:
                        ot_h5_out = base_output_dir / "opentof_deployment.h5"

                    self.deployment.export_to_h5(str(ot_h5_out), driver="OT")
                    self.run_summary["export_paths"]["ot_h5"] = str(ot_h5_out)
                    logging.info(f"✓ OpenTof deployment state saved to: {ot_h5_out}")

                self.run_summary["step_times"]["quantification_export"] = time.time() - step_start
                self.run_summary["total_elapsed"] = time.time() - start_time
                self.run_summary["status"] = "completed"

                # Save JSON Execution Summary
                summary_json_path = base_output_dir / "run_summary.json"
                with open(summary_json_path, "w") as f:
                    json.dump(self.run_summary, f, indent=2)

                self._notify("Pipeline Finished", 1.0)
                return self.deployment

            except Exception as e:
                self.run_summary["status"] = "failed"
                self.run_summary["errors"].append(str(e))
                logging.error(f"Pipeline failed: {e}", exc_info=True)
                raise e


def run_do_it(config_path: str) -> ot.Deployment:
    """Convenience wrapper for CLI execution."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}. Please define one! See examples in documentation.")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    runner = PipelineRunner(config=cfg)
    return runner.run()


def main():
    """Terminal entry point invoked by the `opentof-doit` CLI script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(
        description="OpenTof DO-IT Command Line Workflow Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="config_template.yaml", 
        help="Path to YAML processing configuration file"
    )
    args = parser.parse_args()
    
    run_do_it(args.config)


if __name__ == "__main__":
    elevate_process_priority()
    main()