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

# Suppress Matplotlib GUI Popups for Headless Execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()  # Turn interactive mode off

import time
import yaml
import argparse
import pandas as pd
import opentof as ot

def resolve_calibrant_dict(calibrant_list):
    """Converts a list of chemical formulas into exact mass dictionaries."""
    return {comp: ot.return_mass(comp) for comp in calibrant_list}

# opentof/runner.py

def run_do_it(config_path):
    start_time = time.time()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    inputs = cfg["data_inputs"]
    outputs = cfg["data_outputs"]
    wf = cfg["workflow_settings"]
    
    # Global plot toggles
    plot_flag = outputs.get("save_plots", True)
    default_show_plot = outputs.get("show_plot_flag", False)  # Defaults to False!

    print("==================================================")
    print("           OpenTof DO-IT Pipeline                 ")
    print("==================================================")
    print(f"Loading dataset from: {inputs['directory_path']}")

    # 2. Initialize Deployment
    d = ot.Deployment.from_directory(
        directory=inputs["directory_path"],
        pattern=inputs.get("pattern", "*.h5"),
        chunk_size=inputs.get("chunk_size", 1000),
        plot_dir=outputs.get("plot_dir")
    )

    # 3. Resolve Calibrants
    calibrant_dict = resolve_calibrant_dict(cfg["calibrants"])

    # 4. First-Pass Mass Calibration
    mc_cfg = wf.get("mass_calibration", {})
    show_mc = mc_cfg.get("show_plot_flag", default_show_plot)
    
    print("\n[Step 1/6] Running First-Pass Mass Calibration...")
    d.mass_calibration(
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

    # 5. Reference Spectrum
    ref_cfg = wf.get("reference_spectrum", {})
    ref_mass = ref_cfg.get("reference_peak_mass")
    if isinstance(ref_mass, str):
        ref_mass = ot.return_mass(ref_mass)
    show_ref = ref_cfg.get("show_plot_flag", default_show_plot)

    print("\n[Step 2/6] Determining Reference Spectrum...")
    d.determine_reference_spectrum(
        reference_peak_mass=ref_mass,
        plt_y_max=ref_cfg.get("plt_y_max", 10),
        window_seconds=ref_cfg.get("window_seconds", 600),
        plot_flag=plot_flag,
        show_plot_flag=show_ref
    )

    # 6. Baseline Determination
    bsl_cfg = wf.get("baseline", {})
    show_bsl = bsl_cfg.get("show_plot_flag", default_show_plot)

    print("\n[Step 3/6] Determining Baseline...")
    d.determine_baseline(
        plt_y_max=bsl_cfg.get("plt_y_max", 10),
        window_size=bsl_cfg.get("window_size", None),
        plot_flag=plot_flag,
        show_plot_flag=show_bsl
    )

    # 7. Peak Width & Peak Shape
    pw_cfg = wf.get("peak_width", {})
    show_pw = pw_cfg.get("show_plot_flag", default_show_plot)

    print("\n[Step 4/6] Determining Peak Width and Shape...")
    d.determine_peak_width(
        mode=pw_cfg.get("mode", "ransac"),
        plot_flag=plot_flag,
        show_plot_flag=show_pw
    )

    ps_cfg = wf.get("peak_shape", {})
    show_ps = ps_cfg.get("show_plot_flag", default_show_plot)

    d.determine_peak_shape(
        tail_intensity_cutoff=ps_cfg.get("tail_intensity_cutoff", 0.03),
        omega_r=ps_cfg.get("omega_r", 0.75),
        omega_l=ps_cfg.get("omega_l", 0.65),
        plot_flag=plot_flag,
        show_plot_flag=show_ps
    )

    # 8. Optional Second-Pass Mass Calibration with Custom Shape
    if mc_cfg.get("second_pass_custom_shape", True):
        print("\n-> Re-running Mass Calibration using Empirical Custom Peak Shape...")
        active_calibrants = d.calibration['calibrants'].set_index('calibrants')['mass'].to_dict()
        d.mass_calibration(
            calibrants=active_calibrants,
            averaging_interval=mc_cfg.get("averaging_interval", 300),
            auto_reject=mc_cfg.get("auto_reject", True),
            auto_reject_threshold=mc_cfg.get("second_pass_auto_reject_threshold", 10.0),
            peak_type='custom',
            custom_shape=d.custom_peak_shape,
            plot_flag=plot_flag,
            show_plot_flag=show_mc
        )

    # 9. Peak List Population & Fitting
    pf_cfg = wf.get("peak_fitting", {})
    target_peak_list = pf_cfg.get("peak_list", [])
    print(f"\n[Step 5/6] Populating Peak List ({len(target_peak_list)} target compounds)...")
    d.populate_peak_list_and_isotopes(target_peak_list)

    fitting_mode = pf_cfg.get("fitting_mode", "constrained")
    peak_type = pf_cfg.get("peak_type", "custom")
    
    print(f"-> Running {fitting_mode} peak fitting (shape: '{peak_type}')...")
    if fitting_mode == "constrained":
        d.FFI_constrained(peak_type=peak_type)
    else:
        d.FFI_unconstrained(peak_type=peak_type)

    # 10. Quantification & Export
    print("\n[Step 6/6] Quantifying Signals and Exporting Data...")
    q_cfg = wf.get("quantification", {})
    tank_concs = q_cfg.get("calibration_tank_concentrations", {})
    default_conc = q_cfg.get("default_tank_conc", 1000.0)

    export_df = pd.DataFrame({
        "timestamp": d.timestamps,
        "std_acq_data": d.standard_acquisition_data.astype(int)
    })

    for peak in target_peak_list:
        area_col = f"{peak}_area"
        if area_col not in d.peak_data.columns:
            print(f"⚠️ Warning: Peak area column '{area_col}' missing from fit results. Skipping.")
            continue

        compound_ions_s = d.peak_data[area_col]
        cal_conc = tank_concs.get(peak, default_conc)

        ppb_bc, _, _, _ = ot.quantify_signal(
            peak_area_array=compound_ions_s,
            cycling_dict=d.cycling_status,
            cal_tank_conc=cal_conc,
            zero_flow=q_cfg.get("zero_flow", 250),
            cal_flow=q_cfg.get("cal_flow", 5),
            mode=q_cfg.get("interp_mode", "interp"),
            plot_flag=False
        )
        export_df[f"{peak}_ppb"] = ppb_bc

    # Save Output CSV
    csv_out = outputs["export_csv_path"]
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    export_df.to_csv(csv_out, index=False)
    print(f"✓ Output CSV successfully exported to: {csv_out}")

    if outputs.get("export_ot_h5_path"):
        ot_h5_out = outputs["export_ot_h5_path"]
        d.export_to_h5(ot_h5_out, driver="OT")
        print(f"✓ OpenTof deployment state saved to: {ot_h5_out}")

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" DO-IT Workflow finished in {elapsed:.2f} seconds! ")
    print("==================================================")

def main():
    """Terminal entry point invoked by the `opentof-doit` CLI script."""
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
    main()