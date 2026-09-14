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
# opentof/interactive.py

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pandas as pd
import sys
import time
import os
import re
import inspect
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from opentof.deployment import Deployment
from opentof.peak_fitting import (
    multi_overlap_peak_fit,
    peak_function_selector,
)
from opentof.isotopes import (
    isotope_signal_on_axis,
)
from opentof.utils import (
    calculate_mass_deviation,
    get_nm_segment_data,
    return_mass,
    load_peak_list,
    get_default_plot_dir,
)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def parse_calibrants_file(filepath):
    """
    Parses calibrants/peaks from Tofwerk .cal files, Tofware .txt peak lists, or generic CSV/tab files.
    Filters out unknown placeholders, brackets (<unknown>), and unparsable strings.
    """
    cal_dict = {}
    path_obj = Path(filepath)

    def is_valid_label(label):
        s = str(label).strip()
        if not s or s.lower() == "nan" or s.lower() == "none":
            return False
        if "<" in s or ">" in s or "unknown" in s.lower():
            return False
        return True

    # 1. Tofwerk INI-style .cal format parsing
    try:
        content = path_obj.read_text(errors="ignore")
        names = dict(re.findall(r"^n(\d+)=(.*)$", content, re.MULTILINE))
        masses = dict(re.findall(r"^x(\d+)=([0-9.]+)", content, re.MULTILINE))

        for idx_str, formula in names.items():
            clean_formula = formula.strip()
            if is_valid_label(clean_formula) and idx_str in masses:
                try:
                    m_val = float(masses[idx_str])
                    if m_val > 0:
                        cal_dict[clean_formula] = m_val
                except ValueError:
                    pass

        if cal_dict:
            return cal_dict
    except Exception:
        pass

    # 2. Tab-delimited / CSV Peak Table parsing (Tofware .txt or .csv)
    try:
        df = pd.read_csv(filepath, sep=None, engine="python", comment="#")
        
        formula_col = None
        for col in ["ion", "Formula", "MolecularFormula", "sumFormula", "label", "peak"]:
            if col in df.columns:
                formula_col = col
                break
                
        mass_col = None
        for col in ["x0", "ExactMass", "mass", "Mass", "m/z", "mz", "center_mass"]:
            if col in df.columns:
                mass_col = col
                break

        if formula_col is not None and mass_col is not None:
            for _, row in df.iterrows():
                f_val = str(row[formula_col]).strip()
                if not is_valid_label(f_val):
                    continue
                try:
                    m_val = float(row[mass_col])
                    if not pd.isna(m_val) and m_val > 0:
                        cal_dict[f_val] = m_val
                except (ValueError, TypeError):
                    continue

            if cal_dict:
                return cal_dict
    except Exception:
        pass

    # 3. Fallback to opentof.utils.load_peak_list
    try:
        raw_dict = load_peak_list(filepath, make_calibrant_list=True)
        if raw_dict and isinstance(raw_dict, dict):
            for k, v in raw_dict.items():
                if is_valid_label(k) and v is not None and v > 0:
                    cal_dict[k] = v
            return cal_dict
    except Exception:
        pass

    return cal_dict


class TextRedirector:
    """Redirects stdout/stderr streams to a CTkTextbox, updating carriage returns in place."""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, str_val):
        try:
            self.textbox.configure(state="normal")
            if '\r' in str_val:
                parts = str_val.split('\r')
                for part in parts[:-1]:
                    if part:
                        self.textbox.delete("end-1c linestart", "end-1c")
                        self.textbox.insert("end", part)
                if parts[-1]:
                    self.textbox.delete("end-1c linestart", "end-1c")
                    self.textbox.insert("end", parts[-1])
            else:
                self.textbox.insert("end", str_val)
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
        except Exception:
            pass

    def flush(self):
        pass


class StepSettingsWindow(ctk.CTkToplevel):
    """Generic top-level configuration window for pipeline step parameters."""
    def __init__(self, parent, step_title, param_defs, target_dict):
        super().__init__(parent)
        self.parent = parent
        self.target_dict = target_dict
        self.title(f"{step_title} Parameters")
        self.geometry("440x520")
        self.transient(parent)
        self.grab_set()

        lbl = ctk.CTkLabel(self, text=f"Configure {step_title}", font=ctk.CTkFont(size=15, weight="bold"))
        lbl.pack(padx=15, pady=15)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.entries = {}
        for key, default_val, t_type in param_defs:
            frame = ctk.CTkFrame(self.scroll)
            frame.pack(fill="x", padx=5, pady=4)
            
            lbl_p = ctk.CTkLabel(frame, text=key, font=ctk.CTkFont(size=12))
            lbl_p.pack(side="left", padx=10, pady=5)
            
            ent = ctk.CTkEntry(frame, width=150)
            current_val = self.target_dict.get(key, default_val)
            ent.insert(0, str(current_val))
            ent.pack(side="right", padx=10, pady=5)
            
            self.entries[key] = (ent, t_type)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        save_btn = ctk.CTkButton(btn_frame, text="Save Parameters", fg_color="#2ca02c", command=self.save)
        save_btn.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", fg_color="#dc3545", command=self.destroy)
        cancel_btn.pack(side="right", fill="x", expand=True, padx=5, pady=5)

    def save(self):
        for key, (ent, t_type) in self.entries.items():
            raw_val = ent.get().strip()
            try:
                if raw_val.lower() == "none" or raw_val == "":
                    self.target_dict[key] = None
                elif t_type == bool:
                    self.target_dict[key] = raw_val.lower() in ("true", "1", "yes")
                else:
                    self.target_dict[key] = t_type(raw_val)
            except ValueError:
                pass
        print(f"[GUI]: Updated parameters for step.")
        self.destroy()


class MopfSettingsWindow(ctk.CTkToplevel):
    """Top-level configuration window for MOPF optimization parameters."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("MOPF Optimization Parameters")
        self.geometry("420x520")
        self.transient(parent)
        self.grab_set()

        lbl = ctk.CTkLabel(self, text="Configure MOPF Engine", font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(padx=15, pady=15)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.entries = {}
        params = [
            ("noise_std_mult", self.parent.mopf_params.get("noise_std_mult", 10.0), float),
            ("max_iter", self.parent.mopf_params.get("max_iter", 1000), int),
            ("tol", self.parent.mopf_params.get("tol", 1e-8), float),
            ("deriv_threshold", self.parent.mopf_params.get("deriv_threshold", 0.0), float),
            ("peaks_to_add_post_fit", self.parent.mopf_params.get("peaks_to_add_post_fit", 0), int),
            ("abs_residual_threshold", self.parent.mopf_params.get("abs_residual_threshold", 0.0), float),
            ("peak_width_constraint_factor", self.parent.mopf_params.get("peak_width_constraint_factor", 5.0), float),
        ]

        for key, val, t_type in params:
            frame = ctk.CTkFrame(self.scroll)
            frame.pack(fill="x", padx=5, pady=4)
            
            lbl_p = ctk.CTkLabel(frame, text=key, font=ctk.CTkFont(size=12))
            lbl_p.pack(side="left", padx=10, pady=5)
            
            ent = ctk.CTkEntry(frame, width=130)
            ent.insert(0, str(val))
            ent.pack(side="right", padx=10, pady=5)
            
            self.entries[key] = (ent, t_type)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        save_btn = ctk.CTkButton(btn_frame, text="Save Parameters", fg_color="#2ca02c", command=self.save)
        save_btn.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", fg_color="#dc3545", command=self.destroy)
        cancel_btn.pack(side="right", fill="x", expand=True, padx=5, pady=5)

    def save(self):
        for key, (ent, t_type) in self.entries.items():
            raw_val = ent.get().strip()
            try:
                self.parent.mopf_params[key] = t_type(raw_val)
            except ValueError:
                pass
        print("[GUI]: MOPF parameters updated.")
        self.destroy()


class SpectrumWizardGUI(ctk.CTk):
    def __init__(self, deployment_obj=None, spectra=None, spectra_mass_axis=None, initial_mz_zoom=None):
        super().__init__()

        # --- Data Streams & State ---
        self.deployment = deployment_obj
        self.current_spectra = None
        self.current_mass_axis = None
        self.active_spectra_data = None
        self.calibrants_dict = None
        self.selected_ts_indices = None

        self._initialize_data_sources(spectra, spectra_mass_axis)

        self.showing_subtracted = False

        # Canvas overlay element trackers
        self.peak_list_elements = []
        self.calibrant_elements = []
        self.mopf_elements = []
        self.spectrum_trace = None
        self.fit_trace = None
        self.iso_trace = None
        self.baseline_trace = None

        self.selected_mopf_mass = None
        self.show_peak_list = True

        # Pipeline step Kwargs dictionaries
        self.pipeline_kwargs = {
            "mass_calibration": {
                "averaging_interval": 300,
                "mass_cal_mode": 2,
                "auto_reject": True,
                "auto_reject_threshold": 10.0,
                "track_sip_drift": True,
                "search_range": 31,
                "supersaturated_calibrants": False,
                "mz_anchor": False,
                "peak_type": "gaussian",
                "overwrite_calibration": True
            },
            "determine_reference_spectrum": {
                "reference_peak_mass": None,
                "window_seconds": 600,
                "min_fraction": 0.9,
                "peak_type": "gaussian",
                "fast_mode": False,
                "fit_every_n": 3,
                "max_iter": 100
            },
            "determine_baseline": {
                "window_size": None,
                "cutoff_freq": 5e7,
                "order": 1,
                "smoothing_window": 10,
                "noise_percentile": 10.0,
                "noise_scaling": 1.0,
                "correct_offset": False
            },
            "determine_peak_width": {
                "mode": "ransac",
                "min_prominence": 0.9,
                "minimum_mass_spacing": 1.0,
                "residual_threshold_multiplier": 0.10,
                "top_n": 150,
                "max_num_outlires": 10,
                "peak_type": "gaussian"
            },
            "determine_peak_shape": {
                "iqr_thresh_left": 2.5,
                "iqr_thresh_right": 2.5,
                "omega_l": 0.80,
                "omega_r": 0.70,
                "flatness_thresh": 0.95,
                "flatness_region": 0.5,
                "num_peak_threshold": 70,
                "step": 10,
                "peak_smoothing": 0,
                "mean_smoothing": 1.0,
                "cursor_position": 8,
                "tail_sigma_pos": 9,
                "tail_intensity_cutoff": 0.03
            },
            "FFI_constrained": {
                "peak_type": "pseudo_voigt",
                "nm_search_range": 0.5,
            },
            "FFI_unconstrained": {
                "peak_type": "pseudo_voigt",
                "noise_std_mult": 10.0,
                "nm_search_range": 0.5,
                "max_iter": 25,
                "tol": 0.0001,
            }
        }

        # Default MOPF dictionary parameters
        self.mopf_params = {
            "noise_std_mult": 10.0,
            "max_iter": 1000,
            "tol": 1e-8,
            "deriv_threshold": 0.0,
            "peaks_to_add_post_fit": 0,
            "abs_residual_threshold": 0.0,
            "peak_width_constraint_factor": 5.0
        }

        # Window Layout Setup
        self.title("OpenTof Processing & Analysis Wizard")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        target_w = min(1600, int(screen_w * 0.95))
        target_h = min(900, int(screen_h * 0.85))
        self.geometry(f"{target_w}x{target_h}")

        self.grid_columnconfigure(0, weight=0, minsize=350)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Build UI Sections
        self._build_sidebar()
        self._build_main_panel()

        # Redirect standard output streams to Console Panel
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TextRedirector(self.console_textbox)
        sys.stderr = TextRedirector(self.console_textbox)

        # Plot Initial State
        self.check_and_plot_existing_baseline()
        self.render_deployment_peak_list()
        self.render_deployment_calibrants()
        if initial_mz_zoom is not None:
            self.ax.set_xlim(initial_mz_zoom - 0.5, initial_mz_zoom + 0.5)

        self.protocol("WM_DELETE_WINDOW", self.close_and_resume)
        print("🚀 OpenTof Processing Wizard initialized.")

        if self.deployment is None:
            print("[GUI]: No active Deployment object loaded. Use 'Load Dataset' in the Pipeline tab to get started.")

    def _initialize_data_sources(self, spectra=None, spectra_mass_axis=None):
        """
        Helper to resolve spectrum and mass axis arrays.
        Samples 100-250 random spectra out-of-core if no reference spectrum exists.
        """
        if self.deployment is not None:
            if spectra is not None:
                self.current_spectra = np.asarray(spectra, dtype=float)
            elif getattr(self.deployment, "reference", None) is not None and "reference_spectrum" in self.deployment.reference:
                self.current_spectra = np.asarray(self.deployment.reference['reference_spectrum'], dtype=float)
            elif getattr(self.deployment, "tofdata", None) is not None:
                try:
                    total_spectra = self.deployment.tofdata.shape[0]
                    n_samples = min(250, total_spectra)
                    random_indices = np.sort(np.random.choice(total_spectra, size=n_samples, replace=False))
                    sampled_data = self.deployment.tofdata[random_indices, :].compute()
                    self.current_spectra = np.asarray(np.mean(sampled_data, axis=0), dtype=float)
                    print(f"[GUI]: Sampled and averaged {n_samples} random spectra from deployment.tofdata for initial display.")
                except Exception as e:
                    print(f"[GUI Notice]: Could not compute initial averaged spectrum from tofdata: {e}")
                    self.current_spectra = np.zeros(1000, dtype=float)
            else:
                self.current_spectra = np.zeros(1000, dtype=float)

            if spectra_mass_axis is not None:
                self.current_mass_axis = np.asarray(spectra_mass_axis, dtype=float)
            elif getattr(self.deployment, "reference", None) is not None and "rs_mass_axis" in self.deployment.reference:
                self.current_mass_axis = np.asarray(self.deployment.reference['rs_mass_axis'], dtype=float)
            elif getattr(self.deployment, "first_guess_mass_axis", None) is not None:
                self.current_mass_axis = np.asarray(self.deployment.first_guess_mass_axis, dtype=float)
            else:
                self.current_mass_axis = np.linspace(1, 500, len(self.current_spectra), dtype=float)
        else:
            self.current_spectra = np.zeros(1000, dtype=float)
            self.current_mass_axis = np.linspace(1, 500, 1000, dtype=float)

        self.active_spectra_data = self.current_spectra.copy()

    # =========================================================================
    # UI INITIALIZATION HELPERS
    # =========================================================================
    def _build_sidebar(self):
        """Builds the left scrollable navigation control panel."""
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(1, weight=1)

        title_lbl = ctk.CTkLabel(
            self.sidebar_frame, text="OpenTof HRTS GUI", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_lbl.pack(padx=15, pady=(15, 10))

        # Control Tabview
        self.sidebar_tabs = ctk.CTkTabview(self.sidebar_frame, width=330)
        self.sidebar_tabs.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_pipeline = self.sidebar_tabs.add("Pipeline")
        self.tab_peaks = self.sidebar_tabs.add("Peak List")
        self.tab_view = self.sidebar_tabs.add("View & Fit")

        self._populate_pipeline_tab()
        self._populate_peaks_tab()
        self._populate_view_tab()

        self.close_button = ctk.CTkButton(
            self.sidebar_frame, text="Save & Return to Notebook", 
            fg_color="#5a5a5a", hover_color="#404040", command=self.close_and_resume
        )
        self.close_button.pack(padx=15, pady=10, fill="x")

    def _populate_pipeline_tab(self):
        """Initial Data Loading Cards + Calibrants Config + Pipeline Runners + Export Actions."""
        frame = ctk.CTkScrollableFrame(self.tab_pipeline)
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        # --- DATASET LOADER SECTION ---
        lbl_load = ctk.CTkLabel(frame, text="0. Load Dataset", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_load.pack(padx=10, pady=(5, 5), anchor="w")

        btn_dir = ctk.CTkButton(frame, text="📁 Load Directory (.h5)", fg_color="#17a2b8", command=self._load_from_directory_dialog)
        btn_dir.pack(padx=10, pady=3, fill="x")

        btn_single = ctk.CTkButton(frame, text="📄 Load Single File (.h5)", fg_color="#17a2b8", command=self._load_single_file_dialog)
        btn_single.pack(padx=10, pady=3, fill="x")

        btn_state = ctk.CTkButton(frame, text="💾 Open State File (.h5)", fg_color="#6f42c1", command=self._load_ot_state_dialog)
        btn_state.pack(padx=10, pady=3, fill="x")

        # --- MASS CALIBRANTS SECTION ---
        lbl_cal = ctk.CTkLabel(frame, text="Mass Calibrants Setup", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_cal.pack(padx=10, pady=(15, 5), anchor="w")

        self.calibrants_entry = ctk.CTkEntry(frame, placeholder_text="Formulas (e.g. C2H5+, C6H6+)")
        self.calibrants_entry.pack(padx=10, pady=3, fill="x")

        btn_set_cal = ctk.CTkButton(frame, text="Set Manual Calibrants", fg_color="#6c757d", command=self._set_manual_calibrants)
        btn_set_cal.pack(padx=10, pady=3, fill="x")

        btn_load_cal = ctk.CTkButton(frame, text="📁 Load Calibrants (.cal / .txt)", fg_color="#17a2b8", command=self._load_calibrants_file)
        btn_load_cal.pack(padx=10, pady=3, fill="x")

        # --- PIPELINE METHODS SECTION ---
        lbl_pipe = ctk.CTkLabel(frame, text="Deployment Processing Pipeline", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_pipe.pack(padx=10, pady=(15, 5), anchor="w")

        methods = [
            ("1. Mass Calibration", self._run_mass_cal, "#1f77b4", "mass_calibration", [
                ("averaging_interval", 300, int), ("mass_cal_mode", 2, int), ("auto_reject", False, bool),
                ("auto_reject_threshold", 10.0, float), ("track_sip_drift", True, bool), ("search_range", 31, int),
                ("supersaturated_calibrants", False, bool), ("mz_anchor", False, bool), ("peak_type", "gaussian", str)
            ]),
            ("2. Reference Spectrum", self._run_ref_spec, "#1f77b4", "determine_reference_spectrum", [
                ("reference_peak_mass", "None", str), ("window_seconds", 600, int), ("min_fraction", 0.9, float),
                ("peak_type", "gaussian", str), ("fast_mode", False, bool), ("fit_every_n", 3, int), ("max_iter", 100, int)
            ]),
            ("3. Determine Baseline", self._run_baseline_cal, "#2ca02c", "determine_baseline", [
                ("window_size", "None", str), ("cutoff_freq", 0.001, float), ("order", 1, int),
                ("smoothing_window", 10, int), ("noise_percentile", 10.0, float), ("noise_scaling", 1.0, float),
                ("correct_offset", False, bool)
            ]),
            ("4. Peak Width Function", self._run_peak_width, "#ff7f0e", "determine_peak_width", [
                ("mode", "ransac", str), ("min_prominence", 0.9, float), ("minimum_mass_spacing", 1.0, float),
                ("residual_threshold_multiplier", 0.10, float), ("top_n", 150, int), ("max_num_outlires", 10, int)
            ]),
            ("5. Empirical Peak Shape", self._run_peak_shape, "#9467bd", "determine_peak_shape", [
                ("iqr_thresh_left", 2.5, float), ("iqr_thresh_right", 2.5, float),
                ("omega_l", 0.80, float), ("omega_r", 0.70, float),
                ("flatness_thresh", 0.95, float), ("flatness_region", 0.5, float),
                ("num_peak_threshold", 70, int), ("step", 10, int),
                ("peak_smoothing", 0, int), ("mean_smoothing", 1.0, float), ("cursor_position", 8, int),
                ("tail_sigma_pos", 9, int), ("tail_intensity_cutoff", 0.03, float)
            ]),
            ("6. Define Peak List", self._run_define_peak_list, "#20c997", None, []),
            ("7a. FFI Constrained Fit", self._run_ffi_constrained, "#d62728", "FFI_constrained", [
                ("peak_type", "pseudo_voigt", str), ("nm_search_range", 0.5, float), ("chunk_size", 1000, int)
            ]),
            ("7b. FFI Unconstrained Fit", self._run_ffi_unconstrained, "#8c564b", "FFI_unconstrained", [
                ("peak_type", "pseudo_voigt", str), ("noise_std_mult", 10.0, float), ("nm_search_range", 0.5, float),
                ("max_iter", 25, int), ("tol", 0.0001, float), ("chunk_size", 1000, int)
            ]),
        ]

        for item in methods:
            text, cmd, col, step_key, param_defs = item
            row_frame = ctk.CTkFrame(frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=5, pady=3)

            btn = ctk.CTkButton(row_frame, text=text, fg_color=col, hover_color="#333333", command=cmd)
            btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

            if param_defs:
                gear_btn = ctk.CTkButton(
                    row_frame, text="⚙", width=32, fg_color="#495057", hover_color="#343a40",
                    command=lambda k=step_key, title=text, p_defs=param_defs: self._open_step_settings(title, k, p_defs)
                )
                gear_btn.pack(side="right")

        # --- EXPORT SECTION ---
        lbl_exp = ctk.CTkLabel(frame, text="7. Export & Save Results", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_exp.pack(padx=10, pady=(15, 5), anchor="w")

        btn_exp_h5 = ctk.CTkButton(frame, text="💾 Export Workspace (HDF5)", fg_color="#6f42c1", command=self._export_workspace_h5)
        btn_exp_h5.pack(padx=10, pady=3, fill="x")

        btn_exp_csv = ctk.CTkButton(frame, text="📊 Save Minimal CSV (peak_data)", fg_color="#28a745", command=self._save_minimal_csv)
        btn_exp_csv.pack(padx=10, pady=3, fill="x")

    def _open_step_settings(self, title, step_key, param_defs):
        StepSettingsWindow(self, title, param_defs, self.pipeline_kwargs[step_key])

    def _populate_view_tab(self):
        """Spectrum View & Fitting Controls."""
        frame = ctk.CTkScrollableFrame(self.tab_view)
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(frame, text="Data Stream Source", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(5, 2), anchor="w")
        self.stream_menu = ctk.CTkOptionMenu(
            frame, values=["Reference Spectrum", "Global Averaged (GAD)"],
            command=self._change_data_stream
        )
        self.stream_menu.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(frame, text="Baseline & Trace Options", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 2), anchor="w")
        
        self.window_size_entry = ctk.CTkEntry(frame, placeholder_text="Baseline Window Size (e.g. 31)")
        self.window_size_entry.pack(padx=10, pady=4, fill="x")

        self.baseline_button = ctk.CTkButton(
            frame, text="Toggle Baseline Floor", fg_color="#2ca02c", command=self.toggle_baseline
        )
        self.baseline_button.pack(padx=10, pady=4, fill="x")

        self.subtraction_button = ctk.CTkButton(
            frame, text="Apply Baseline Subtraction", fg_color="#9467bd", command=self.toggle_subtraction
        )
        self.subtraction_button.pack(padx=10, pady=4, fill="x")

        ctk.CTkLabel(frame, text="Display & Scale", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 2), anchor="w")
        
        self.log_checkbox = ctk.CTkCheckBox(frame, text="Log Scale (Y-axis)", command=self.toggle_log_scale)
        self.log_checkbox.pack(padx=10, pady=4, fill="x")

        self.autoy_button = ctk.CTkButton(frame, text="Auto-Y Scale to Zoom", fg_color="#e377c2", command=self.auto_y_scale)
        self.autoy_button.pack(padx=10, pady=4, fill="x")

        ctk.CTkLabel(frame, text="MOPF Fitting Engine", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 2), anchor="w")
        
        self.fit_button = ctk.CTkButton(frame, text="Run MOPF on Zoom Window", command=self.run_mopf)
        self.fit_button.pack(padx=10, pady=4, fill="x")

        self.config_mopf_button = ctk.CTkButton(frame, text="⚙ MOPF Parameters", fg_color="#17a2b8", command=self.open_mopf_settings)
        self.config_mopf_button.pack(padx=10, pady=4, fill="x")

        self.discovery_button = ctk.CTkButton(frame, text="🔍 Automated Peak Discovery", fg_color="#1f77b4", command=self.run_global_peak_discovery)
        self.discovery_button.pack(padx=10, pady=4, fill="x")

    def _populate_peaks_tab(self):
        """Peak List & Isotope Overlays."""
        frame = ctk.CTkScrollableFrame(self.tab_peaks)
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(frame, text="MOPF Resolved Selection", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(5, 2), anchor="w")
        self.peak_dropdown = ctk.CTkOptionMenu(frame, values=["No fits resolved yet"], command=self.on_peak_dropdown_select)
        self.peak_dropdown.pack(padx=10, pady=4, fill="x")

        ctk.CTkLabel(frame, text="Manual Entry / Formula", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 2), anchor="w")
        self.formula_entry = ctk.CTkEntry(frame, placeholder_text="Formula (C6H6+) or Mass (78.04)")
        self.formula_entry.pack(padx=10, pady=4, fill="x")

        self.add_peak_button = ctk.CTkButton(frame, text="Commit Entry to Peak List", fg_color="#ff7f0e", command=self.commit_entry_to_list)
        self.add_peak_button.pack(padx=10, pady=4, fill="x")

        ctk.CTkLabel(frame, text="List Management", font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(10, 2), anchor="w")
        
        btn_load_pl = ctk.CTkButton(frame, text="📁 Load Peak List File (.txt / .cal)", fg_color="#17a2b8", command=self._load_peak_list_file)
        btn_load_pl.pack(padx=10, pady=4, fill="x")

        self.visibility_button = ctk.CTkButton(frame, text="Hide Peak Annotations", fg_color="#6c757d", command=self.toggle_peak_visibility)
        self.visibility_button.pack(padx=10, pady=4, fill="x")

        self.clear_list_button = ctk.CTkButton(frame, text="Clear Peak List", fg_color="#dc3545", command=self.clear_peak_list)
        self.clear_list_button.pack(padx=10, pady=4, fill="x")

        self.iso_button = ctk.CTkButton(frame, text="Overlay Isotope Footprints", fg_color="#1f77b4", command=self.add_isotopes_to_spectra)
        self.iso_button.pack(padx=10, pady=4, fill="x")

    def _build_main_panel(self):
        """Builds Canvas Display + Formula DB Lookup + TS Explorer + Console Stream Panel."""
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_rowconfigure(0, weight=3)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.main_tabs = ctk.CTkTabview(self.main_container)
        self.main_tabs.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.tab_plot = self.main_tabs.add("Spectrum Display")
        self.tab_ts = self.main_tabs.add("Time-Series Explorer")
        self.tab_db = self.main_tabs.add("Formula DB & Peak Assignment")
        self.tab_diag = self.main_tabs.add("Diagnostic Figures")

        # 1. Plot Canvas Tab
        self.tab_plot.grid_columnconfigure(0, weight=1)
        self.tab_plot.grid_rowconfigure(0, weight=1)
        self.tab_plot.grid_rowconfigure(1, weight=0)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.spectrum_trace, = self.ax.plot(self.current_mass_axis, self.active_spectra_data, color="black", label="Spectrum", lw=1.2)
        self.ax.set_xlabel("Mass-to-Charge (m/z)")
        self.ax.set_ylabel("Intensity (ions/s)")
        self.ax.grid(True, linestyle=":", alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.tab_plot, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # 2. Time-Series Explorer Tab
        self._build_ts_panel()

        # 3. Database & Peak Assignment Panel
        self._build_db_panel()

        # 4. Diagnostic Viewport Tab
        self.tab_diag.grid_columnconfigure(0, weight=1)
        self.tab_diag.grid_rowconfigure(0, weight=0)
        self.tab_diag.grid_rowconfigure(1, weight=1)

        diag_ctrl = ctk.CTkFrame(self.tab_diag)
        diag_ctrl.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(diag_ctrl, text="Select Figure:").pack(side="left", padx=5)
        self.diag_dropdown = ctk.CTkOptionMenu(diag_ctrl, values=["No figures generated yet"], command=self._on_diag_dropdown_select)
        self.diag_dropdown.pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkButton(diag_ctrl, text="🔄 Refresh List", width=100, command=self._refresh_diagnostic_plots).pack(side="right", padx=5)

        self.diag_fig = Figure(figsize=(8, 5), dpi=100)
        self.diag_ax = self.diag_fig.add_subplot(111)
        self.diag_ax.text(0.5, 0.5, "Diagnostic plots generated by Deployment operations\nwill appear here.", ha="center", va="center", color="gray")
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=self.tab_diag)
        self.diag_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Console Log Panel
        self.console_frame = ctk.CTkFrame(self.main_container)
        self.console_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(10, 0))
        self.console_frame.grid_rowconfigure(1, weight=1)
        self.console_frame.grid_columnconfigure(0, weight=1)

        c_hdr = ctk.CTkFrame(self.console_frame, height=28)
        c_hdr.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        
        ctk.CTkLabel(c_hdr, text="System & Pipeline Execution Log", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(c_hdr, text="Clear Log", width=60, height=20, fg_color="#6c757d", command=self._clear_console).pack(side="right", padx=5)

        self.console_textbox = ctk.CTkTextbox(self.console_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.console_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
        self.console_textbox.configure(state="disabled")

    def _build_ts_panel(self):
        """Time-Series Explorer Tab & Interactive Selection Engine."""
        self.tab_ts.grid_columnconfigure(0, weight=1)
        self.tab_ts.grid_rowconfigure(0, weight=0)
        self.tab_ts.grid_rowconfigure(1, weight=1)

        # Control Bar
        ts_ctrl = ctk.CTkFrame(self.tab_ts)
        ts_ctrl.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(ts_ctrl, text="Target Compound/Parameter:").pack(side="left", padx=5)
        self.ts_param_dropdown = ctk.CTkComboBox(ts_ctrl, values=["Run FFI to populate time-series"], command=self._plot_selected_ts_param, width=220)
        self.ts_param_dropdown.pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkButton(ts_ctrl, text="🔄 Refresh Options", width=110, command=self._update_ts_dropdown_options).pack(side="left", padx=5)

        self.ts_extract_btn = ctk.CTkButton(ts_ctrl, text="Extract & Set Active Spectrum", fg_color="#2ca02c", command=self._extract_selected_range_spectrum)
        self.ts_extract_btn.pack(side="right", padx=5)

        # Plot Canvas
        ts_plot_frame = ctk.CTkFrame(self.tab_ts)
        ts_plot_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ts_plot_frame.grid_columnconfigure(0, weight=1)
        ts_plot_frame.grid_rowconfigure(0, weight=1)
        ts_plot_frame.grid_rowconfigure(1, weight=0)

        self.ts_fig = Figure(figsize=(8, 5), dpi=100)
        self.ts_ax = self.ts_fig.add_subplot(111)
        self.ts_ax.set_xlabel("Spectrum Index")
        self.ts_ax.set_ylabel("Intensity / Area")
        self.ts_ax.grid(True, linestyle=":", alpha=0.5)

        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, master=ts_plot_frame)
        self.ts_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.ts_toolbar = NavigationToolbar2Tk(self.ts_canvas, ts_plot_frame, pack_toolbar=False)
        self.ts_toolbar.update()
        self.ts_toolbar.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # Matplotlib SpanSelector for range highlighting
        self.ts_span = SpanSelector(
            self.ts_ax, self._on_ts_span_select, "horizontal",
            useblit=True, props=dict(alpha=0.3, facecolor="red"),
            interactive=True, drag_from_anywhere=True
        )

    def _build_db_panel(self):
        """Compound Database Search & Ionization Adduct Assignment Engine."""
        self.tab_db.grid_columnconfigure(0, weight=1)
        self.tab_db.grid_rowconfigure(1, weight=1)

        ctrl_frame = ctk.CTkFrame(self.tab_db)
        ctrl_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(ctrl_frame, text="Ionization Adduct:").pack(side="left", padx=(10, 2))
        self.ion_entry = ctk.CTkEntry(ctrl_frame, width=80)
        self.ion_entry.insert(0, "H+")
        self.ion_entry.pack(side="left", padx=5)

        ctk.CTkLabel(ctrl_frame, text="Target m/z / Formula:").pack(side="left", padx=(10, 2))
        self.target_mz_entry = ctk.CTkEntry(ctrl_frame, width=130, placeholder_text="e.g. 79.05 or C6H7+")
        self.target_mz_entry.pack(side="left", padx=5)

        search_btn = ctk.CTkButton(ctrl_frame, text="Search Candidates", width=120, command=self._search_compound_db)
        search_btn.pack(side="left", padx=10)

        assign_btn = ctk.CTkButton(ctrl_frame, text="Commit Selected to Peak List", fg_color="#2ca02c", command=self._assign_selected_db_candidate)
        assign_btn.pack(side="right", padx=10)

        table_frame = ctk.CTkFrame(self.tab_db)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        cols = ("Formula", "Compound Name", "Exact Neutral Mass", "Adduct m/z", "PPM Dev")
        self.db_tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        
        for c in cols:
            self.db_tree.heading(c, text=c)
            self.db_tree.column(c, anchor="center", width=120)
        self.db_tree.column("Compound Name", width=250, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.db_tree.yview)
        self.db_tree.configure(yscrollcommand=scrollbar.set)

        self.db_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    # =========================================================================
    # CALIBRANTS & PEAK LIST MANAGEMENT
    # =========================================================================
    def _set_manual_calibrants(self):
        """Parses manual formulas or numeric exact masses from calibrants_entry."""
        raw_input = self.calibrants_entry.get().strip()
        if not raw_input:
            print("[GUI]: Calibrants entry is empty.")
            return

        items = [s.strip() for s in raw_input.split(",") if s.strip()]
        cal_dict = {}

        for item in items:
            try:
                m_val = float(item)
                cal_dict[f"Mass_{m_val:.4f}"] = m_val
            except ValueError:
                m_val = return_mass(item)
                if m_val is not None:
                    cal_dict[item] = m_val
                else:
                    print(f"[GUI Warning]: Could not parse mass for calibrant formula '{item}'.")

        if cal_dict:
            self.calibrants_dict = cal_dict
            print(f"[GUI]: Successfully configured {len(cal_dict)} calibrants manually:")
            for k, v in cal_dict.items():
                print(f"   -> {k}: {v:.5f} m/z")
            self.render_deployment_calibrants()
        else:
            print("[GUI Error]: No valid calibrants could be parsed.")

    def _load_calibrants_file(self):
        """Loads calibrants from .cal, .txt peak lists, or CSV files using parse_calibrants_file."""
        filepath = filedialog.askopenfilename(
            title="Select Calibrants or Peak List File",
            filetypes=[("Calibrants / Peak List Files", "*.cal *.txt *.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            cal_dict = parse_calibrants_file(filepath)
            if cal_dict:
                self.calibrants_dict = cal_dict
                print(f"[GUI]: Loaded {len(cal_dict)} calibrants from '{Path(filepath).name}':")
                for k, v in cal_dict.items():
                    print(f"   -> {k}: {v:.5f} m/z")
                self.render_deployment_calibrants()
            else:
                print("[GUI Warning]: No valid calibrant entries found in file.")
        except Exception as e:
            print(f"[GUI Error]: Failed to load calibrants file: {e}")

    def _load_peak_list_file(self):
        """Loads peak list from a .txt, .cal, or .csv file into Deployment.peak_list."""
        filepath = filedialog.askopenfilename(
            title="Select Peak List File",
            filetypes=[("Peak List / Calibrants Files", "*.txt *.cal *.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            cal_dict = parse_calibrants_file(filepath)
            if cal_dict:
                peak_items = list(cal_dict.keys())
                if self.deployment is None:
                    print("[GUI Error]: Load a dataset first before attaching a peak list!")
                    return

                self.deployment.populate_peak_list_and_isotopes(peak_items)
                self.render_deployment_peak_list()
                self.canvas.draw()
                print(f"[GUI]: Successfully populated Deployment.peak_list with {len(peak_items)} entries from '{Path(filepath).name}'.")
            else:
                print("[GUI Warning]: No valid peak entries parsed from file.")
        except Exception as e:
            print(f"[GUI Error]: Failed to load peak list file: {e}")

    def _run_define_peak_list(self):
        """Pipeline step button handler for defining or loading peak list."""
        print("\n--- Pipeline Step 6: Define Peak List ---")
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        p_list = getattr(self.deployment, "peak_list", None)
        if p_list and "peaks" in p_list and len(p_list["peaks"]) > 0:
            print(f"[GUI]: Peak list currently defined with {len(p_list['peaks'])} entries.")
        else:
            print("[GUI]: Peak list is currently empty. Use the 'Peak List' tab or 'Load Peak List File' button to populate targets.")

    # =========================================================================
    # EXPORT HELPERS (STEP 7 & MINIMAL CSV)
    # =========================================================================
    def _export_workspace_h5(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        out_dir = filedialog.askdirectory(title="Select Destination Export Directory")
        if out_dir:
            self._run_in_thread(self.deployment.export_to_h5, output_dir=out_dir, driver="PIF")

    def _save_minimal_csv(self):
        if self.deployment is None or getattr(self.deployment, "peak_data", None) is None:
            print("[GUI Error]: No peak_data available. Run FFI constrained/unconstrained fitting first!")
            return

        out_file = filedialog.asksaveasfilename(
            title="Save Minimal CSV", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if not out_file:
            return

        try:
            df = self.deployment.peak_data.copy()
            df["timestamps"] = self.deployment.timestamps
            if hasattr(self.deployment, "standard_acquisition_data"):
                df["std_acq_data"] = self.deployment.standard_acquisition_data.astype(int)

            df.to_csv(out_file, index=False)
            print(f"✅ Minimal CSV saved successfully ({len(df)} rows) -> {out_file}")
        except Exception as e:
            print(f"[GUI Error]: Failed to save CSV: {e}")

    # =========================================================================
    # TIME-SERIES EXPLORER & SELECTION ENGINE
    # =========================================================================
    def _update_ts_dropdown_options(self):
        if self.deployment is None or getattr(self.deployment, "peak_data", None) is None:
            self.ts_param_dropdown.configure(values=["Run FFI to populate time-series"])
            return

        cols = [c for c in self.deployment.peak_data.columns if c not in ("MS_index", "time")]
        if cols:
            self.ts_param_dropdown.configure(values=cols)
            self.ts_param_dropdown.set(cols[0])
            self._plot_selected_ts_param(cols[0])
        else:
            self.ts_param_dropdown.configure(values=["No peak parameters found"])

    def _plot_selected_ts_param(self, choice):
        if self.deployment is None or getattr(self.deployment, "peak_data", None) is None:
            return

        df = self.deployment.peak_data
        if choice not in df.columns:
            return

        y_vals = df[choice].values
        x_vals = np.arange(len(y_vals))

        self.ts_ax.clear()
        self.ts_ax.plot(x_vals, y_vals, color="#1f77b4", lw=1.2, label=choice)
        self.ts_ax.set_xlabel("Spectrum Index")
        self.ts_ax.set_ylabel("Parameter Value")
        self.ts_ax.set_title(f"Time-Series: {choice}", fontsize=10)
        self.ts_ax.grid(True, linestyle=":", alpha=0.5)
        self.ts_ax.legend(loc="upper right")
        self.ts_canvas.draw()

    def _on_ts_span_select(self, xmin, xmax):
        idx_min = max(0, int(np.floor(xmin)))
        idx_max = min(len(self.deployment.timestamps) - 1, int(np.ceil(xmax)))
        self.selected_ts_indices = np.arange(idx_min, idx_max + 1)
        print(f"[GUI TS Explorer]: Range selected -> Spectrum Indices [{idx_min} to {idx_max}] ({len(self.selected_ts_indices)} spectra)")

    def _extract_selected_range_spectrum(self):
        if self.deployment is None or self.selected_ts_indices is None or len(self.selected_ts_indices) == 0:
            print("[GUI Error]: Highlight a valid spectrum index range on the Time-Series plot first.")
            return

        print(f"[GUI]: Extracting out-of-core mean spectrum for {len(self.selected_ts_indices)} spectra...")

        def extractor():
            try:
                sampled_data = self.deployment.tofdata[self.selected_ts_indices, :].compute()
                avg_spec = np.asarray(np.mean(sampled_data, axis=0), dtype=float)

                idx_start = self.selected_ts_indices[0]
                idx_end = self.selected_ts_indices[-1]
                key_str = f"{idx_start}_{idx_end}"

                if not hasattr(self.deployment, "interactive") or not isinstance(self.deployment.interactive, dict):
                    self.deployment.interactive = {}

                self.deployment.interactive[key_str] = avg_spec
                print(f"[GUI]: Saved extracted spectrum to Deployment.interactive['{key_str}']")

                # Set active spectrum in GUI
                self.current_spectra = avg_spec
                self.active_spectra_data = avg_spec.copy()
                self.showing_subtracted = False

                self.after(0, self._update_stream_menu_options)
                self.after(0, lambda: self.stream_menu.set(f"Extracted: {key_str}"))
                self.after(0, lambda: self.main_tabs.set("Spectrum Display"))
                self.after(0, self.update_plot_trace)
            except Exception as e:
                print(f"[GUI Error]: Spectrum extraction failed: {e}")

        threading.Thread(target=extractor, daemon=True).start()

    # =========================================================================
    # DATA LOADING DIALOG ENGINE
    # =========================================================================
    def _load_from_directory_dialog(self):
        target_dir = filedialog.askdirectory(title="Select Target HDF5 Acquisition Directory")
        if not target_dir:
            return
        
        print(f"\n--- Loading Deployment from Directory: {target_dir} ---")
        
        def loader():
            try:
                dep = Deployment.from_directory(target_dir)
                self.after(0, lambda: self._on_dataset_loaded(dep))
            except Exception as e:
                print(f"[GUI Error]: Failed to load directory: {e}")

        threading.Thread(target=loader, daemon=True).start()

    def _load_single_file_dialog(self):
        target_file = filedialog.askopenfilename(title="Select Raw HDF5 Acquisition File", filetypes=[("HDF5 Files", "*.h5")])
        if not target_file:
            return

        print(f"\n--- Loading Deployment from File: {target_file} ---")

        def loader():
            try:
                dep = Deployment.single_file(target_file)
                self.after(0, lambda: self._on_dataset_loaded(dep))
            except Exception as e:
                print(f"[GUI Error]: Failed to load file: {e}")

        threading.Thread(target=loader, daemon=True).start()

    def _load_ot_state_dialog(self):
        target_file = filedialog.askopenfilename(title="Select OpenTof State File", filetypes=[("OpenTof State", "*.h5")])
        if not target_file:
            return

        print(f"\n--- Loading Deployment State from: {target_file} ---")

        def loader():
            try:
                dep = Deployment.from_ot_h5(target_file)
                self.after(0, lambda: self._on_dataset_loaded(dep))
            except Exception as e:
                print(f"[GUI Error]: Failed to load state file: {e}")

        threading.Thread(target=loader, daemon=True).start()

    def _on_dataset_loaded(self, new_deployment):
        """Callback executed on GUI thread when a new Deployment object finishes loading."""
        self.deployment = new_deployment
        self._initialize_data_sources()
        
        self.ax.set_xlim(self.current_mass_axis.min(), self.current_mass_axis.max())
        self.update_plot_trace()
        self.check_and_plot_existing_baseline()
        self.render_deployment_peak_list()
        self.render_deployment_calibrants()
        self._update_ts_dropdown_options()
        self._update_stream_menu_options()
        print("✅ Dataset successfully attached to SpectrumWizardGUI!")

    # =========================================================================
    # BASELINE & CALIBRANT PLOTTING
    # =========================================================================
    def check_and_plot_existing_baseline(self):
        """Renders baseline trace if present on deployment object."""
        if getattr(self, "deployment", None) is not None and getattr(self.deployment, "baseline", None) is not None:
            adj_b = self.deployment.baseline["adjusted_baseline"]
            if len(adj_b) == len(self.current_mass_axis):
                if self.baseline_trace is not None:
                    self.baseline_trace.remove()
                self.baseline_trace, = self.ax.plot(
                    self.current_mass_axis, adj_b, color="#9467bd", linestyle="-.", label="Baseline Floor", lw=1.0
                )
                self.ax.legend(loc="upper right")
                self.canvas.draw()

    def render_deployment_calibrants(self):
        """Renders configured calibrant positions as vertical dashed red lines."""
        for item in self.calibrant_elements:
            item.remove()
        self.calibrant_elements.clear()

        if not self.calibrants_dict:
            return

        ymin, ymax = self.ax.get_ylim()
        view_height = ymax - ymin

        for name, m_val in self.calibrants_dict.items():
            try:
                idx = np.argmin(np.abs(self.current_mass_axis - m_val))
                local_signal_height = self.active_spectra_data[idx]
                text_y_position = local_signal_height + (view_height * 0.05)

                line = self.ax.axvline(x=m_val, color="red", alpha=0.5, linestyle="--", lw=1.0)
                txt = self.ax.text(m_val, text_y_position, f" {name}", color="red", alpha=0.7, rotation=90, fontsize=8)
                self.calibrant_elements.extend([line, txt])
            except Exception:
                continue
        self.canvas.draw()

    # =========================================================================
    # DIAGNOSTIC PLOT RENDERING & DROPDOWN
    # =========================================================================
    def _refresh_diagnostic_plots(self, selected_filename=None):
        """Scans plot directories for generated diagnostic PNGs and displays the selected or newest one."""
        plot_dir = getattr(self.deployment, "plot_dir", None) or get_default_plot_dir()
        subdir = getattr(self.deployment, "default_plot_subdir", "")
        search_dirs = [Path(plot_dir) / subdir, Path(plot_dir), Path.cwd()]

        png_dict = {}
        for d in search_dirs:
            if d.exists():
                for p in d.glob("*.png"):
                    png_dict[p.name] = p

        if not png_dict:
            self.diag_dropdown.configure(values=["No figures generated yet"])
            return

        filenames = sorted(list(png_dict.keys()))
        self.diag_dropdown.configure(values=filenames)

        if selected_filename and selected_filename in png_dict:
            target_path = png_dict[selected_filename]
            self.diag_dropdown.set(selected_filename)
        else:
            target_path = max(png_dict.values(), key=lambda p: p.stat().st_mtime)
            self.diag_dropdown.set(target_path.name)

        try:
            img = mpimg.imread(str(target_path))
            self.diag_ax.clear()
            self.diag_ax.imshow(img)
            self.diag_ax.axis("off")
            self.diag_ax.set_title(f"Diagnostic Figure: {target_path.name}", fontsize=10)
            self.diag_canvas.draw()
            print(f"[GUI]: Diagnostic plot rendered -> {target_path.name}")
        except Exception as e:
            print(f"[GUI Notice]: Could not display diagnostic figure: {e}")

    def _on_diag_dropdown_select(self, choice):
        if choice and choice != "No figures generated yet":
            self._refresh_diagnostic_plots(selected_filename=choice)

    # =========================================================================
    # DEPLOYMENT METHOD THREAD RUNNERS
    # =========================================================================
    def _run_in_thread(self, target_func, *args, **kwargs):
        """Helper to run heavy analysis methods in a background thread."""
        if self.deployment is None:
            print("[GUI Error]: No active Deployment object loaded. Load a dataset first!")
            return

        def worker():
            try:
                sig = inspect.signature(target_func).parameters
                has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())

                if "show_plot_flag" in sig or has_var_kwargs:
                    kwargs["show_plot_flag"] = False
                if "plot_flag" in sig or has_var_kwargs:
                    if "plot_flag" not in kwargs:
                        kwargs["plot_flag"] = True

                if has_var_kwargs:
                    filtered_kwargs = kwargs.copy()
                    if target_func.__name__ in ("FFI_constrained", "FFI_unconstrained"):
                        filtered_kwargs.pop("show_plot_flag", None)
                        filtered_kwargs.pop("plot_flag", None)
                        filtered_kwargs.pop("chunk_size", None)
                else:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig}
                
                target_func(*args, **filtered_kwargs)
                self.after(0, self.update_plot_trace)
                self.after(0, self.check_and_plot_existing_baseline)
                
                # Only refresh diagnostic plots for steps that generate PNG figures
                if target_func.__name__ not in ("FFI_constrained", "FFI_unconstrained", "export_to_h5"):
                    self.after(0, self._refresh_diagnostic_plots)

                self.after(0, self._update_ts_dropdown_options)
                self.after(0, self._update_stream_menu_options)
            except Exception as e:
                print(f"[Error in {target_func.__name__}]: {e}")
        
        threading.Thread(target=worker, daemon=True).start()

    def _run_mass_cal(self):
        print("\n--- Executing Deployment.mass_calibration ---")
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        cal_to_use = getattr(self, "calibrants_dict", None)

        if cal_to_use is None and hasattr(self, "calibrants_entry") and self.calibrants_entry.get().strip():
            self._set_manual_calibrants()
            cal_to_use = getattr(self, "calibrants_dict", None)

        if cal_to_use is None and self.deployment.calibration is None:
            print("[GUI Error]: Mass calibration requires calibrants. Set manual formulas or load a .cal file first!")
            return

        kwargs = self.pipeline_kwargs["mass_calibration"].copy()
        self._run_in_thread(self.deployment.mass_calibration, calibrants=cal_to_use, **kwargs)

    def _run_ref_spec(self):
        print("\n--- Executing Deployment.determine_reference_spectrum ---")
        kwargs = self.pipeline_kwargs["determine_reference_spectrum"].copy()
        
        ref_m_raw = str(kwargs.get("reference_peak_mass", "")).strip()
        if ref_m_raw and ref_m_raw.lower() != "none":
            try:
                kwargs["reference_peak_mass"] = float(ref_m_raw)
            except ValueError:
                parsed = return_mass(ref_m_raw)
                if parsed is not None:
                    kwargs["reference_peak_mass"] = parsed
                else:
                    kwargs.pop("reference_peak_mass", None)
        else:
            kwargs["reference_peak_mass"] = None

        self._run_in_thread(self.deployment.determine_reference_spectrum, **kwargs)

    def _run_baseline_cal(self):
        print("\n--- Executing Deployment.determine_baseline ---")
        kwargs = self.pipeline_kwargs["determine_baseline"].copy()
        
        ws_raw = str(kwargs.get("window_size", "")).strip()
        if ws_raw and ws_raw.lower() != "none" and ws_raw.isdigit():
            kwargs["window_size"] = int(ws_raw)
        else:
            kwargs["window_size"] = None

        self._run_in_thread(self.deployment.determine_baseline, **kwargs)

    def _run_peak_width(self):
        print("\n--- Executing Deployment.determine_peak_width ---")
        kwargs = self.pipeline_kwargs["determine_peak_width"].copy()

        pw_mode = kwargs['mode']
        if pw_mode == 'ransac':
            kwargs.pop("top_n", None)
            kwargs.pop("max_num_outlires", None)
        else:
            kwargs.pop("min_prominence", None)
            kwargs.pop("residual_threshold_multiplier", None)

        self._run_in_thread(self.deployment.determine_peak_width, **kwargs)

    def _run_peak_shape(self):
        print("\n--- Executing Deployment.determine_peak_shape ---")
        kwargs = self.pipeline_kwargs["determine_peak_shape"].copy()
        self._run_in_thread(self.deployment.determine_peak_shape, **kwargs)

    def _run_ffi_constrained(self):
        print("\n--- Executing Deployment.FFI_constrained ---")
        kwargs = self.pipeline_kwargs["FFI_constrained"].copy()
        self._run_in_thread(self.deployment.FFI_constrained, **kwargs)

    def _run_ffi_unconstrained(self):
        print("\n--- Executing Deployment.FFI_unconstrained ---")
        kwargs = self.pipeline_kwargs["FFI_unconstrained"].copy()
        self._run_in_thread(self.deployment.FFI_unconstrained, **kwargs)

    # =========================================================================
    # DB SEARCH & IONIZATION LOGIC
    # =========================================================================
    def _get_ionization_mass(self, ion_str):
        """Calculates mass shift based on ionization adduct formula."""
        if not ion_str:
            return 0.0
        try:
            return return_mass(ion_str)
        except Exception:
            defaults = {"H+": 1.007276, "H": 1.007276, "I-": 126.90447, "NH4+": 18.033826, "Na+": 22.989769}
            return defaults.get(ion_str, 0.0)

    def _search_compound_db(self):
        """Searches peak_assignment_db for candidate neutral formulas matching target m/z + ionization."""
        raw_mz = self.target_mz_entry.get().strip()
        if not raw_mz:
            if self.selected_mopf_mass:
                target_mz = self.selected_mopf_mass
                self.target_mz_entry.delete(0, "end")
                self.target_mz_entry.insert(0, f"{target_mz:.4f}")
            else:
                print("[GUI]: Enter a target m/z or formula, or select a fitted MOPF peak first.")
                return
        else:
            try:
                target_mz = float(raw_mz)
            except ValueError:
                parsed_m = return_mass(raw_mz)
                if parsed_m is not None:
                    target_mz = parsed_m
                else:
                    print(f"[GUI Error]: Could not parse '{raw_mz}' as float or valid chemical formula.")
                    return

        ion_str = self.ion_entry.get().strip()
        ion_mass = self._get_ionization_mass(ion_str)
        target_neutral_mass = target_mz - ion_mass

        print(f"\n--- Searching Database ---")
        print(f"Target m/z: {target_mz:.4f} | Ionization: '{ion_str}' ({ion_mass:.4f} Da) | Neutral Mass: {target_neutral_mass:.4f} Da")

        db = getattr(self.deployment, "peak_assignment_db", None) if self.deployment else None
        if db is None or db.empty:
            print("[GUI]: Deployment.peak_assignment_db is empty or not loaded.")
            return

        mask = (db["ExactMass"] >= target_neutral_mass - 0.5) & (db["ExactMass"] <= target_neutral_mass + 0.5)
        matches = db[mask].copy()

        if matches.empty:
            print("[GUI]: No candidates found within +/- 0.5 Da window.")
            self._clear_tree()
            return

        matches["AdductMass"] = matches["ExactMass"] + ion_mass
        matches["PPM_Dev"] = ((matches["AdductMass"] - target_mz) / target_mz) * 1e6
        matches["Abs_PPM"] = matches["PPM_Dev"].abs()
        matches = matches.sort_values("Abs_PPM").head(30)

        self._clear_tree()
        for _, row in matches.iterrows():
            formula_col = row.get("MolecularFormula", row.get("Formula", row.get("formula", "")))
            name_col = row.get("CompoundName", row.get("name", row.get("Name", "-")))
            if not name_col or pd.isna(name_col):
                name_col = "-"

            exact_m = row["ExactMass"]
            adduct_m = row["AdductMass"]
            ppm = row["PPM_Dev"]

            self.db_tree.insert("", "end", values=(formula_col, name_col, f"{exact_m:.5f}", f"{adduct_m:.5f}", f"{ppm:+.2f}"))

        print(f"[GUI]: Displaying top {len(matches)} formula candidates.")

    def _assign_selected_db_candidate(self):
        """Commits selected database formula candidate (plus adduct) to Deployment.peak_list."""
        sel = self.db_tree.selection()
        if not sel:
            print("[GUI]: No candidate selected in the database table.")
            return

        item = self.db_tree.item(sel[0])["values"]
        neutral_formula = str(item[0]).strip()

        if not neutral_formula or neutral_formula in ("Unknown", "nan", "None", "-"):
            print("[GUI Error]: Selected entry has no valid chemical formula.")
            return

        ion_str = self.ion_entry.get().strip()
        combined_formula = f"{neutral_formula}{ion_str}" if ion_str else neutral_formula
        self.formula_entry.delete(0, "end")
        self.formula_entry.insert(0, combined_formula)
        self.commit_entry_to_list()

    def _clear_tree(self):
        for item in self.db_tree.get_children():
            self.db_tree.delete(item)

    # =========================================================================
    # UTILITY & EVENT HANDLERS
    # =========================================================================
    def _update_stream_menu_options(self):
        """Refreshes Data Stream Source options including extracted interactive spectra."""
        base_options = ["Reference Spectrum", "Global Averaged (GAD)"]
        interactive_keys = []
        if self.deployment is not None and getattr(self.deployment, "interactive", None) is not None:
            if isinstance(self.deployment.interactive, dict):
                interactive_keys = [f"Extracted: {k}" for k in self.deployment.interactive.keys()]
        
        all_options = base_options + interactive_keys
        current_val = self.stream_menu.get()
        self.stream_menu.configure(values=all_options)
        if current_val in all_options:
            self.stream_menu.set(current_val)

    def _change_data_stream(self, choice):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        if choice == "Global Averaged (GAD)":
            if getattr(self.deployment, "_averaged_dataset", None) is not None:
                self.current_spectra = np.mean(self.deployment._averaged_dataset[0], axis=0)
                print("[GUI]: Stream switched to Global Averaged Dataset (GAD).")
            else:
                print("Notice: GAD dataset uncalculated. Call generate_averaged_dataset() first.")
                self.stream_menu.set("Reference Spectrum")
                return
        elif choice.startswith("Extracted: "):
            key = choice.replace("Extracted: ", "")
            if (hasattr(self.deployment, "interactive") and 
                isinstance(self.deployment.interactive, dict) and 
                key in self.deployment.interactive):
                self.current_spectra = np.asarray(self.deployment.interactive[key], dtype=float)
                print(f"[GUI]: Stream switched to Extracted Spectrum '{key}'.")
            else:
                print(f"[GUI Error]: Extracted spectrum '{key}' not found in Deployment.interactive.")
                return
        else:
            if getattr(self.deployment, "reference", None) is not None:
                self.current_spectra = np.asarray(self.deployment.reference['reference_spectrum'], dtype=float)
                print("[GUI]: Stream switched to Reference Spectrum.")

        self.active_spectra_data = self.current_spectra.copy()
        self.showing_subtracted = False
        self.update_plot_trace()

    def open_mopf_settings(self):
        MopfSettingsWindow(self)

    def render_deployment_peak_list(self):
        """Renders assigned peaks onto the plot canvas."""
        for item in self.peak_list_elements:
            item.remove()
        self.peak_list_elements.clear()

        if not self.show_peak_list or self.deployment is None:
            return

        p_list = getattr(self.deployment, "peak_list", None)
        if not p_list or "peaks" not in p_list or len(p_list["peaks"]) == 0:
            return

        ymin, ymax = self.ax.get_ylim()
        view_height = ymax - ymin

        for entry in p_list["peaks"]:
            try:
                if isinstance(entry, (int, float)):
                    center_m = entry
                else:
                    idx_entry = list(p_list["peaks"]).index(entry)
                    center_m = p_list["centers"][idx_entry]
                
                label_str = str(entry)
                idx = np.argmin(np.abs(self.current_mass_axis - center_m))
                local_signal_height = self.active_spectra_data[idx]
                text_y_position = local_signal_height + (view_height * 0.03)

                line = self.ax.axvline(x=center_m, color="blue", alpha=0.3, linestyle="--", lw=1.0)
                txt = self.ax.text(center_m, text_y_position, f" {label_str}", color="blue", alpha=0.6, rotation=90, fontsize=8)
                self.peak_list_elements.extend([line, txt])
            except Exception:
                continue

    def toggle_log_scale(self):
        self.update_plot_trace()

    def toggle_baseline(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        if self.baseline_trace is not None:
            self.baseline_trace.remove()
            self.baseline_trace = None
            self.ax.legend(loc="upper right")
            self.canvas.draw()
            return

        ws_input = self.window_size_entry.get().strip()
        ws_param = int(ws_input) if ws_input.isdigit() else None

        raw_spectrum = getattr(self.deployment, "reference", {}).get("reference_spectrum", self.current_spectra)
        if getattr(self.deployment, "baseline", None) is None or ws_param is not None:
            self.deployment.determine_baseline(target_spectrum=raw_spectrum, window_size=ws_param, plot_flag=False)

        adj_b = self.deployment.baseline["adjusted_baseline"]
        self.baseline_trace, = self.ax.plot(self.current_mass_axis, adj_b, color="#9467bd", linestyle="-.", label="Baseline Floor", lw=1.0)
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def toggle_subtraction(self):
        if self.deployment is None or getattr(self.deployment, "baseline", None) is None:
            print("[GUI]: Compute a baseline floor first before subtracting.")
            return

        adj_b = self.deployment.baseline["adjusted_baseline"]
        if not self.showing_subtracted:
            self.active_spectra_data = self.current_spectra - adj_b
            self.showing_subtracted = True
            self.subtraction_button.configure(text="Restore Raw Trace", fg_color="#d62728")
        else:
            self.active_spectra_data = self.current_spectra.copy()
            self.showing_subtracted = False
            self.subtraction_button.configure(text="Apply Baseline Subtraction", fg_color="#9467bd")

        self.update_plot_trace()

    def run_mopf(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        pw_func = getattr(self.deployment, "peak_width_function", None)
        if pw_func is None:
            print("[GUI]: peak_width_function is undefined. Run determine_peak_width() first.")
            return

        xmin, xmax = self.ax.get_xlim()
        mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
        
        if np.sum(mask) < 5:
            print("[GUI]: Zoom window too narrow.")
            return

        for elem in self.mopf_elements:
            elem.remove()
        self.mopf_elements.clear()
        if self.fit_trace:
            self.fit_trace.remove()
            self.fit_trace = None

        mz_seg = self.current_mass_axis[mask]
        int_seg = self.active_spectra_data[mask] 
        tof_seg = self.deployment.tof_axis[mask] if getattr(self.deployment, "tof_axis", None) is not None else np.arange(len(mz_seg), dtype=float)

        min_sep = pw_func(np.median(mz_seg))
        peak_type = "custom" if getattr(self.deployment, "custom_peak_shape", None) is not None else "gaussian"
        custom_shape = getattr(self.deployment, "custom_peak_shape", None)

        fit_res = multi_overlap_peak_fit(
            mass_axis=mz_seg, intensity_axis=int_seg, tof_axis=tof_seg,
            peak_type=peak_type, custom_shape=custom_shape, min_separation_fwhm=min_sep,
            plot_flag=False, verbose=False, **self.mopf_params
        )

        if fit_res is None or len(fit_res.get("peaks", [])) == 0:
            print("[GUI]: No peaks resolved in window.")
            self.canvas.draw()
            return

        dropdown_options = []
        resolved_func = peak_function_selector(peak_type, custom_shape=custom_shape)
        n_p = 4 if peak_type == "pseudo_voigt" else 3
        composite = np.zeros_like(tof_seg)

        print(f"\n--- MOPF Zoom Window Fit Results ({len(fit_res['peaks'])} peaks) ---")
        for i, peak in enumerate(fit_res["peaks"]):
            m_center = peak["center_mass"]
            amp = peak.get("amplitude", 0.0)
            area = peak.get("area", 0.0)
            dropdown_options.append(f"Peak {i+1}: {m_center:.5f}")
            print(f"  Peak {i+1:2d}: m/z = {m_center:.5f}  |  Amp = {amp:.2e}  |  Area = {area:.2e}")

            v_line = self.ax.axvline(x=m_center, color="grey", linestyle=":", alpha=0.4)
            self.mopf_elements.append(v_line)

            p_vals = fit_res["popt"][i * n_p : (i + 1) * n_p]
            peak_curve = resolved_func(tof_seg, *p_vals)
            composite += peak_curve

            indiv_trace, = self.ax.plot(mz_seg, peak_curve, linestyle="--", lw=1.0, alpha=0.6)
            self.mopf_elements.append(indiv_trace)

        print("--------------------------------------------------")

        self.fit_trace, = self.ax.plot(mz_seg, composite, color="#2ca02c", lw=2, label="MOPF Fit")
        self.peak_dropdown.configure(values=dropdown_options)
        self.peak_dropdown.set(dropdown_options[0])
        self.selected_mopf_mass = fit_res["peaks"][0]["center_mass"]

        self.target_mz_entry.delete(0, "end")
        self.target_mz_entry.insert(0, f"{self.selected_mopf_mass:.4f}")

        self.ax.legend(loc="upper right")
        self.canvas.draw()
        print(f"[GUI]: Resolved {len(fit_res['peaks'])} overlapping peaks in window.")

    def on_peak_dropdown_select(self, choice):
        try:
            self.selected_mopf_mass = float(choice.split(": ")[1])
            self.target_mz_entry.delete(0, "end")
            self.target_mz_entry.insert(0, f"{self.selected_mopf_mass:.4f}")
        except Exception:
            self.selected_mopf_mass = None

    def commit_entry_to_list(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        user_input = self.formula_entry.get().strip()
        if not user_input:
            print("[GUI]: Entry cannot be blank.")
            return

        if not hasattr(self.deployment, "peak_list") or self.deployment.peak_list is None:
            self.deployment.peak_list = {"peaks": [], "centers": np.array([]), "fwhms": np.array([])}

        current_peaks = list(self.deployment.peak_list["peaks"])
        
        try:
            target_value = float(user_input)
        except ValueError:
            target_value = user_input

        if target_value not in current_peaks:
            current_peaks.append(target_value)
            self.deployment.populate_peak_list_and_isotopes(current_peaks)
            self.render_deployment_peak_list()
            self.canvas.draw()
            print(f"[GUI]: Added {target_value} to Deployment.peak_list.")

    def add_isotopes_to_spectra(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return

        p_list = getattr(self.deployment, "peak_list", None)
        if not p_list or "peaks" not in p_list or len(p_list["peaks"]) == 0:
            print("[GUI]: No peaks defined in Deployment.peak_list.")
            return

        xmin, xmax = self.ax.get_xlim()
        mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
        mz_seg = self.current_mass_axis[mask]

        total_isotope_profile = np.zeros_like(mz_seg, dtype=float)
        pw_func = getattr(self.deployment, "peak_width_function", lambda m: max(0.005, m / 2000.0))
        peak_type = "custom" if getattr(self.deployment, "custom_peak_shape", None) is not None else "gaussian"
        custom_shape = getattr(self.deployment, "custom_peak_shape", None)

        for entry in p_list["peaks"]:
            if isinstance(entry, str):
                try:
                    idx_entry = list(p_list["peaks"]).index(entry)
                    center_m = p_list["centers"][idx_entry]
                    idx = np.argmin(np.abs(self.current_mass_axis - center_m))
                    parent_amplitude = self.active_spectra_data[idx]
                    
                    if parent_amplitude > 0:
                        profile = isotope_signal_on_axis(
                            formula=entry, mass_axis=mz_seg, parent_amplitude=parent_amplitude,
                            peak_width_function=pw_func, peak_type=peak_type, custom_shape=custom_shape
                        )
                        total_isotope_profile += profile
                except Exception:
                    continue

        if self.iso_trace:
            self.iso_trace.remove()

        self.iso_trace, = self.ax.plot(mz_seg, total_isotope_profile, color="#d62728", lw=1.5, linestyle="--", label="Isotopes")
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def update_plot_trace(self):
        """Updates Matplotlib trace coordinates for both x and y to prevent shape mismatches."""
        if self.spectrum_trace is None:
            return

        if self.log_checkbox.get() == 1:
            floor_val = 1e-3
            display_y = np.where(self.active_spectra_data < floor_val, floor_val, self.active_spectra_data)
            self.spectrum_trace.set_xdata(self.current_mass_axis)
            self.spectrum_trace.set_ydata(display_y)
            self.ax.set_yscale("log")
            self.ax.set_ylim(bottom=floor_val, top=np.max(display_y) * 3)
        else:
            self.spectrum_trace.set_xdata(self.current_mass_axis)
            self.spectrum_trace.set_ydata(self.active_spectra_data)
            self.ax.set_yscale("linear")
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True)

        self.render_deployment_peak_list()
        self.render_deployment_calibrants()
        self.canvas.draw()

    def run_global_peak_discovery(self):
        if self.deployment is None:
            print("[GUI Error]: Load a dataset first!")
            return
        print("[GUI]: Starting global peak discovery...")
        self._run_in_thread(self.deployment.automated_peak_discovery, overwrite=True)

    def auto_y_scale(self):
        xmin, xmax = self.ax.get_xlim()
        mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
        if np.sum(mask) == 0:
            return

        seg_y = self.active_spectra_data[mask]
        max_val = np.max(seg_y)

        if self.log_checkbox.get() == 1:
            floor_val = 1e-3
            self.ax.set_ylim(bottom=floor_val, top=max(floor_val * 10, max_val * 3))
        else:
            min_val = np.min(seg_y)
            padding = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
            self.ax.set_ylim(bottom=min_val - padding, top=max_val + padding)

        self.canvas.draw()

    def toggle_peak_visibility(self):
        self.show_peak_list = not self.show_peak_list
        self.render_deployment_peak_list()
        self.canvas.draw()

    def clear_peak_list(self):
        if self.deployment is None:
            return
        self.deployment.peak_list = {"peaks": [], "centers": np.array([]), "fwhms": np.array([])}
        self.render_deployment_peak_list()
        self.canvas.draw()
        print("[GUI]: Peak list cleared.")

    def _clear_console(self):
        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("1.0", "end")
        self.console_textbox.configure(state="disabled")

    def close_and_resume(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.destroy()


def launch_wizard(deployment_obj=None, spectra=None, spectra_mass_axis=None, initial_mz_zoom=None):
    """Standalone launcher helper for launching SpectrumWizardGUI."""
    app = SpectrumWizardGUI(
        deployment_obj=deployment_obj, 
        spectra=spectra, 
        spectra_mass_axis=spectra_mass_axis, 
        initial_mz_zoom=initial_mz_zoom
    )
    app.mainloop()
    return app