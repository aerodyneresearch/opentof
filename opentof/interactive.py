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
import numpy as np
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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
)

# Set the global theme look
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 


class MopfSettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("MOPF Optimization Parameters")
        self.geometry("450x550")
        self.transient(parent)
        self.grab_set()

        # Title
        lbl = ctk.CTkLabel(self, text="Configure Fitting Engine", font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(padx=15, pady=15)

        # Scrollable area for parameters
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Build entry dictionary for tracking configurations
        self.entries = {}
        
        # Define parameters to adjust (key, default_value, type)
        params = [
            ("noise_std_thres", self.parent.mopf_params.get("noise_std_thres", 10.0), float),
            ("max_iter", self.parent.mopf_params.get("max_iter", 1000), int),
            ("tol", self.parent.mopf_params.get("tol", 1e-8), float),
            ("deriv_threshold", self.parent.mopf_params.get("deriv_threshold", 0.0), float),
            ("peaks_to_add_post_fit", self.parent.mopf_params.get("peaks_to_add_post_fit", 0), int),
            ("abs_residual_threshold", self.parent.mopf_params.get("abs_residual_threshold", 0.0), float),
            ("peak_width_constraint_factor", self.parent.mopf_params.get("peak_width_constraint_factor", 5.0), float),
        ]

        for key, val, t_type in params:
            frame = ctk.CTkFrame(self.scroll)
            frame.pack(fill="x", padx=5, pady=5)
            
            lbl_p = ctk.CTkLabel(frame, text=key, font=ctk.CTkFont(size=12))
            lbl_p.pack(side="left", padx=10, pady=5)
            
            ent = ctk.CTkEntry(frame, width=140)
            ent.insert(0, str(val))
            ent.pack(side="right", padx=10, pady=5)
            
            self.entries[key] = (ent, t_type)

        # Action Buttons
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
                pass # Gracefully retain prior defaults if parsing drops empty or corrupt strings
        self.parent.status_label.configure(text="✅ MOPF configurations successfully applied.")
        self.destroy()


class SpectrumWizardGUI(ctk.CTk):
    def __init__(self, deployment_obj, spectra, spectra_mass_axis=None, initial_mz_zoom=None):
        super().__init__()
        
        # Core data streams
        self.deployment = deployment_obj
        self.current_spectra = np.asarray(spectra, dtype=float)
        self.current_mass_axis = np.asarray(spectra_mass_axis, dtype=float)

        if spectra_mass_axis is None:
            self.current_mass_axis = self.deployment.first_guess_mass_axis
        
        self.showing_subtracted = False
        self.active_spectra_data = self.current_spectra.copy()
        
        # Overlay graphic tracking references
        self.peak_list_elements = []  
        self.mopf_elements = []       
        self.spectrum_trace = None    
        self.fit_trace = None
        self.iso_trace = None
        self.baseline_trace = None
        
        self.selected_mopf_mass = None
        self.show_peak_list = True

        # Default MOPF dictionary storage
        self.mopf_params = {
            "noise_std_thres": 10.0,
            "max_iter": 1000,
            "tol": 1e-8,
            "deriv_threshold": 0.0,
            "peaks_to_add_post_fit": 0,
            "abs_residual_threshold": 0.0,
            "peak_width_constraint_factor": 5.0
        }

        # Configure Window Dimensions & Adaptive Target Scaling boundaries
        self.title("OpenTof Processing Wizard")
        
        # Calculate maximum limits based on monitor size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        target_w = min(1600, int(screen_w * 0.9))
        target_h = min(750, int(screen_h * 0.8))
        self.geometry(f"{target_w}x{target_h}")
        
        # Responsive Layout Grid
        self.grid_columnconfigure(0, weight=0, minsize=310)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SCROLLABLE SIDEBAR CONTROL FRAME ---
        self.sidebar_frame = ctk.CTkScrollableFrame(self, corner_radius=0, width=300)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame, text="OpenTof Wizard", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(padx=15, pady=15)

        # SECTION 1: Configurable Baseline Tools
        self.base_lbl = ctk.CTkLabel(self.sidebar_frame, text="Baseline Tools", font=ctk.CTkFont(size=13, weight="bold"))
        self.base_lbl.pack(padx=15, pady=(10, 5), anchor="w")

        self.window_size_entry = ctk.CTkEntry(
            self.sidebar_frame, 
            placeholder_text="Window Size (Optional - e.g., 31)"
        )
        self.window_size_entry.pack(padx=15, pady=5, fill="x")

        self.baseline_button = ctk.CTkButton(
            self.sidebar_frame, text="Toggle/Compute Baseline Floor", 
            fg_color="#2ca02c", hover_color="#218021", command=self.toggle_baseline
        )
        self.baseline_button.pack(padx=15, pady=5, fill="x")

        self.subtraction_button = ctk.CTkButton(
            self.sidebar_frame, text="Apply Subtraction to Plot", 
            fg_color="#9467bd", hover_color="#7a4f9e", command=self.toggle_subtraction
        )
        self.subtraction_button.pack(padx=15, pady=5, fill="x")

        # Display Customization Section (Log Scale Checkbox)
        self.display_lbl = ctk.CTkLabel(self.sidebar_frame, text="Display Settings", font=ctk.CTkFont(size=13, weight="bold"))
        self.display_lbl.pack(padx=15, pady=(15, 5), anchor="w")

        self.log_checkbox = ctk.CTkCheckBox(
            self.sidebar_frame, text="Use Log Scale (Y-axis)", command=self.toggle_log_scale
        )
        self.log_checkbox.pack(padx=15, pady=5, fill="x")

        self.autoy_button = ctk.CTkButton(
            self.sidebar_frame, text="Auto-Y Scale to Zoom",
            fg_color="#ff00ea", hover_color="#c200b2", command=self.auto_y_scale
        )
        self.autoy_button.pack(padx=15, pady=5, fill="x")

        # SECTION 2: MOPF Fitting Controls
        self.fit_lbl = ctk.CTkLabel(self.sidebar_frame, text="Peak Fitting Options", font=ctk.CTkFont(size=13, weight="bold"))
        self.fit_lbl.pack(padx=15, pady=(20, 5), anchor="w")

        self.fit_button = ctk.CTkButton(self.sidebar_frame, text="Run MOPF on Zoom Window", command=self.run_mopf)
        self.fit_button.pack(padx=15, pady=5, fill="x")

        self.config_mopf_button = ctk.CTkButton(
            self.sidebar_frame, text="Adjust MOPF Parameters", 
            fg_color="#1f77b4", hover_color="#155d8b", command=self.open_mopf_settings
        )
        self.config_mopf_button.pack(padx=15, pady=5, fill="x")

        self.discovery_button = ctk.CTkButton(
            self.sidebar_frame, text="Automated Peak Discovery", 
            fg_color="#17a2b8", hover_color="#117a8b", command=self.run_global_peak_discovery
        )
        self.discovery_button.pack(padx=15, pady=5, fill="x")

        # SECTION 3: Compound & Mass Assignment Engine
        self.assign_lbl = ctk.CTkLabel(self.sidebar_frame, text="Peak List Management", font=ctk.CTkFont(size=13, weight="bold"))
        self.assign_lbl.pack(padx=15, pady=(25, 5), anchor="w")

        self.peak_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame, values=["No fits resolved yet"], command=self.on_peak_dropdown_select
        )
        self.peak_dropdown.pack(padx=15, pady=5, fill="x")

        self.formula_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Formula (C6H6+) or Mass (78.04)")
        self.formula_entry.pack(padx=15, pady=5, fill="x")

        self.add_peak_button = ctk.CTkButton(
            self.sidebar_frame, text="Commit to Peak List", 
            fg_color="#ff7f0e", hover_color="#d6680a", command=self.commit_entry_to_list
        )
        self.add_peak_button.pack(padx=15, pady=5, fill="x")

        self.visibility_button = ctk.CTkButton(
            self.sidebar_frame, text="Hide Peak List",
            fg_color="#6c757d", hover_color="#5a6268", command=self.toggle_peak_visibility
        )
        self.visibility_button.pack(padx=15, pady=5, fill="x")

        self.clear_list_button = ctk.CTkButton(
            self.sidebar_frame, text="Clear Peak List",
            fg_color="#dc3545", hover_color="#bd2130", command=self.clear_peak_list
        )
        self.clear_list_button.pack(padx=15, pady=5, fill="x")

        # SECTION 4: Global Isotope Generator
        self.iso_lbl = ctk.CTkLabel(self.sidebar_frame, text="Isotopic Footprint Modeling", font=ctk.CTkFont(size=13, weight="bold"))
        self.iso_lbl.pack(padx=15, pady=(25, 5), anchor="w")

        self.iso_button = ctk.CTkButton(
            self.sidebar_frame, text="Viz Peak List Isotopes", 
            fg_color="#1f77b4", hover_color="#155d8b", command=self.add_isotopes_to_spectra
        )
        self.iso_button.pack(padx=15, pady=5, fill="x")

        # Dashboard status layout label with dynamic text-wrap configurations to prevent clipping
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame, text="Ready.", text_color="gray", 
            font=ctk.CTkFont(size=12, slant="italic"), wraplength=270, justify="left"
        )
        self.status_label.pack(padx=15, pady=20, fill="x")

        self.close_button = ctk.CTkButton(self.sidebar_frame, text="Save & Return to Notebook", fg_color="#5a5a5a", command=self.close_and_resume)
        self.close_button.pack(padx=15, pady=10, fill="x")

        # --- RIGHT PANEL: CANVAS GRAPH DISPLAY CONTAINER ---
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(1, weight=0)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.spectrum_trace, = self.ax.plot(self.current_mass_axis, self.active_spectra_data, color="black", label="Spectrum", lw=1.2)
        self.ax.set_xlabel("Mass-to-Charge (m/z)")
        self.ax.set_ylabel("Intensity (ions/s)")
        self.ax.grid(True, linestyle=":", alpha=0.5)

        if initial_mz_zoom is not None:
            self.ax.set_xlim(initial_mz_zoom - 0.5, initial_mz_zoom + 0.5)

        self.check_and_plot_existing_baseline()
        self.render_deployment_peak_list()
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(5, 0))

    def open_mopf_settings(self):
        MopfSettingsWindow(self)

    def render_deployment_peak_list(self):
        """Draws defined global peaks as thin lines."""
        for item in self.peak_list_elements:
            item.remove()
        self.peak_list_elements.clear()

        if not self.show_peak_list:
            return

        p_list = getattr(self.deployment, "peak_list", None)
        if not p_list or "peaks" not in p_list or len(p_list["peaks"]) == 0:
            return

        ymin, ymax = self.ax.get_ylim()
        view_height = ymax - ymin

        for entry in p_list["peaks"]:
            try:
                # Handle tracking logic indexing if it's a string identifier versus an absolute float mass value
                if isinstance(entry, (int, float)):
                    center_m = entry
                else:
                    idx_entry = list(p_list["peaks"]).index(entry)
                    center_m = p_list["centers"][idx_entry]
                
                label_str = str(entry)

                idx = np.argmin(np.abs(self.current_mass_axis - center_m))
                local_signal_height = self.active_spectra_data[idx]
                
                text_y_position = local_signal_height + (view_height * 0.03)

                if self.log_checkbox.get() == 1:
                    clamped_height = max(1e-3, local_signal_height)
                    text_y_position = clamped_height * 1.5
                    if text_y_position > ymax * 0.5: 
                        text_y_position = ymax * 0.5
                else:
                    if text_y_position > ymax - (view_height * 0.15): text_y_position = ymax - (view_height * 0.15)
                    if text_y_position < ymin + (view_height * 0.1): text_y_position = ymin + (view_height * 0.1)

                line = self.ax.axvline(x=center_m, color="blue", alpha=0.25, linestyle="--", lw=1.0)
                txt = self.ax.text(center_m, text_y_position, f" {label_str}", color="blue", alpha=0.5, rotation=90, fontsize=8)
                
                self.peak_list_elements.extend([line, txt])
            except Exception:
                continue

    def toggle_log_scale(self):
        try:
            self.update_plot_trace()
            self.status_label.configure(text="Y-axis scale rendering updated.")
        except Exception as e:
            self.status_label.configure(text=f"⚠️ Scale Adjustment Error: {str(e)[:25]}")

    def toggle_baseline(self):
        if self.baseline_trace is not None:
            self.baseline_trace.remove()
            self.baseline_trace = None
            self.status_label.configure(text="Baseline trace hidden.")
            self.ax.legend(loc="upper right")
            self.canvas.draw()
            return

        ws_input = self.window_size_entry.get().strip()
        ws_param = None
        if ws_input:
            try:
                ws_param = int(ws_input)
                if ws_param <= 0: raise ValueError
            except ValueError:
                self.status_label.configure(text="❌ Window size must be a positive integer!")
                return

        self.status_label.configure(text="⏳ Evaluating baseline distributions...")
        self.update_idletasks()

        raw_spectrum = getattr(self.deployment, "reference", {}).get("reference_spectrum", self.current_spectra)
        if ws_param is not None:
            self.deployment.baseline = None

        try:
            if getattr(self.deployment, "baseline", None) is None:
                self.deployment.determine_baseline(target_spectrum=raw_spectrum, window_size=ws_param, plot_flag=False)

            adj_b = self.deployment.baseline["adjusted_baseline"]
            actual_ws = self.deployment.baseline.get("window_size", ws_param if ws_param else int(len(raw_spectrum) * 0.001))

            self.baseline_trace, = self.ax.plot(
                self.current_mass_axis, adj_b, color="#9467bd", linestyle="-.", label="Baseline Floor", lw=1.0
            )
            self.ax.legend(loc="upper right")
            self.canvas.draw()
            self.status_label.configure(text=f"✅ Baseline drawn!\nUsed window_size={actual_ws}")
        except Exception as e:
            self.status_label.configure(text=f"❌ Baseline Error: {str(e)[:30]}")

    def toggle_subtraction(self):
        if getattr(self.deployment, "baseline", None) is None:
            self.status_label.configure(text="❌ Compute a baseline floor first before subtracting!")
            return

        adj_b = self.deployment.baseline["adjusted_baseline"]

        if not self.showing_subtracted:
            self.active_spectra_data = self.current_spectra - adj_b
            self.showing_subtracted = True
            self.subtraction_button.configure(text="Restore Raw Trace", fg_color="#d62728", hover_color="#b51a1a")
            self.status_label.configure(text="✅ Displaying baseline-subtracted view matrix.")
        else:
            self.active_spectra_data = self.current_spectra.copy()
            self.showing_subtracted = False
            self.subtraction_button.configure(text="Apply Subtraction to Plot", fg_color="#9467bd", hover_color="#7a4f9e")
            self.status_label.configure(text="Restored raw unfiltered trace view.")

        self.update_plot_trace()

    def run_mopf(self):
        pw_func = getattr(self.deployment, "peak_width_function", None)
        if pw_func is None:
            self.status_label.configure(
                text="❌ MOPF Denied: peak_width_function is undefined.\nRun determine_peak_width() first!",
                text_color="#ff3333"
            )
            return

        xmin, xmax = self.ax.get_xlim()
        mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
        
        if np.sum(mask) < 5:
            self.status_label.configure(text="❌ Viewport narrow! Zoom out to fit peaks.")
            return

        self.status_label.configure(text="⏳ Processing Multi-Overlap System Fit...")
        self.update_idletasks()

        try:
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

            # Unpack dynamically stored user preferences directly into parent fitting module call
            fit_res = multi_overlap_peak_fit(
                mass_axis=mz_seg, intensity_axis=int_seg, tof_axis=tof_seg,
                peak_type=peak_type, custom_shape=custom_shape, min_separation_fwhm=min_sep,
                plot_flag=False, verbose=False, **self.mopf_params
            )

            if fit_res is None or len(fit_res.get("peaks", [])) == 0:
                self.status_label.configure(text="⚠️ Complete. No peaks resolved in this window.")
                self.canvas.draw()
                return

            dropdown_options = []
            ymin, ymax = self.ax.get_ylim()
            text_y = ymin * 2.0 if self.log_checkbox.get() == 1 else ymin + (ymax - ymin) * 0.75

            resolved_func = peak_function_selector(peak_type, custom_shape=custom_shape)
            n_p = 4 if peak_type == "pseudo_voigt" else 3
            composite = np.zeros_like(tof_seg)

            for i, peak in enumerate(fit_res["peaks"]):
                m_center = peak["center_mass"]
                dropdown_options.append(f"Peak {i+1}: {m_center:.5f}")

                v_line = self.ax.axvline(x=m_center, color="grey", linestyle=":", alpha=0.3)
                t_lbl = self.ax.text(m_center, text_y, f" m={m_center:.3f}", color="grey", rotation=90, fontsize=8)
                self.mopf_elements.extend([v_line, t_lbl])

                p_vals = fit_res["popt"][i * n_p : (i + 1) * n_p]
                if peak_type == "pseudo_voigt":
                    peak_curve = resolved_func(tof_seg, p_vals[0], p_vals[1], p_vals[2], p_vals[3])
                else:
                    peak_curve = resolved_func(tof_seg, p_vals[0], p_vals[1], p_vals[2])

                composite += peak_curve

                indiv_trace, = self.ax.plot(
                    mz_seg, peak_curve, 
                    linestyle="--", lw=1.0, alpha=0.6,
                    label="Peak Components" if i == 0 else ""
                )
                self.mopf_elements.append(indiv_trace)

            self.fit_trace, = self.ax.plot(mz_seg, composite, color="#2ca02c", lw=2, label="MOPF Total Fit")
            
            self.peak_dropdown.configure(values=dropdown_options)
            self.peak_dropdown.set(dropdown_options[0])
            self.selected_mopf_mass = fit_res["peaks"][0]["center_mass"]

            self.ax.legend(loc="upper right")
            self.canvas.draw()
            self.status_label.configure(text=f"✅ Resolved {len(fit_res['peaks'])} overlapping peaks!", text_color="gray")

        except Exception as e:
            self.status_label.configure(text=f"❌ Fit Processing Exception: {str(e)[:35]}", text_color="#ff3333")

    def on_peak_dropdown_select(self, choice):
        try:
            self.selected_mopf_mass = float(choice.split(": ")[1])
        except Exception:
            self.selected_mopf_mass = None

    def commit_entry_to_list(self):
        user_input = self.formula_entry.get().strip()
        if not user_input:
            self.status_label.configure(text="❌ Input entry cannot be blank!")
            return

        if not hasattr(self.deployment, "peak_list") or self.deployment.peak_list is None:
            self.deployment.peak_list = {"peaks": [], "centers": np.array([]), "fwhms": np.array([])}

        current_peaks = list(self.deployment.peak_list["peaks"])
        
        is_numeric = False
        try:
            target_value = float(user_input)
            is_numeric = True
        except ValueError:
            target_value = user_input 

        if target_value in current_peaks:
            self.status_label.configure(text=f"⚠️ Target marker already tracking in peak list.")
            return

        try:
            if is_numeric:
                current_peaks.append(target_value)
                self.deployment.populate_peak_list_and_isotopes(current_peaks)
                self.render_deployment_peak_list()
                self.canvas.draw()
                
                if self.selected_mopf_mass is not None:
                    ppm_dev = ((self.selected_mopf_mass - target_value) / target_value) * 1e6
                    self.status_label.configure(text=f"✅ Added mass: {target_value:.4f}\n[{ppm_dev:.2f} PPM]")
                else:
                    self.status_label.configure(text=f"✅ Added numeric mass {target_value:.4f} to list.")
            else:
                theo_m, ppm_dev = calculate_mass_deviation(self.selected_mopf_mass if self.selected_mopf_mass else 1.0, target_value)
                if theo_m is None:
                    self.status_label.configure(text="❌ Invalid formula syntax parsed!")
                    return
                
                current_peaks.append(target_value)
                self.deployment.populate_peak_list_and_isotopes(current_peaks)
                self.render_deployment_peak_list()
                self.canvas.draw()
                
                if self.selected_mopf_mass is not None:
                    self.status_label.configure(text=f"✅ Bound formula {target_value}\nTheo: {theo_m:.4f} |\nDev: {ppm_dev:.2f} PPM")
                else:
                    self.status_label.configure(text=f"✅ Bound formula {target_value} (Theo Mass: {theo_m:.4f})")
                    
        except Exception as e:
            self.status_label.configure(text=f"❌ List Append Failure:\n{str(e)[:40]}")

    def check_and_plot_existing_baseline(self):
        if getattr(self.deployment, "baseline", None) is not None:
            adj_b = self.deployment.baseline["adjusted_baseline"]
            if len(adj_b) == len(self.current_mass_axis):
                self.baseline_trace, = self.ax.plot(
                    self.current_mass_axis, adj_b, color="#9467bd", linestyle="-.", label="Baseline Floor", lw=1.0
                )

    def add_isotopes_to_spectra(self):
        p_list = getattr(self.deployment, "peak_list", None)
        if not p_list or "peaks" not in p_list or len(p_list["peaks"]) == 0:
            self.status_label.configure(text="❌ No peaks defined in deployment.peak_list!")
            return

        xmin, xmax = self.ax.get_xlim()
        mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
        mz_seg = self.current_mass_axis[mask]
         
        if len(mz_seg) < 2:
            self.status_label.configure(text="❌ Zoom window invalid for isotope modeling.")
            return

        self.status_label.configure(text="⏳ Reconstructing cumulative isotope footprint...")
        self.update_idletasks()

        total_isotope_profile = np.zeros_like(mz_seg, dtype=float)
        pw_func = getattr(self.deployment, "peak_width_function", lambda m: max(0.005, m / 2000.0))
        peak_type = "custom" if getattr(self.deployment, "custom_peak_shape", None) is not None else "gaussian"
        custom_shape = getattr(self.deployment, "custom_peak_shape", None)
        
        formulas_modeled = 0

        for entry in p_list["peaks"]:
            if not isinstance(entry, str):
                continue  

            try:
                idx_entry = list(p_list["peaks"]).index(entry)
                center_m = p_list["centers"][idx_entry]
                idx = np.argmin(np.abs(self.current_mass_axis - center_m))
                
                parent_amplitude = self.active_spectra_data[idx]
                if parent_amplitude <= 0:
                    continue

                profile = isotope_signal_on_axis(
                    formula=entry, mass_axis=mz_seg, parent_amplitude=parent_amplitude,
                    peak_width_function=pw_func, peak_type=peak_type, custom_shape=custom_shape, cutoff=1e-4
                )
                total_isotope_profile += profile
                formulas_modeled += 1
            except Exception:
                continue

        if formulas_modeled == 0:
            self.status_label.configure(text="⚠️ No valid formula chemical peaks found to overlay.")
            return

        if self.iso_trace:
            self.iso_trace.remove()
            self.iso_trace = None

        self.iso_trace, = self.ax.plot(
            mz_seg, total_isotope_profile, color="#d62728", lw=1.5, linestyle="--", 
            label="Peak List Combined Isotopes"
        )
        self.ax.legend(loc="upper right")
        self.canvas.draw()
        self.status_label.configure(text=f"✅ Combined isotope overlay drawn for {formulas_modeled} formulas!")

    def update_plot_trace(self):
        if self.log_checkbox.get() == 1:
            floor_val = 1e-3
            display_y = np.where(self.active_spectra_data < floor_val, floor_val, self.active_spectra_data)
            
            self.spectrum_trace.set_ydata(display_y)
            self.ax.set_yscale("log")
            self.ax.set_ylim(bottom=floor_val, top=np.max(display_y) * 3)
        else:
            self.spectrum_trace.set_ydata(self.active_spectra_data)
            self.ax.set_yscale("linear")
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True)

        self.render_deployment_peak_list()
        self.canvas.draw()

    def run_global_peak_discovery(self):
        pw_func = getattr(self.deployment, "peak_width_function", None)
        if pw_func is None:
            self.status_label.configure(
                text="❌ Discovery Denied: peak_width_function is undefined.\nRun determine_peak_width() first!",
                text_color="#ff3333"
            )
            return

        orig_xmin, orig_xmax = self.ax.get_xlim()
        nm_list = np.unique(np.round(self.current_mass_axis)).astype(int)
        
        self.status_label.configure(text=f"⏳ Discovering peaks across {len(nm_list)} channels...", text_color="gray")
        self.update_idletasks()

        for elem in self.mopf_elements:
            elem.remove()
        self.mopf_elements.clear()
        if self.fit_trace:
            self.fit_trace.remove()
            self.fit_trace = None

        found_peaks_list = []
        peak_type = "custom" if getattr(self.deployment, "custom_peak_shape", None) is not None else "gaussian"
        custom_shape = getattr(self.deployment, "custom_peak_shape", None)
        resolved_func = peak_function_selector(peak_type, custom_shape=custom_shape)
        n_p = 4 if peak_type == "pseudo_voigt" else 3

        active_flash_lines = []

        try:
            for count, nm in enumerate(nm_list):
                self.status_label.configure(text=f"Scanning channel: {nm} m/z ({count+1}/{len(nm_list)})")
                self.ax.set_xlim(nm - 0.5, nm + 0.5)
                
                for line in active_flash_lines:
                    line.remove()
                active_flash_lines.clear()

                seg_i, seg_m, seg_t = get_nm_segment_data(
                    nm=nm, 
                    spectra=self.active_spectra_data, 
                    mass_axis=self.current_mass_axis, 
                    tof_axis=getattr(self.deployment, "tof_axis", np.arange(len(self.current_mass_axis)))
                )

                if len(seg_i) > 0:
                    max_val = np.max(seg_i)
                    if self.log_checkbox.get() == 1:
                        floor_val = 1e-3
                        self.ax.set_ylim(bottom=floor_val, top=max(floor_val * 10, max_val * 3))
                    else:
                        min_val = np.min(seg_i)
                        padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
                        self.ax.set_ylim(bottom=min_val - padding, top=max_val + padding)

                if len(seg_m) < 5 or np.max(seg_i) <= 0:
                    self.canvas.draw()
                    self.update()
                    continue

                min_sep = pw_func(nm)
                
                # Dynamic parameter dictionary integration
                fit_results = multi_overlap_peak_fit(
                    mass_axis=seg_m, intensity_axis=seg_i, tof_axis=seg_t,
                    peak_type=peak_type, custom_shape=custom_shape, min_separation_fwhm=min_sep,
                    plot_flag=False, verbose=False, **self.mopf_params
                )

                if fit_results and len(fit_results.get("peaks", [])) > 0:
                    composite_segment = np.zeros_like(seg_t)
                    
                    for peak in fit_results["peaks"]:
                        found_peaks_list.append(peak["center_mass"])
                    
                    for i_p in range(len(fit_results["popt"]) // n_p):
                        p_vals = fit_results["popt"][i_p * n_p : (i_p + 1) * n_p]
                        if peak_type == "pseudo_voigt":
                            composite_segment += resolved_func(seg_t, p_vals[0], p_vals[1], p_vals[2], p_vals[3])
                        else:
                            composite_segment += resolved_func(seg_t, p_vals[0], p_vals[1], p_vals[2])

                    f_trace, = self.ax.plot(seg_m, composite_segment, color="#2ca02c", lw=1.8)
                    active_flash_lines.append(f_trace)

                self.canvas.draw()
                self.update()
                time.sleep(0.02)  

            for line in active_flash_lines:
                line.remove()

            print(f"[GUI]: Automated Discovery concluded. Found {len(found_peaks_list)} peaks globally.")
            self.deployment.populate_peak_list_and_isotopes(found_peaks_list)

            self.ax.set_xlim(orig_xmin, orig_xmax)
            self.update_plot_trace()
            
            self.status_label.configure(
                text=f"✅ Discovery Done! Registered {len(found_peaks_list)} peaks to list.", 
                text_color="green"
            )

        except Exception as e:
            self.ax.set_xlim(orig_xmin, orig_xmax)
            self.update_plot_trace()
            self.status_label.configure(text=f"❌ Discovery Error: {str(e)[:30]}", text_color="#ff3333")

    def auto_y_scale(self):
        try:
            xmin, xmax = self.ax.get_xlim()
            mask = (self.current_mass_axis >= xmin) & (self.current_mass_axis <= xmax)
            
            if np.sum(mask) == 0:
                self.status_label.configure(text="⚠️ Auto-Y Failed: No spectral points visible in this window.")
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
   
            self.render_deployment_peak_list()
            self.canvas.draw()
            self.status_label.configure(text="✅ Viewport Y-limits scaled to local data window.")
            
        except Exception as e:
            self.status_label.configure(text=f"❌ Auto-Y Error: {str(e)[:30]}", text_color="#ff3333")

    def toggle_peak_visibility(self):
        self.show_peak_list = not self.show_peak_list
        if self.show_peak_list:
            self.visibility_button.configure(text="Hide Peak List", fg_color="#6c757d", hover_color="#5a6268")
            self.status_label.configure(text="Peak annotations shown.")
        else:
            self.visibility_button.configure(text="Show Peak List", fg_color="#28a745", hover_color="#218838")
            self.status_label.configure(text="Peak annotations hidden to\nreduce rendering lag.")
        
        self.render_deployment_peak_list()
        self.canvas.draw()

    def clear_peak_list(self):
        self.deployment.peak_list = {"peaks": [], "centers": np.array([]), "fwhms": np.array([])}
        self.render_deployment_peak_list()
        self.canvas.draw()
        self.status_label.configure(text="✅ Global peak list cleared successfully.", text_color="green")

    def close_and_resume(self):
        self.destroy()