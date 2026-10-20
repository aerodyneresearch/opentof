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

# quantification.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import warnings

from sklearn.cluster import KMeans

from opentof.utils import (
    median_absolute_deviation,
    ensure_dir, 
    get_default_plot_dir
)

# long term TODO: Maybe in this .py file is where we can keep dictionary of all or 
#  few standard k_ptr rates? or perhaps build functions into OpenTof 
#  for KPTR estimation based on atomic principles (calling psi4??) for example:
#  proton affinity, dipole moment, polarizability. Seems like a longer project?


# Victor Note: LEGACY FUNCTION SPECIALIZED TO CDPHE OPERATIONS. CAN LIKELY BE MADE MORE GENERAL AT SOME POINT
def load_TWEB_sensitivity_calibration_file(file_path, skiprows=9, plot_flag=False,
                                           output_dir=None, plot_filename=None):
    """
    Loads and parses sensitivity calibration data from a specific TWEB file format.
    Fits a linear function for kPTR vs Sensitivity and a logistic transmission curve.
    """
    # Currently only available/updated for the EMU
    sens_file = pd.read_csv(file_path, skiprows=skiprows)
    df_kPTR = sens_file[sens_file["Used for kPTR"] == 1]
    df_transmission = sens_file[sens_file["Used for transmission"] == 1]

    # Define a linear function for kPTR vs Sensitivity
    def linear_func(x, a, b):
        return a * x + b
    
    # Fit the linear model
    popt_lin, _ = curve_fit(linear_func, df_kPTR["kPTR (10⁻⁹ cm³ molec⁻¹ s⁻¹)"], df_kPTR["Sensitivity (ions/s/ppbV)"])
    slope, intercept = popt_lin

    # Define a logistic function with base and max, ensuring max is capped at 1
    def logistic(x, x0, k, L, U):
        return L + (U - L) / (1 + np.exp(-k * (x - x0)))

    # Estimate `base` and `max` from the data
    base_guess = df_transmission["Transmission"].min()
    max_guess = df_transmission["Transmission"].max()

    # Initial guesses for x_half and rate
    x_half_guess = np.median(df_transmission["m/Q (Th)"])
    rate_guess = 0.1  # Higher value makes transition steeper

    # clamp helper
    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    p0 = [
        clamp(x_half_guess, 20, 400),
        clamp(rate_guess, 0.001, 10),
        clamp(base_guess, 0, 0.1),
        clamp(max_guess, 0.5, 1.5),
    ]

    # Fit with broader parameter ranges
    try:
        popt_sig, _ = curve_fit(
            logistic,
            df_transmission["m/Q (Th)"],
            df_transmission["Transmission"],
            p0=p0,
            bounds=([20, 0.001, 0, 0.5], [400, 10, 0.1, 1.5]),
            maxfev=20000
        )
        x_half, rate, base, max_val = popt_sig
    except RuntimeError:
        print("Transmission curve fitting failed!")
        x_half, rate, base, max_val = np.nan, np.nan, np.nan, np.nan  

    # Generate fitted values for plotting
    x_kPTR_extended = np.linspace(1.50, 4.25, 100)
    y_kPTR_extended = linear_func(x_kPTR_extended, *popt_lin)

    # Define extended range for transmission curve
    x_trans_fit_extended = np.linspace(12, 400, 4000)  
    y_trans_fit_extended = logistic(x_trans_fit_extended, *popt_sig)

    if plot_flag:
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        if plot_filename is None:
            plot_filename = "sensitivity_calibration.png"

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # First subplot: kPTR vs Sensitivity
        axs[0].scatter(df_kPTR["kPTR (10⁻⁹ cm³ molec⁻¹ s⁻¹)"], df_kPTR["Sensitivity (ions/s/ppbV)"], label="Data", color="blue")
        axs[0].plot(x_kPTR_extended, y_kPTR_extended, label="Linear Fit", color="red", linestyle="dashed")
        axs[0].set_xlabel("kPTR (10⁻⁹ cm³ molec⁻¹ s⁻¹)")
        axs[0].set_ylabel("Sensitivity (ions/s/ppbV)")
        axs[0].set_title("kPTR vs Sensitivity")
        axs[0].grid(True)
        axs[0].legend()

        # Annotate slope and intercept
        axs[0].annotate(f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}", 
                        xy=(0.74, 0.05), xycoords='axes fraction', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.6))

        # Second subplot: m/Q vs Transmission (Extended)
        axs[1].scatter(df_transmission["m/Q (Th)"], df_transmission["Transmission"], label="Measured Data", color="green")
        axs[1].plot(x_trans_fit_extended, y_trans_fit_extended, label="Extended Logistic Fit", color="orange", linestyle="dashed")
        axs[1].set_xlabel("m/Q (Th)")
        axs[1].set_ylabel("Transmission Efficiency")
        axs[1].set_title("Transmission Efficiency Curve")
        axs[1].grid(True)
        axs[1].legend()

        # Annotate x_half and rate
        axs[1].annotate(f"x_half: {x_half:.2f}\nRate: {rate:.2f}\nBase: {base:.2f}\nMax: {max_val:.2f}", 
                        xy=(0.80, 0.05), xycoords='axes fraction', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.6))

        # Adjust layout and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
        plt.show()
        plt.close()

    # Build the dictionary mapped by compound formula
    sens_dict = {}
    for i in range(len(sens_file)):
        if sens_file.iloc[i].iloc[13] == 1.00:
            sens_dict[sens_file.iloc[i].iloc[0]] = {
                "Sensitivity": sens_file.iloc[i].iloc[2],
                "kPTR": np.nan,
                "Transmission": np.nan,
            }
        else:
            sens_dict[sens_file.iloc[i].iloc[0]] = {
                "Sensitivity": sens_file.iloc[i].iloc[2],
                "kPTR": sens_file.iloc[i].iloc[13],
                "Transmission": sens_file.iloc[i].iloc[14],
            }

    def sensitivity_kptr_linear(k_ptr):
        return linear_func(k_ptr, *popt_lin)

    def transmission_mz_sigmoid(mz):
        return logistic(mz, *popt_sig)

    # Return the dictionary itself and functions defined based on sensitivities
    return sens_dict, sensitivity_kptr_linear, transmission_mz_sigmoid


def _locate_index_chunks(filterd_auto_cycling_arr, gap_tol=5, min_chunk_size=1):
    """
    Identify and group contiguous sequences of indices within a 1D array.

    Partitions a sorted array of indices into discrete sub-lists ("chunks"). 
    A new chunk is started whenever the numerical index gap between two consecutive 
    elements exceeds the specified tolerance `gap_tol`.

    Parameters
    ----------
    filterd_auto_cycling_arr : list or numpy.ndarray
        Sorted 1D sequence of integer array indices representing active cycling events 
        (e.g., auto-zeros or sensitivity calibrations).
    gap_tol : int, default=5
        Maximum allowable numerical index gap between consecutive elements before 
        starting a new chunk block.
    min_chunk_size : int, default=1
        Minimum number of indices required for a chunk to be considered valid.

    Returns
    -------
    all_chunks : list of list of int
        List of sub-lists, where each sub-list contains contiguous or closely spaced 
        indices belonging to a single event block.
    """
    if len(filterd_auto_cycling_arr) == 0:
        return []

    all_chunks = []
    chunk = [filterd_auto_cycling_arr[0]]
    prev_ms_i = filterd_auto_cycling_arr[0]
    
    # Iterate through remaining indices starting from the second element
    for ms_i in filterd_auto_cycling_arr[1:]:
        # Append to current active chunk if index step is within gap tolerance
        if (ms_i - prev_ms_i) < gap_tol:
            chunk.append(ms_i)
        else:
            # Index gap breached: save completed chunk if it meets length requirements
            if len(chunk) >= min_chunk_size:
                all_chunks.append(chunk)
            chunk = [ms_i]
            
        prev_ms_i = ms_i
    
    # Append the final remaining chunk after loop termination
    if len(chunk) >= min_chunk_size:
        all_chunks.append(chunk)
        
    return all_chunks


def _assign_blocks_from_chunks(chunks_list, peak_area_array):
    """
    Map chunked event indices and intermediate gaps into a continuous ID labeling array.

    Constructs a 1D integer block ID array aligned with `peak_area_array`. Active event 
    chunks (e.g., auto-zeros, sensitivity calibrations) are assigned sequential negative 
    integers (``-1, -2, -3, ...``), while intervening gap periods (standard acquisition) 
    are assigned sequential positive integers (``1, 2, 3, ...``).

    Parameters
    ----------
    chunks_list : list of list of int
        List of index chunks returned by :func:`_locate_index_chunks`.
    peak_area_array : numpy.ndarray
        1D target signal array used to establish total timeline length.

    Returns
    -------
    block_array : numpy.ndarray
        1D integer array (``dtype=int32``) matching `peak_area_array` length, containing 
        assigned block and gap region IDs.
    """
    array_length = len(peak_area_array)
    block_array = np.zeros(array_length, dtype=np.int32)
    last_chunk_end_index = 0
    
    # Counter for assigning negative chunk IDs (-1, -2, -3, ...)
    chunk_counter = 1
    # Counter for assigning positive gap IDs (1, 2, 3, ...)
    gap_id = 1 

    for chunk in chunks_list:
        if not chunk:
            continue
            
        first_index = int(chunk[0])
        last_index = int(chunk[-1])
        
        # Mark all sample channels inside active chunk with negative chunk ID
        chunk_id_value = -chunk_counter
        block_array[first_index: last_index + 1] = chunk_id_value
        chunk_counter += 1 
        
        # Assign gap ID to standard acquisition channels between previous and current chunk
        if last_chunk_end_index > 0 and last_chunk_end_index < first_index:
            block_array[last_chunk_end_index:first_index] = gap_id
            gap_id += 1
            
        last_chunk_end_index = last_index + 1

    # Mark trailing gap from end of last chunk to array boundary with current gap ID
    if last_chunk_end_index < array_length:
        block_array[last_chunk_end_index: array_length] = gap_id
        
    return block_array


def _perform_background_subtraction(zero_blocks, peak_area_array, mode="interp", 
                                    min_zero_points=5, trim_transient_points=2,
                                    outlier_mad_mult=4.0):
    """
    Subtracts interpolated or static ions/s value determined from auto-zero periods.

    To protect against transient effects, electronic noise, and outlier zero events, the following
    filtering/quality criteria are applied to each auto-zero block before baseline interpolation:
      1. **Cycle Duration Gating**: Rejects aborted or prematurely truncated zero 
         sequences containing fewer than `min_zero_points` spectra.
      2. **Inlet Line Flush Trimming**: Discards the initial `trim_transient_points`
         of each zero event in an effort to remove transient effects.
      3. **Within-Block Spike Filtering**: Rejects transient electronic spikes, pressure
         fluctuations, or detector noise within a block using a 4 * MAD filter centered
         on the settled block median.
      4. **Cross-Block Outlier Screening**: Evaluates all block zero averages across the
         entire deployment. An automated zero is determined to be an outlier if a zero-periods's 
         mean signal deviates sharply from the global median. Outliers exceeding 
         `outlier_mad_mult` * MAD are excluded from baseline interpolation to prioritize
         stability of the background subtraciton procesure during the final quantification 
         step.

    Parameters
    ----------
    zero_blocks : numpy.ndarray of int
        1D integer block array matching `peak_area_array` length (from `_assign_blocks_from_chunks`).
        Negative values (``-1, -2, ...``) identify active auto-zero periods; positive values
        (``1, 2, ...``) identify ambient sample acquisition tracking gaps.
    peak_area_array : numpy.ndarray of float
        1D writebuf-aligned array of peak area or amplitude intensities (ions/s), with any
        primary reagent ion normalization already applied.
    mode : {"interp", "median"}, default="interp"
        Baseline construction strategy across the timeline:
        * ``"interp"``: Performs 1D linear interpolation (`np.interp`) between the midpoints
          of validated auto-zero blocks. Recommended for capturing dynamic baseline shifts
          driven by diurnal ambient humidity changes, drift tube temperature, or source decay.
        * ``"median"``: Calculates the global median across all validated zero blocks and
          applies a flat, constant baseline across the entire dataset.
    min_zero_points : int, default=5
        Minimum number of spectra required to consider a zero sequence valid. Sequences
        shorter than this threshold (e.g., from instrument reboots or aborted sequences)
        are flagged as ``'skipped_too_short'`` and ignored.
    trim_transient_points : int, default=2
        Number of leading points discarded from the beginning of each zero block. Accounts
        for the physical residence and flushing time of the inlet tubing and manifold volume.
    outlier_mad_mult : float, default=4.0
        Multiplier applied to the Median Absolute Deviation (MAD) of all block averages:
        Tolerance = outlier_mad_mult * max(MAD(Z_bar), 0.5)
        Blocks whose mean zero level exceeds Median(Z_bar) ± Tolerance
        are flagged as 'skipped_outlier' and excluded from baseline interpolation.

    Returns
    -------
    full_background_subtraction : numpy.ndarray of float
        1D array of background-subtracted signal values matching `peak_area_array` length.
        Raw values during active zero periods (block IDs <= 0) are preserved intact,
        while ambient tracking blocks have `background_line` subtracted.
    background_line : numpy.ndarray of float
        1D continuous baseline array modeled across the entire acquisition timeline.
    zero_diagnostics : list of dict
        Inspection records for every zero block encountered. Each dictionary contains:
        * ``'block_id'`` (int): Identifier corresponding to the negative block ID.
        * ``'center'`` (float): Median sample index position of the block.
        * ``'start_idx'`` (int): Index of the first writebuf in the block.
        * ``'end_idx'`` (int): Index of the last writebuf in the block.
        * ``'mean'`` (float): Settled mean intensity (ions/s) of the zero block.
        * ``'points'`` (int): Total points recorded during the zero event.
        * ``'status'`` (str): One of ``'accepted'``, ``'skipped_too_short'``, or ``'skipped_outlier'``.
        * ``'reason'`` (str): Human-readable string summarizing the filtering outcome.

    Notes
    -----
    In mode="interp", the baseline is evaluated at sample index i by:

    B[i] = interp(i, c_valid, z_bar_valid)

    where c_valid are the temporal centers and z_bar_valid are the settled mean 
    values of accepted zero blocks. At the boundaries (before the first zero 
    and after the last zero), np.interp flat-lines at the nearest boundary zero value.
    """
    if mode not in ["interp", "median"]:
        raise ValueError("mode must be either 'interp' or 'median'")

    array_length = len(peak_area_array)
    full_background_subtraction = np.zeros_like(peak_area_array, dtype=float)
    unique_zero_blocks = np.unique(zero_blocks)
    neg_blocks = [b for b in unique_zero_blocks if b < 0]

    raw_centers, raw_zeros = [], []
    raw_block_info = {}
    zero_diagnostics = []

    # --- PASS 1: Calculate Mean for Each Block & Flag Aborted Sequences ---
    for b_id in neg_blocks:
        auto_zero_idx_temp = np.where(zero_blocks == b_id)[0]
        center_idx = np.median(auto_zero_idx_temp)
        start_idx = auto_zero_idx_temp[0]
        end_idx = auto_zero_idx_temp[-1]
        raw_mean = float(np.mean(peak_area_array[auto_zero_idx_temp]))
        n_pts = len(auto_zero_idx_temp)

        # Flag aborted or cut-off zero block
        if n_pts < min_zero_points:
            zero_diagnostics.append({
                'block_id': b_id,
                'center': center_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'mean': raw_mean,
                'points': n_pts,
                'status': 'skipped_too_short',
                'reason': f"Cut off ({n_pts} pts < {min_zero_points})"
            })
            continue

        # Trim initial transient points caused by valve switching and line flushing
        if n_pts > (trim_transient_points + 2):
            settled_idx = auto_zero_idx_temp[trim_transient_points:]
        else:
            settled_idx = auto_zero_idx_temp

        local_signal = peak_area_array[settled_idx]
        local_med = np.median(local_signal)
        local_mad = median_absolute_deviation(local_signal)

        # 4*MAD statistical acceptance bounds around local median
        upper_lim = local_med + (4.0 * max(local_mad, 1e-4))
        lower_lim = local_med - (4.0 * max(local_mad, 1e-4))

        clean_idx = settled_idx[(local_signal >= lower_lim) & (local_signal <= upper_lim)]
        current_block_zero = float(np.mean(peak_area_array[clean_idx]) if len(clean_idx) > 0 else np.mean(local_signal))

        raw_centers.append(center_idx)
        raw_zeros.append(current_block_zero)
        raw_block_info[b_id] = {
            'block_id': b_id,
            'center': center_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean': current_block_zero,
            'points': n_pts,
            'status': 'accepted',
            'reason': 'Accepted'
        }

    # --- PASS 2: Cross-Block Outlier Filtering ---
    valid_centers, valid_zeros = [], []

    if len(raw_zeros) >= 3:
        global_zero_median = float(np.median(raw_zeros))
        global_zero_mad = float(median_absolute_deviation(np.array(raw_zeros)))
        tolerance = float(outlier_mad_mult * max(global_zero_mad, 0.5))

        for b_id, info in raw_block_info.items():
            if abs(info['mean'] - global_zero_median) <= tolerance:
                valid_centers.append(info['center'])
                valid_zeros.append(info['mean'])
                zero_diagnostics.append(info)
            else:
                info['status'] = 'skipped_outlier'
                info['reason'] = f"Outlier ({info['mean']:.2f} vs med {global_zero_median:.2f} ± {tolerance:.2f})"
                zero_diagnostics.append(info)
                # # Victor Note: Commented out the warning to reduce verbosity in normal operation. Can be re-enabled if needed.
                # warnings.warn(
                #     f"Auto-zero block {b_id} (mean={info['mean']:.2f}) rejected as a cross-block outlier."
                # )
    else:
        valid_centers = raw_centers
        valid_zeros = raw_zeros
        zero_diagnostics.extend(raw_block_info.values())

    # --- PASS 3: Baseline Construction ---
    if not valid_zeros:
        warnings.warn("No valid auto-zero blocks found. Skipping background subtraction.")
        background_line = np.zeros(array_length, dtype=float)
    elif mode == "interp":
        if len(valid_zeros) == 1:
            background_line = np.full(array_length, valid_zeros[0], dtype=float)
        else:
            sort_order = np.argsort(valid_centers)
            background_line = np.interp(
                np.arange(array_length), 
                np.array(valid_centers)[sort_order], 
                np.array(valid_zeros)[sort_order]
            )
    elif mode == "median":
        background_line = np.full(array_length, np.median(valid_zeros), dtype=float)

    # --- PASS 4: Apply Subtraction ---
    for block_id in unique_zero_blocks:
        idx = np.where(zero_blocks == block_id)[0]
        if block_id <= 0:
            full_background_subtraction[idx] = peak_area_array[idx]
        else:
            full_background_subtraction[idx] = peak_area_array[idx] - background_line[idx]

    return full_background_subtraction, background_line, zero_diagnostics


def _cluster_sensitivity_cal_signal(data_subset):
    """
    Partition sensitivity calibration signal into low (baseline) and high (challenge) states using KMeans.
    
    When calibration gas is injected into the inlet, the signal should step up. 
    K-Means naturally clusters the lower-intensity background points and the higher-intensity 
    challenge points. 
    Returns masks for low and high points, as well as the 'delta_step' difference between their centers.
    """
    # Guard clause: Require at least 4 points to form meaningful clusters
    if len(data_subset) < 4:
        return np.ones(len(data_subset), dtype=bool), np.zeros(len(data_subset), dtype=bool), 0.0

    # Reshape 1D input array to a 2D column vector (N, 1) required by Scikit-Learn KMeans API
    data_reshaped = np.asarray(data_subset, dtype=float).reshape(-1, 1)
    
    # Fit 2-cluster K-Means model to separate the two intensity states (Baseline vs. Challenge)
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(data_reshaped)
    low_cluster_label = np.argmin(kmeans.cluster_centers_)
    
    # Generate boolean masks
    low_mask = (kmeans.labels_ == low_cluster_label)
    high_mask = ~low_mask
    
    # Measure the absolute step size between the two cluster centers
    delta_step = float(np.max(kmeans.cluster_centers_) - np.min(kmeans.cluster_centers_))
    
    return low_mask, high_mask, delta_step


def _extract_event_flow(flow_param, active_indices, default_val=250.0):
    """
    Extract a scalar flow rate from either a static float or a dynamic time-series monitor array.
    Used to pull MFC mass flow controller values during a calibration event.
    """
    if flow_param is None:
        return float(default_val)
    if isinstance(flow_param, (int, float, np.floating, np.integer)):
        return float(flow_param)
    
    flow_arr = np.asarray(flow_param, dtype=float)
    if flow_arr.ndim == 0:
        return float(flow_arr)
        
    # If a full telemetry array is provided, slice out the values corresponding to the event
    if len(active_indices) > 0 and len(flow_arr) > np.max(active_indices):
        event_vals = flow_arr[active_indices]
        valid_vals = event_vals[np.isfinite(event_vals)]
        if len(valid_vals) > 0:
            return float(np.median(valid_vals))
            
    # Fallback to global median if event window is invalid
    valid_global = flow_arr[np.isfinite(flow_arr)]
    return float(np.median(valid_global)) if len(valid_global) > 0 else float(default_val)


def _get_cyclic_sensitivity_calibrations(sens_cal_blocks, 
                                         background_corrected, 
                                         cal_tank_conc=1000,
                                         zero_flow=250,
                                         cal_flow=5,
                                         mode="interp",
                                         static_sens_factor=1.0,
                                         min_cal_points=6,
                                         min_signal_delta=0.5,
                                         trim_cal_transients=2):
    """
    Generate sensitivity calibration factors (in the form of a 1D array) from on-line 
    sensitivity calibrations performed during data acquisition.

    Methodology & Guardrails:
      1. **Two-State K-Means Partitioning**: A K-Means clustering (k=2)
         partitions each calibration window into baseline (low signal) and challenge
         (high signal).
      2. **Species Response Gatekeeping (`min_signal_delta`)**: Calibration gas cylinders
         contain only specific chemical compounds. For non-target compounds, random baseline
         noise would otherwise be split into artificial "low" and "high" clusters. If the
         step increase Delta_S = mean(S_high) - mean(S_low) is below
         `min_signal_delta`, the compound is flagged as not within the tank
         (``'skipped_no_response'``) and bypassed to avoid non-physical sensitivity values.
      3. **Transient Trimming**: Discards the initial `trim_cal_transients`
         points of the challenge step to account for transient/disjointed initial signals 
         as cal/zero valves open and close. 
      4. **Dynamic Mass Flow Dilution**: Calculates diluted concentrations dynamically for
         each event using:
         C_diluted = (Q_cal / (Q_zero + Q_cal)) * C_tank
         Accepts either static flow rates or dynamic 1D telemetry monitor arrays.
      5. **Sensitivity Factor Calculation**: Solves the response factor in concentration
         units per signal count:
         S_c = C_diluted / Delta_S  [ppb / (ions/s)]
         *(Multiplying background-subtracted ions/s by S_c directly yields mixing ratio in ppb).*
      6. **Cross-Block Outlier Filtering**: Rejects blocks with anomalous sensitivities
         exceeding the tolerance window:
         Tolerance = 4.0 * max(MAD(S), 0.15 * Median(S))

    Parameters
    ----------
    sens_cal_blocks : numpy.ndarray of int
        1D integer block ID array (from `_assign_blocks_from_chunks`). Negative integers
        mark active calibration challenge periods; positive integers mark ambient acquisition.
    background_corrected : numpy.ndarray of float
        1D background-subtracted signal array (ions/s) output by `_perform_background_subtraction`.
    cal_tank_conc : float, default=1000
        Certified gas cylinder concentration for the target compound in parts-per-billion (ppbV).
    zero_flow : float or array-like, default=250
        Dilution zero air flow rate in standard cubic centimeters per minute (sccm).
        Can be passed as a constant float or as a full 1D telemetry monitor array
        (e.g., ``d.tps2['Zero MFC monitor [sccm]']``).
    cal_flow : float or array-like, default=5
        Calibration gas standard flow rate in sccm. Can be passed as a constant float or
        as a full 1D telemetry monitor array (e.g., ``d.tps2['Cal MFC monitor [sccm]']``).
    mode : {"interp", "median"}, default="interp"
        Modeling mode for constructing the continuous sensitivity axis:
        * ``"interp"``: Linearly interpolates sensitivity factors between valid calibration
          event midpoints over time.
        * ``"median"``: Applies the global median sensitivity factor across all spectra.
    static_sens_factor : float, default=1.0
        Fallback sensitivity factor applied across all points if no valid calibration
        events are identified (or if instrument cycling was not run).
    min_cal_points : int, default=6
        Minimum number of sample points required to accept a calibration challenge block.
        Aborted or interrupted calibration events are discarded.
    min_signal_delta : float, default=0.5
        Minimum required step height ΔS (ions/s) between the challenge and baseline segments of 
        the sensitivity calibration.
        Prevents calculating artificial sensitivities for compounds not present in the gas standard.
    trim_cal_transients : int, default=2
        Number of leading challenge points trimmed to allow mass flow controllers to settle
        and inlet line surfaces to passivate.

    Returns
    -------
    sensitivity_calibration_axis : numpy.ndarray of float
        1D array matching `background_corrected` length containing the continuous time-series
        sensitivity calibration conversion factor (ppb / (ions/s)).
    cal_diagnostics : list of dict
        Diagnostic inspection records for each calibration challenge block. Each dictionary contains:
        * ``'block_id'`` (int): Identifier corresponding to the negative block ID.
        * ``'center'`` (float): Median sample index position of the calibration block.
        * ``'sens'`` (float): Solved sensitivity factor (ppb / (ions/s)).
        * ``'delta_s'`` (float): Measured signal step increase ΔS (ions/s) between the challenge and baseline states.
        * ``'high_idx'`` (numpy.ndarray): Indices used for the stable challenge state.
        * ``'low_idx'`` (numpy.ndarray): Indices used for the stable baseline state.
        * ``'status'`` (str): One of ``'accepted'``, ``'skipped_too_short'``,
          ``'skipped_no_response'``, or ``'skipped_outlier'``.
        * ``'reason'`` (str): Human-readable string summarizing the filtering outcome.
    """
    if mode not in ["interp", "median"]:
        raise ValueError("mode must be either 'interp' or 'median'")

    array_length = len(background_corrected)
    unique_sens_cal_blocks = np.unique(sens_cal_blocks)
    neg_blocks = [b for b in unique_sens_cal_blocks if b < 0]

    raw_centers, raw_sens = [], []
    raw_block_info = {}
    cal_diagnostics = []

    # --- PASS 1: Calculate Block Response Factors ---
    for b_id in neg_blocks:
        auto_sens_cal_idx = np.where(sens_cal_blocks == b_id)[0]
        center_idx = np.median(auto_sens_cal_idx)

        # Aborted or cut-off calibration challenge block
        if len(auto_sens_cal_idx) < min_cal_points:
            cal_diagnostics.append({
                'block_id': b_id,
                'center': center_idx,
                'sens': np.nan,
                'delta_s': 0.0,
                'status': 'skipped_too_short'
            })
            continue

        block_signal = background_corrected[auto_sens_cal_idx]
        low_mask, high_mask, delta_step = _cluster_sensitivity_cal_signal(block_signal)

        # Compound did not respond (not in calibration tank)
        if delta_step < min_signal_delta or np.sum(high_mask) < 2:
            cal_diagnostics.append({
                'block_id': b_id,
                'center': center_idx,
                'sens': np.nan,
                'delta_s': delta_step,
                'status': 'skipped_no_response'
            })
            continue

        high_indices_local = np.where(high_mask)[0]
        low_indices_local = np.where(low_mask)[0]

        # Trim rising-edge transients
        stable_high_idx = high_indices_local[trim_cal_transients:] if len(high_indices_local) > (trim_cal_transients + 1) else high_indices_local
        stable_low_idx = low_indices_local

        high_signal = block_signal[stable_high_idx]
        low_signal = block_signal[stable_low_idx] if len(stable_low_idx) > 0 else [0.0]

        delta_s = np.mean(high_signal) - np.mean(low_signal)
        if delta_s <= 0:
            continue

        event_indices = auto_sens_cal_idx[stable_high_idx]
        z_flow = _extract_event_flow(zero_flow, event_indices, default_val=250.0)
        c_flow = _extract_event_flow(cal_flow, event_indices, default_val=5.0)

        total_flow = z_flow + c_flow
        if total_flow <= 0:
            continue

        diluted_conc = (c_flow / total_flow) * cal_tank_conc
        block_sens = diluted_conc / delta_s

        raw_centers.append(center_idx)
        raw_sens.append(block_sens)
        raw_block_info[b_id] = {
            'block_id': b_id,
            'center': center_idx,
            'sens': block_sens,
            'delta_s': delta_s,
            'high_idx': auto_sens_cal_idx[stable_high_idx],
            'low_idx': auto_sens_cal_idx[stable_low_idx],
            'status': 'accepted'
        }

    # --- PASS 2: Cross-Block Outlier Rejection ---
    valid_centers, valid_sens = [], []

    if len(raw_sens) >= 3:
        sens_median = np.median(raw_sens)
        sens_mad = median_absolute_deviation(np.array(raw_sens))
        sens_tol = 4.0 * max(sens_mad, sens_median * 0.15)

        for b_id, info in raw_block_info.items():
            if abs(info['sens'] - sens_median) <= sens_tol:
                valid_centers.append(info['center'])
                valid_sens.append(info['sens'])
                cal_diagnostics.append(info)
            else:
                info['status'] = 'skipped_outlier'
                cal_diagnostics.append(info)
                warnings.warn(
                    f"Cal block {b_id} (sens={info['sens']:.4f}) rejected as a cross-block outlier."
                )
    else:
        valid_centers = raw_centers
        valid_sens = raw_sens
        cal_diagnostics.extend(raw_block_info.values())

    # --- PASS 3: Timeline Construction ---
    if not valid_sens:
        warnings.warn(f"No valid sensitivity calibration blocks found. Falling back to static_sens_factor ({static_sens_factor}).")
        return np.full(array_length, static_sens_factor, dtype=float), cal_diagnostics

    if mode == "interp":
        if len(valid_sens) == 1:
            sensitivity_calibration_axis = np.full(array_length, valid_sens[0], dtype=float)
        else:
            sort_order = np.argsort(valid_centers)
            sensitivity_calibration_axis = np.interp(
                np.arange(array_length),
                np.array(valid_centers)[sort_order],
                np.array(valid_sens)[sort_order]
            )
    elif mode == "median":
        sensitivity_calibration_axis = np.full(array_length, np.median(valid_sens), dtype=float)

    return sensitivity_calibration_axis, cal_diagnostics


def quantify_signal(peak_area_array, cycling_dict=None, 
                    cycling_zeros=True, cycling_sens_cals=True, normalization=False,
                    normalization_reference_values=None, static_target_value=1e6,
                    cal_tank_conc=1000.0, zero_flow=250.0, cal_flow=5.0, 
                    static_sens_factor=1.0,
                    mode="interp",
                    min_zero_points=5,
                    min_cal_points=6,
                    trim_zero_transients=2,
                    trim_cal_transients=2,
                    min_signal_delta=1.0,
                    plot_flag=False,
                    output_dir=None,
                    plot_subdir="quantification_plots",
                    timestamps=None,
                    standard_acq_mask=None,
                    compound_name="Target Compound",
                    plot_filename=None,
                    show_plot=False,
                    **kwargs):
    """
    Quantify a peak area time-series into physical concentration units (ppb).

    This backend routine handles the full physical data reduction pipeline:
      1. Primary reagent ion normalization (correcting for ion source fluctuations).
      2. Dynamic auto-zero background subtraction (overflowed zero air baseline).
      3. Dynamic sensitivity calibration factor scaling (standard gas challenges).
      4. Diagnostic summary plotting (visualizing baseline drift, accepted/rejected
         cycles, challenge steps, and ambient mixing ratios).

    It includes guards against instrument artifacts:
      - Aborted/cut-off cycles (< min points) are automatically skipped.
      - Valve switching and inlet line flushing transients are trimmed from calculations.
      - Compounds not in the calibration standard (delta_signal < min_signal_delta) are
        bypassed to avoid dividing by near-zero noise.
      - Deployments lacking cycling telemetry (flags all -1) gracefully fall back to 
        the user-specified `static_sens_factor` rather than multiplying by zero.

    Parameters
    ----------
    peak_area_array : array-like
        1D array of writebuf-aligned peak areas or amplitudes for a target compound.
    cycling_dict : dict or None, default=None
        Instrument cycling dictionary containing a ``'type'`` array of status flags.
        Required if `cycling_zeros=True` or `cycling_sens_cals=True`.
    cycling_zeros : bool, default=True
        If True, locates auto-zero periods (type flag 1.0) to subtract dynamic baselines.
    cycling_sens_cals : bool, default=True
        If True, locates sensitivity cal periods (type flag 4.0) to calculate response factors.
        If False (or if cycling was not run), applies `static_sens_factor`.
    normalization : bool, default=False
        If True, normalizes peak areas against primary reagent ion signals.
    normalization_reference_values : array-like or None, default=None
        1D array of primary reagent ion signal counts (e.g., H3O+ or I-) used as the denominator.
        Required if `normalization=True`.
    static_target_value : float, default=1e6
        Normalization count scaling reference (e.g., 10^6 counts).
    cal_tank_conc : float, default=1000.0
        Concentration of the target compound in the calibration gas standard cylinder (ppbV).
    zero_flow : float or array-like, default=250.0
        Zero air dilution flow rate in sccm. Can be a constant float or a 1D time-series 
        monitor array (e.g., from ``d.tps2['Zero MFC monitor [sccm]']``).
    cal_flow : float or array-like, default=5.0
        Calibration standard gas flow rate in sccm. Can be a constant float or a 1D time-series 
        monitor array (e.g., from ``d.tps2['Cal MFC monitor [sccm]']``).
    static_sens_factor : float, default=1.0
        Fallback sensitivity conversion factor applied when cycling calibrations are absent or disabled.
    mode : {"interp", "median"}, default="interp"
        Baseline and sensitivity timeline modeling mode:
        * ``"interp"``: Linearly interpolates between event centers across time.
        * ``"median"``: Applies the global median value across all points.
    min_zero_points : int, default=5
        Minimum valid sample points required to accept a zero block (filters out aborted cycles).
    min_cal_points : int, default=6
        Minimum valid sample points required to accept a calibration challenge block.
    trim_zero_transients : int, default=2
        Number of leading points to discard from each zero block to allow inlet lines to flush.
    trim_cal_transients : int, default=2
        Number of leading points to discard from the challenge step while the signal rises.
    min_signal_delta : float, default=1.0
        Minimum required step increase (ions/s) during a calibration event. If the step is 
        below this floor, the compound is treated as absent from the calibration tank.
    plot_flag : bool, default=False
        If True, generates the 3-panel diagnostic quantification summary figure.
    output_dir : str or None, default=None
        Directory where summary plots are saved. Defaults to ``~/OpenTof``.
    plot_subdir : str or None, default=None
        Sub-folder prefix appended to output plot filenames.
    timestamps : array-like or None, default=None
        1D array of acquisition timestamps for x-axis time labeling on summary plots.
    standard_acq_mask : array-like of bool, or None, default=None
        Boolean mask marking ambient data vs. calibration/zero intervals.
    compound_name : str, default="Target Compound"
        Label or chemical formula used in plot titles, legends, and filenames.
    plot_filename : str or None, default=None
        Custom filename for the saved summary image.
    show_plot : bool, default=False
        If True, displays the summary figure interactively.
    **kwargs : dict
        Additional keyword arguments forwarded to the plotting backend.

    Returns
    -------
    ppb_bc : numpy.ndarray
        1D array of final background-corrected and calibrated mixing ratios in ppb.
    sensitivity_axis : numpy.ndarray
        1D array of dynamic or static sensitivity factors applied across time.
    background_corrected : numpy.ndarray
        1D array of signal values after zero subtraction (prior to sensitivity scaling).
    working_array : numpy.ndarray
        1D array of signal values prior to background subtraction (with normalization applied).

    Raises
    ------
    ValueError
        If `normalization=True` but `normalization_reference_values` is missing.
    """
    # --- Step 1: Reagent Ion Normalization ---
    # Normalization scales raw counts to a fixed primary ion count (e.g. 10^6) to remove ion source drift
    if normalization:
        if normalization_reference_values is None:
            raise ValueError("`normalization_reference_values` must be provided if `normalization` is True.")
        ref_arr = np.asarray(normalization_reference_values, dtype=float)
        # Prevent division by zero if reagent ion momentarily drops out
        valid_ref = np.where(ref_arr > 0, ref_arr, np.nan)
        correction_factor = static_target_value / valid_ref
        working_array = np.nan_to_num(peak_area_array * correction_factor, nan=0.0)
    else:
        working_array = np.asarray(peak_area_array, dtype=float).copy()

    # --- Step 2: Cycling Telemetry Inspection ---
    # Check if instrument cycling was active during this deployment
    has_valid_cycling = (
        cycling_dict is not None and 
        'type' in cycling_dict and 
        not (cycling_dict['type'] == -1).all()
    )
    cycling_flags_type = cycling_dict['type'] if has_valid_cycling else np.full(len(working_array), -1)

    # Pre-initialize baseline and diagnostic containers to guarantee clean references
    background_line = np.zeros_like(working_array, dtype=float)
    zero_diagnostics = []
    cal_diagnostics = []

    # --- Step 3: Background Subtraction (Auto-Zeros) ---
    auto_zeros = np.where(cycling_flags_type == 1.0)[0] if has_valid_cycling else np.array([])
    
    if cycling_zeros and len(auto_zeros) > 0:
        # Group contiguous zero indices into chunks; drop chunks that aborted immediately
        zero_chunks = _locate_index_chunks(auto_zeros, min_chunk_size=min_zero_points)
        if len(zero_chunks) > 0:
            zero_blocks = _assign_blocks_from_chunks(zero_chunks, working_array)
            
            # Execute background subtraction with defensive unpacking
            sub_res = _perform_background_subtraction(
                zero_blocks=zero_blocks, 
                peak_area_array=working_array, 
                mode=mode, 
                min_zero_points=min_zero_points,
                trim_transient_points=trim_zero_transients,
            )
            
            # Handle both 3-tuple return (bg_corr, bg_line, diag) and legacy 1-array return
            if isinstance(sub_res, tuple):
                background_corrected = sub_res[0]
                background_line = sub_res[1]
                if len(sub_res) > 2:
                    zero_diagnostics = sub_res[2]
            else:
                background_corrected = sub_res
                background_line = working_array - background_corrected
        else:
            warnings.warn("All auto-zero chunks fell below min_zero_points. Skipping dynamic zero subtraction.")
            background_corrected = working_array.copy()
            background_line = np.zeros_like(working_array, dtype=float)
    else:
        if cycling_zeros and not has_valid_cycling:
            warnings.warn("No cycling zero events found (type array is missing or all -1). Skipping dynamic zero subtraction.")
        background_corrected = working_array.copy()
        background_line = np.zeros_like(working_array, dtype=float)

    # --- Step 4: Sensitivity Calibration Modeling ---
    auto_sens_cals = np.where(cycling_flags_type == 4.0)[0] if has_valid_cycling else np.array([])

    if cycling_sens_cals and len(auto_sens_cals) > 0:
        # Group contiguous calibration challenge indices into chunks
        sens_cal_chunks = _locate_index_chunks(auto_sens_cals, min_chunk_size=min_cal_points)
        if len(sens_cal_chunks) > 0:
            sens_cal_blocks = _assign_blocks_from_chunks(sens_cal_chunks, working_array)
            
            # Execute calibration modeling with defensive unpacking
            cal_res = _get_cyclic_sensitivity_calibrations(
                sens_cal_blocks=sens_cal_blocks,
                background_corrected=background_corrected,
                cal_tank_conc=cal_tank_conc,
                zero_flow=zero_flow,
                cal_flow=cal_flow,
                mode=mode,
                static_sens_factor=static_sens_factor,
                min_cal_points=min_cal_points,
                min_signal_delta=min_signal_delta,
                trim_cal_transients=trim_cal_transients,
            )
            
            # Handle both 2-tuple return (sens_axis, diag) and legacy 1-array return
            if isinstance(cal_res, tuple):
                sensitivity_axis = cal_res[0]
                if len(cal_res) > 1:
                    cal_diagnostics = cal_res[1]
            else:
                sensitivity_axis = cal_res
        else:
            warnings.warn("All cal chunks fell below min_cal_points. Falling back to static_sens_factor.")
            sensitivity_axis = np.full_like(working_array, static_sens_factor, dtype=float)
    else:
        if cycling_sens_cals and not has_valid_cycling:
            warnings.warn("No calibration events found in cycling status. Using static_sens_factor.")
        sensitivity_axis = np.full_like(working_array, static_sens_factor, dtype=float)

    # --- Step 5: Final Concentration Calculation ---
    # Multiply net signal (ions/s) by calibration factor (ppb / (ions/s)) to yield mixing ratio (ppb)
    ppb_bc = background_corrected * sensitivity_axis

    # --- Step 6: Unified Summary Visualization ---
    if plot_flag:
        # Resolve plot directory with sub-directory prefix if provided
        final_plot_dir = output_dir if output_dir is not None else get_default_plot_dir()
        if plot_subdir:
            final_plot_dir = os.path.join(final_plot_dir, plot_subdir)

        plot_quantification_summary(
            timestamps=timestamps,
            working_array=working_array,
            background_line=background_line,
            background_corrected=background_corrected,
            sensitivity_axis=sensitivity_axis,
            ppb_bc=ppb_bc,
            zero_diagnostics=zero_diagnostics,
            cal_diagnostics=cal_diagnostics,
            standard_acq_mask=standard_acq_mask,
            compound_name=compound_name,
            mode=mode,
            output_dir=final_plot_dir,
            plot_filename=plot_filename,
            show_plot=show_plot
        )

    return ppb_bc, sensitivity_axis, background_corrected, working_array

def plot_quantification_summary(timestamps,
                                working_array,
                                background_line,
                                background_corrected,
                                sensitivity_axis,
                                ppb_bc,
                                zero_diagnostics=None,
                                cal_diagnostics=None,
                                standard_acq_mask=None,
                                compound_name="Target Compound",
                                mode="interp",
                                output_dir=None,
                                plot_filename=None,
                                show_plot=False):
    """
    Generate a 3-panel publication-ready diagnostic summary figure for signal quantification.

    This figure consolidates all steps of the OpenTof signal quantification pipeline
    into a single view. It allows for quick viewing of the quantification process, including
    auto-zero background subtraction and sensitivity calibration scaling into the final
    quantified mixing ratios (ppb), it will also display diagnostic information for 
    rejected or aborted cycles.

    Figure Panel Structure:
      - **Panel 1: Dynamic Baseline Floor & Auto-Zeros (Zoomed)**
        Displays the raw (or normalized) signal and the continuous zero baseline curve.
        The vertical axis automatically zooms to the baseline region (clamping out ambient
        plumes) so researchers can inspect the zero air floor. Individual auto-zero blocks
        are shaded:
          * Light Green: Accepted auto-zero events.
          * Light Orange (hatched): Aborted or prematurely truncated cycles (< `min_zero_points`).
          * Light Red (hatched): Statistical outlier blocks (> 4 * MAD from median).
        Skipped blocks include visual callout annotations detailing why the block was rejected
        (e.g., ``"Cut off (3 pts < 5)"`` or ``"Outlier (8.4 vs med 1.8 ± 2.1)"``).
      - **Panel 2: Net Signal, Cal Challenges, and Sensitivity Tracking**
        Displays the background-corrected signal (ions/s) on the left axis with calibration
        baseline (red dots) and challenge response steps (blue dots). The secondary right
        axis shows the continuous sensitivity calibration timeline ($S_c$, in ppb / (ions/s))
        with diamond markers indicating accepted calibration events. Skipped challenge events
        are annotated with callouts (e.g., ``"No Step (ΔS < 1)"`` or ``"Aborted"``).
      - **Panel 3: Final Calibrated Ambient Timeseries**
        Displays the fully quantified mixing ratio in parts-per-billion (ppb). Valid ambient
        sampling periods are plotted in bold green, while non-standard acquisition intervals
        (zeros and calibrations) are masked and shaded in muted gray. An inset metadata box
        reports ambient timeseries summary statistics (mean, median, and 95th percentile).

    Parameters
    ----------
    timestamps : array-like of datetime, or None
        1D array of acquisition timestamps matching `working_array` length. If ``None`` or
        dimensionally mismatched, the x-axis falls back to spectrum sample indices (MS Index).
    working_array : numpy.ndarray of float
        1D array of signal intensity (ions/s) prior to background subtraction, with primary
        reagent ion normalization applied if active.
    background_line : numpy.ndarray of float
        1D continuous baseline curve (ions/s) modeled from zero air periods.
    background_corrected : numpy.ndarray of float
        1D net signal array (ions/s) after subtracting `background_line`.
    sensitivity_axis : numpy.ndarray of float
        1D time-series sensitivity calibration factor axis (ppb / (ions/s)).
    ppb_bc : numpy.ndarray of float
        1D array of final calibrated compound concentrations in ppb:
        ppb_bc = background_corrected * sensitivity_axis
    zero_diagnostics : list of dict, optional
        Diagnostic metadata for every auto-zero block generated by `_perform_background_subtraction`.
        Required to render shaded spans, accepted markers, and skip callouts in Panel 1.
    cal_diagnostics : list of dict, optional
        Diagnostic metadata for every calibration challenge block generated by
        `_get_cyclic_sensitivity_calibrations`. Required to render response steps and cal callouts in Panel 2.
    standard_acq_mask : numpy.ndarray of bool, optional
        1D boolean mask where ``True`` marks valid ambient sampling writebufs and ``False``
        marks non-standard acquisition (zeros, calibrations, or corrupt scans). Used to isolate
        ambient data and calculate summary statistics in Panel 3.
    compound_name : str, default="Target Compound"
        Formula string or compound label (e.g., ``"C6H6+"``) used for titles and filenames.
    mode : {"interp", "median"}, default="interp"
        Baseline and sensitivity modeling mode used across time.
    output_dir : str or None, default=None
        Directory path where the generated figure will be saved. Defaults to `OpenTof` default plot dir.
    plot_filename : str or None, default=None
        Custom image filename. Defaults to ``"quantification_summary_<compound_name>.png"``.
    show_plot : bool, default=False
        If True, displays the figure interactively via `plt.show()`.

    Returns
    -------
    None
        Exports the 3-panel figure to disk as a high-resolution PNG image.
    """
    if output_dir is None:
        output_dir = get_default_plot_dir()
    ensure_dir(output_dir)

    if plot_filename is None:
        clean_name = str(compound_name).replace("+", "_").replace("-", "_")
        plot_filename = f"quantification_summary_{clean_name}.png"

    if timestamps is None or len(timestamps) != len(working_array):
        x = np.arange(len(working_array))
        x_label = "Sample Index (MS Index)"
    else:
        x = timestamps
        x_label = "Time (UTC)"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, 
        figsize=(14, 10), 
        sharex=True, 
        dpi=150,
        gridspec_kw={'height_ratios': [1.3, 1.1, 1.0]}
    )
    fig.suptitle(f"Quantification Diagnostic: {compound_name} (Mode: {mode})", fontsize=14, y=0.995)

    # =========================================================================
    # PANEL 1: DYNAMIC BASELINE ZEROING (ZOOMED TO ZERO REGION)
    # =========================================================================
    ax1.plot(x, working_array, color="slategray", alpha=0.35, lw=0.75, label="Signal (Raw/Normalized)")
    ax1.plot(x, background_line, color="crimson", lw=1.8, linestyle="--", label=f"Applied Baseline ({mode})")

    # Draw shaded time spans and markers for every zero block
    if zero_diagnostics:
        for z in zero_diagnostics:
            s_idx = int(z['start_idx'])
            e_idx = int(z['end_idx'])
            c_idx = int(z['center'])
            x_pos = x[c_idx]
            z_mean = z['mean']

            if z['status'] == 'accepted':
                # Light green span for accepted zeros
                ax1.axvspan(x[s_idx], x[e_idx], color="forestgreen", alpha=0.18, zorder=2)
                ax1.scatter(x_pos, z_mean, color="forestgreen", s=50, zorder=5, edgecolors="black",
                            label="Accepted Zero" if "Accepted Zero" not in ax1.get_legend_handles_labels()[1] else "")
            
            elif z['status'] == 'skipped_too_short':
                # Light orange hatched span for cut-off zeros
                ax1.axvspan(x[s_idx], x[e_idx], color="darkorange", alpha=0.25, hatch="//", zorder=2)
                ax1.scatter(x_pos, z_mean, color="darkorange", marker="^", s=65, zorder=6, edgecolors="black",
                            label="Skipped: Aborted" if "Skipped: Aborted" not in ax1.get_legend_handles_labels()[1] else "")
            
            elif z['status'] == 'skipped_outlier':
                # Light red hatched span for statistical outliers
                ax1.axvspan(x[s_idx], x[e_idx], color="crimson", alpha=0.25, hatch="\\\\", zorder=2)
                ax1.scatter(x_pos, z_mean, color="crimson", marker="X", s=65, zorder=6, edgecolors="black",
                            label="Skipped: Outlier" if "Skipped: Outlier" not in ax1.get_legend_handles_labels()[1] else "")

    # --- DYNAMIC Y-AXIS ZOOM CALCULATION ---
    all_zero_means = [z['mean'] for z in zero_diagnostics if np.isfinite(z['mean'])] if zero_diagnostics else []
    accepted_means = [z['mean'] for z in zero_diagnostics if z.get('status') == 'accepted' and np.isfinite(z['mean'])] if zero_diagnostics else []
    finite_bg = background_line[np.isfinite(background_line)]

    if all_zero_means and len(finite_bg) > 0:
        base_min = min(all_zero_means + [float(np.min(finite_bg))])
        base_max = max(accepted_means) if accepted_means else max(all_zero_means)

        # Soft cap to prevent wild aborted zeros (during plumes) from squishing the scale
        med_ref = float(np.median(accepted_means)) if accepted_means else float(np.median(all_zero_means))
        mad_ref = float(median_absolute_deviation(np.array(accepted_means))) if accepted_means else 1.0
        soft_cap = med_ref + 10.0 * max(mad_ref, 1.0)

        # Moderate skipped zeros stay in frame; extreme ones are clipped
        moderate_skipped = [z['mean'] for z in zero_diagnostics if z.get('status') != 'accepted' and z['mean'] <= soft_cap]
        plot_y_top = max([base_max] + moderate_skipped) if moderate_skipped else base_max

        y_span = max(plot_y_top - base_min, 1.0)
        y_lower = base_min - 0.2 * y_span
        y_upper = plot_y_top + 0.65 * y_span  # 65% headroom above highest zero point for callout boxes
        ax1.set_ylim(y_lower, y_upper)

        # Draw callout annotations with arrows pointing to the markers
        for z in zero_diagnostics:
            if z['status'] != 'accepted':
                c_idx = int(z['center'])
                x_pos = x[c_idx]
                z_mean = z['mean']
                is_offscale = z_mean > y_upper

                color = "darkorange" if z['status'] == 'skipped_too_short' else "crimson"
                target_y = y_upper - (0.08 * y_span) if is_offscale else z_mean
                prefix = f"Zero #{abs(z['block_id'])}\n"
                
                ax1.annotate(
                    f"{prefix}{z['reason']}",
                    xy=(x_pos, target_y),
                    xytext=(0, 22 if not is_offscale else -25), textcoords="offset points",
                    ha="center", fontsize=7.5, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
                )

    ax1.set_ylabel("Signal (ions/s)", fontweight="bold")
    ax1.set_title("1. Auto-Zero Baseline Floor (Zoomed to Zero Region)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize="small", ncol=2)

    # =========================================================================
    # PANEL 2: NET SIGNAL, CAL CHALLENGES, AND SENSITIVITY TRACKING
    # =========================================================================
    ax2.plot(x, background_corrected, color="black", alpha=0.6, lw=0.8, label="Net Signal (Zero-Subtracted)")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)

    ax2_sens = ax2.twinx()
    ax2_sens.plot(x, sensitivity_axis, color="dodgerblue", lw=1.6, linestyle="-.", label="Sensitivity Axis")
    ax2_sens.set_ylabel("Sensitivity (ppb / (ions/s))", color="dodgerblue", fontweight="bold")
    ax2_sens.tick_params(axis="y", labelcolor="dodgerblue")

    if cal_diagnostics:
        for c in cal_diagnostics:
            c_idx = int(c['center'])
            x_pos = x[c_idx]

            if c['status'] == 'accepted':
                if len(c.get('low_idx', [])) > 0:
                    ax2.scatter(x[c['low_idx']], background_corrected[c['low_idx']], color="red", s=10, zorder=4,
                                label="Cal Baseline" if "Cal Baseline" not in ax2.get_legend_handles_labels()[1] else "")
                if len(c.get('high_idx', [])) > 0:
                    ax2.scatter(x[c['high_idx']], background_corrected[c['high_idx']], color="blue", s=10, zorder=4,
                                label="Cal Challenge" if "Cal Challenge" not in ax2.get_legend_handles_labels()[1] else "")
                ax2_sens.scatter(x_pos, c['sens'], color="dodgerblue", marker="D", s=40, zorder=5, edgecolors="black")

            else:
                reason_str = "No Step (ΔS < 1)" if c['status'] == 'skipped_no_response' else ("Aborted (<6 pts)" if c['status'] == 'skipped_too_short' else "Outlier Sens")
                ax2.annotate(
                    f"Cal #{abs(c['block_id'])}\n{reason_str}",
                    xy=(x_pos, 0),
                    xytext=(0, 22), textcoords="offset points",
                    ha="center", fontsize=7.5, fontweight="bold", color="navy",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="navy", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="navy", lw=1.1)
                )

    ax2.set_ylabel("Net Signal (ions/s)", fontweight="bold")
    ax2.set_title("2. Calibration Gas Response & Sensitivity Calibration Factor", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_sens.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize="small", ncol=2)

    # =========================================================================
    # PANEL 3: FINAL QUANTIFIED CONCENTRATION (PPB)
    # =========================================================================
    if standard_acq_mask is not None and len(standard_acq_mask) == len(ppb_bc):
        valid_ppb = ppb_bc[standard_acq_mask]
        valid_x = x[standard_acq_mask]
        ax3.plot(valid_x, valid_ppb, color="forestgreen", lw=1.2, label=f"{compound_name} Ambient (ppb)")

        non_std_mask = ~standard_acq_mask
        if np.any(non_std_mask):
            ax3.plot(x[non_std_mask], ppb_bc[non_std_mask], color="gray", alpha=0.25, lw=0.6, label="Cycle Data (Non-Ambient)")
    else:
        valid_ppb = ppb_bc
        ax3.plot(x, ppb_bc, color="forestgreen", lw=1.2, label=f"{compound_name} (ppb)")

    finite_ppb = valid_ppb[np.isfinite(valid_ppb)]
    if len(finite_ppb) > 0:
        mean_ppb = np.mean(finite_ppb)
        median_ppb = np.median(finite_ppb)
        p95_ppb = np.percentile(finite_ppb, 95)
        stat_text = (
            f"Mean: {mean_ppb:.2f} ppb\n"
            f"Median: {median_ppb:.2f} ppb\n"
            f"p95: {p95_ppb:.2f} ppb"
        )
        ax3.text(
            0.015, 0.90, stat_text,
            transform=ax3.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85)
        )

    ax3.set_ylabel("Mixing Ratio (ppb)", fontweight="bold")
    ax3.set_xlabel(x_label)
    ax3.set_title("3. Final Calibrated Ambient Timeseries", fontsize=10.5, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close()



# TODO is it possible to auto determine sensisitivity calibration periods somehow?