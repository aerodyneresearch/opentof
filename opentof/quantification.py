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

# quantization.py
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from tqdm.auto import tqdm
import os
from itertools import product

from sklearn.cluster import KMeans

from opentof.utils import (
    return_mass,
    median_absolute_deviation,
    ensure_dir, 
    get_default_plot_dir
)


# long term TODO: Maybe in this .py file is where we can keep dictionary of all or 
#  few standard k_ptr rates? or perhaps build fuctions into OpenTof 
#  for KPTR estimation based on atomic principles (calling psi4??) for example:
#  proton affinity, dipolemoment, polarizability. Seems like a longer project?


# Victor Note: LEGACY FUNCTION SPECIALIZED TO CDPHE OPERATIONS. CAN LIKELY BE MADE MORE GENERAL AT SOME POINT
def load_TWEB_sensitivity_calibration_file(file_path, skiprows=9, plot_flag=False,
                                      output_dir=None,
                                      plot_filename=None):
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

    # Generate fitted values
    x_kPTR_extended = np.linspace(1.50, 4.25, 100)
    y_kPTR_extended = linear_func(x_kPTR_extended, *popt_lin)

    # Define extended range for transmission curve
    x_trans_fit_extended = np.linspace(12, 400, 4000)  
    y_trans_fit_extended = logistic(x_trans_fit_extended, *popt_sig) #if not np.isnan(x_half) else np.zeros_like(x_trans_fit_extended)

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
        # axs[0].legend()
        axs[0].grid(True)

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
        # axs[1].legend()
        axs[1].grid(True)

        # Annotate x_half and rate
        axs[1].annotate(f"x_half: {x_half:.2f}\nRate: {rate:.2f}\nBase: {base:.2f}\nMax: {max_val:.2f}", 
                        xy=(0.80, 0.05), xycoords='axes fraction', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.6))

        # Adjust layout and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
        plt.show()
        plt.close()

    # Store parameters
    sigmoid_fit_params = {"x_half": x_half, "rate": rate, "base": base, "max": max_val}
    linear_fit_params = {"slope": slope, "intercept": intercept}

    sens_dict = {}
    for i in range(len(sens_file)):
        if sens_file.iloc[i].iloc[13] == 1.00:
            sens_dict[sens_file.iloc[i].iloc[0]] = {
                "Sensitivity" : sens_file.iloc[i].iloc[2],
                "kPTR" : np.nan,
                "Transmission" : np.nan,
                }
        else:
            sens_dict[sens_file.iloc[i].iloc[0]] = {
                "Sensitivity" : sens_file.iloc[i].iloc[2],
                "kPTR" : sens_file.iloc[i].iloc[13],
                "Transmission" : sens_file.iloc[i].iloc[14],
                }

    def sensitivity_kptr_linear(k_ptr):
        return linear_func(k_ptr, *popt_lin)

    def transmission_mz_sigmoid(mz):
        return logistic(mz, *popt_sig)

    # Return the dictionary itself and functions defined based on sensitivities
    return sens_dict, sensitivity_kptr_linear, transmission_mz_sigmoid


def merge_chunks(n_chunks, output_dir):
    """
    Merge sequential chunked CSV result files from disk into a unified DataFrame.

    Iterates through numbered chunk result files (e.g., ``results_chunk_000.csv``) 
    located in `output_dir`, loads existing files into pandas DataFrames, and 
    concatenates them vertically into a single cohesive output DataFrame.

    Parameters
    ----------
    n_chunks : int
        Total number of expected chunk files to search for.
    output_dir : str or pathlib.Path
        Directory path containing the chunk CSV files.

    Returns
    -------
    final_df : pandas.DataFrame
        Concatenated DataFrame containing aggregated rows across all discovered chunk files.
    """
    # Container list to store DataFrames loaded from valid chunk CSV files
    all_dfs = []
    
    # Iterate through expected chunk indices
    for chunk_idx in range(n_chunks):
        # Format standardized chunk filename with 3-digit zero padding (e.g., results_chunk_001.csv)
        chunk_file = os.path.join(output_dir, f"results_chunk_{chunk_idx:03d}.csv")
        
        # Read CSV file into memory if it exists on disk
        if os.path.exists(chunk_file):
            all_dfs.append(pd.read_csv(chunk_file))
            
    # Concatenate all loaded chunk DataFrames vertically along axis 0, resetting index
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    return final_df


def _locate_index_chunks(filterd_auto_cycling_arr, gap_tol=5):
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

    Returns
    -------
    all_chunks : list of list of int
        List of sub-lists, where each sub-list contains contiguous or closely spaced 
        indices belonging to a single event block.
    """
    # Guard clause: Return empty list if input index array is empty
    if len(filterd_auto_cycling_arr) == 0:
        return []

    all_chunks = []
    # Initialize the first chunk with the starting index element
    chunk = [filterd_auto_cycling_arr[0]]
    prev_ms_i = filterd_auto_cycling_arr[0]
    
    # Iterate through remaining indices starting from the second element
    for ms_i in filterd_auto_cycling_arr[1:]:
        # Append to current active chunk if index step is within gap tolerance
        if (ms_i - prev_ms_i) < gap_tol:
            chunk.append(ms_i)
        else:
            # Index gap breached: save completed chunk and start a new chunk block
            all_chunks.append(chunk)
            chunk = [ms_i]
            
        # Update previous index tracker for next iteration comparison
        prev_ms_i = ms_i
    
    # Append the final remaining chunk after loop termination
    if chunk:
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
    
    # Initialize output block array filled with 0s (covers initial unassigned prefix)
    block_array = np.zeros(array_length, dtype=np.int32)
    
    last_chunk_end_index = 0
    
    # Counter for assigning negative chunk IDs (-1, -2, -3, ...)
    chunk_counter = 1
    
    # Counter for assigning positive gap IDs (1, 2, 3, ...)
    gap_id = 1 

    # Iterate through each event chunk list
    for chunk in chunks_list:
        if not chunk:
            continue
            
        first_index = int(chunk[0])
        last_index = int(chunk[-1])
        
        # --- A. Assign Negative Chunk ID to Active Event ---
        # Mark all sample channels inside active chunk with negative chunk ID (-1, -2, etc.)
        chunk_id_value = -chunk_counter
        block_array[first_index : last_index + 1] = chunk_id_value
        
        # Increment active chunk counter
        chunk_counter += 1 
        
        # --- B. Assign Positive Gap ID to Gap Preceding Current Chunk ---
        # Assign gap ID to standard acquisition channels between previous and current chunk
        if last_chunk_end_index > 0 and last_chunk_end_index < first_index:
            block_array[last_chunk_end_index : first_index] = gap_id
            gap_id += 1
            
        # --- C. Update Boundary Tracker ---
        last_chunk_end_index = last_index + 1

    # --- D. Assign Final Gap ID ---
    # Mark trailing gap from end of last chunk to array boundary with current gap ID
    if last_chunk_end_index < array_length:
        block_array[last_chunk_end_index : array_length] = gap_id
        
    return block_array


def _perform_background_subtraction(zero_blocks, peak_area_array, mode="interp", plot_flag=False,
                                    output_dir=None, plot_subdir=None):
    """
    Subtract dynamic or global baseline backgrounds using auto-zero periods.

    Filters outlier points during auto-zero periods (negative block IDs), calculates mean 
    background levels for each zero chunk, constructs a continuous 1D background baseline 
    (via linear interpolation or global median), and subtracts the baseline from active 
    tracking periods (positive block IDs) while preserving raw zero signals.

    Parameters
    ----------
    zero_blocks : numpy.ndarray
        1D integer block ID array created by :func:`_assign_blocks_from_chunks`. 
        Negative IDs mark active auto-zero periods; positive IDs mark sample acquisition gaps.
    peak_area_array : numpy.ndarray
        1D array of un-subtracted peak area or amplitude signal values.
    mode : {"interp", "median"}, default="interp"
        Background construction mode:
        
        * ``"interp"``: Linearly interpolates a continuous baseline between auto-zero chunk means over time.
        * ``"median"``: Applies a constant baseline using the global median of all auto-zero chunk averages.
    plot_flag : bool, default=False
        If ``True``, exports diagnostic background correction figures for each zero block.
    output_dir : str or None, default=None
        Export directory path for saved plots. Defaults to OpenTof default plot dir if ``None``.
    plot_subdir : str or None, default=None
        Optional sub-folder prefix appended to output plot filenames.

    Returns
    -------
    full_background_subtraction : numpy.ndarray
        1D background-corrected signal array matching `peak_area_array` length.

    Raises
    ------
    ValueError
        If `mode` is not one of ``"interp"`` or ``"median"``.
    """
    # Validate requested background calculation mode
    if mode not in ["interp", "median"]:
        raise ValueError("mode must be either 'interp' or 'median'")

    full_background_subtraction = np.zeros_like(peak_area_array)
    array_length = len(peak_area_array)
    unique_zero_blocks = np.unique(zero_blocks)
    
    # Isolate active auto-zero block IDs (negative integers)
    neg_blocks = [b for b in unique_zero_blocks if b < 0]

    center_indices = []
    zero_values = []
    block_zero_map = {}

    # --- PASS 1: Calculate Clean Zero Mean & Center Index for Each Zero Block ---
    for b_id in neg_blocks:
        auto_zero_idx_temp = np.where(zero_blocks == b_id)[0]
        if len(auto_zero_idx_temp) == 0:
            continue

        # Outlier filtering: calculate local median and Median Absolute Deviation (MAD)
        pre_filter_threshold_median = np.median(peak_area_array[auto_zero_idx_temp])
        pre_filter_threshold_deviation = median_absolute_deviation(peak_area_array[auto_zero_idx_temp])

        # Define 4*MAD statistical acceptance bounds around local median
        upper_lim = pre_filter_threshold_median + (4 * pre_filter_threshold_deviation)
        lower_lim = pre_filter_threshold_median - (4 * pre_filter_threshold_deviation)

        # Filter out extreme noise spikes or transient points inside zero period
        auto_zero_idx = auto_zero_idx_temp[(peak_area_array[auto_zero_idx_temp] >= lower_lim) & 
                                           (peak_area_array[auto_zero_idx_temp] <= upper_lim)]

        # Calculate mean zero signal from clean points (fallback to all points if all filtered)
        if auto_zero_idx.size > 0:
            current_block_zero = np.mean(peak_area_array[auto_zero_idx])
        else:
            current_block_zero = np.mean(peak_area_array[auto_zero_idx_temp])

        # Store block center index for time-series baseline interpolation
        center_idx = np.median(auto_zero_idx_temp)
        center_indices.append(center_idx)
        zero_values.append(current_block_zero)
        
        # Cache block information for downstream plotting context
        block_zero_map[b_id] = {
            'current_zero': current_block_zero,
            'auto_zero_idx': auto_zero_idx,
            'auto_zero_idx_temp': auto_zero_idx_temp
        }

    # --- PASS 2: Construct Continuous Background Baseline Line ---
    if not zero_values:
        # Fallback to zero baseline if no zero periods were found
        background_line = np.zeros(array_length)
    elif mode == "interp":
        if len(zero_values) == 1:
            # Single zero block: apply constant flat baseline across whole run
            background_line = np.full(array_length, zero_values[0])
        else:
            # Multiple zero blocks: linearly interpolate continuous baseline across time
            sort_order = np.argsort(center_indices)
            background_line = np.interp(
                np.arange(array_length), 
                np.array(center_indices)[sort_order], 
                np.array(zero_values)[sort_order]
            )
    elif mode == "median":
        # Apply constant global median zero level across entire dataset
        global_median_zero = np.median(zero_values)
        background_line = np.full(array_length, global_median_zero)

    # --- PASS 3: Apply Subtraction to Active Signal Periods ---
    for block_id in unique_zero_blocks:
        idx = np.where(zero_blocks == block_id)[0]

        # Preserve raw un-subtracted signal during auto-zero periods (negative IDs)
        if block_id <= 0:
            full_background_subtraction[idx] = peak_area_array[idx]
            continue

        # Subtract baseline curve from standard sample acquisition tracking blocks (positive IDs)
        full_background_subtraction[idx] = peak_area_array[idx] - background_line[idx]

        # Render visual diagnostic plot if requested
        if plot_flag:
            if output_dir is None:
                output_dir = get_default_plot_dir()
            ensure_dir(output_dir)
            
            prefix = f"{plot_subdir}_" if plot_subdir else ""
            plot_filename = f"{prefix}bg_subtraction_block_{block_id}.png"

            neg_id = -block_id
            if neg_id in block_zero_map:
                info = block_zero_map[neg_id]
                start = max(info['auto_zero_idx_temp'][0] - 100, 0)
                end = min(info['auto_zero_idx_temp'][-1] + 100, array_length - 1)
                extended_idx = np.arange(start, end + 1)
                applied_zero_avg = np.mean(background_line[idx])
                
                print(f"zero block: {block_id}")
                print(f"Current Zero: {info['current_zero']:.2f}, Applied ({mode}): {applied_zero_avg:.2f}")
                
                plt.figure(figsize=(10, 4))
                plt.plot(peak_area_array[extended_idx], color='orange', label='Context')
                plt.plot(peak_area_array[info['auto_zero_idx']], color='blue', label='Used for Calc')
                plt.axhline(applied_zero_avg, linestyle="--", color="red", label=f'Applied Zero: {applied_zero_avg:.1f}')
                plt.title(f"Background Correction: Block {block_id} ({mode} mode)")
                plt.legend()
                plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
                plt.show()
                plt.close()

    return full_background_subtraction

def _cluster_sensitivity_cal_signal(data_subset):
    """
    Partition a sensitivity calibration signal subset into high and low states using K-Means.

    Applies a 2-cluster 1D K-Means model to separate background/baseline periods ('low') 
    from active calibration challenge periods ('high') within a calibration event block.

    Parameters
    ----------
    data_subset : array-like
        1D array or list of signal intensity values extracted during a sensitivity 
        calibration event window.

    Returns
    -------
    low_mask : numpy.ndarray of bool
        1D boolean array where ``True`` indicates baseline/low-signal data points.
    high_mask : numpy.ndarray of bool
        1D boolean array where ``True`` indicates active calibration challenge/high-signal data points.
    """
    # Guard clause: Return all True for low and all False for high if array has fewer than 2 points
    if len(data_subset) < 2:
        return np.ones(len(data_subset), dtype=bool), np.zeros(len(data_subset), dtype=bool)

    # Reshape 1D input array to a 2D column vector (N, 1) required by Scikit-Learn KMeans API
    data_reshaped = np.asarray(data_subset).reshape(-1, 1)
    
    # Fit 2-cluster K-Means model to separate the two intensity states (Baseline vs. Challenge)
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(data_reshaped)
    
    # Identify which cluster label corresponds to the lower signal intensity mean
    low_cluster_label = np.argmin(kmeans.cluster_centers_)
    
    # Generate boolean mask for low cluster points
    low_mask = (kmeans.labels_ == low_cluster_label)
    
    # Invert low mask to yield high cluster points
    high_mask = ~low_mask
    
    return low_mask, high_mask

def _get_cyclic_sensitivity_calibrations(sens_cal_blocks, 
                                         background_corrected, 
                                         cal_tank_conc=995,
                                         zero_flow=250,
                                         cal_flow=5,
                                         mode="interp",
                                         plot_flag=False,
                                         output_dir=None,
                                         plot_subdir=None):
    """
    Generate a continuous time-series sensitivity calibration factor axis.

    Processes active sensitivity calibration blocks (negative IDs), clusters baseline versus 
    challenge signal states using :func:`_cluster_sensitivity_cal_signal`, calculates response 
    factors based on gas standard dilution equations, and interpolates or averages response 
    factors across the entire acquisition timeline.

    Parameters
    ----------
    sens_cal_blocks : numpy.ndarray
        1D integer block ID array created by :func:`_assign_blocks_from_chunks`. 
        Negative IDs mark active calibration challenge periods; positive IDs mark sample acquisition gaps.
    background_corrected : numpy.ndarray
        1D background-subtracted signal intensity array.
    cal_tank_conc : float, default=995
        Concentration of the calibration standard in the gas cylinder (e.g., ppbV).
    zero_flow : float, default=250
        Dilution zero air flow rate in standard cubic centimeters per minute (sccm).
    cal_flow : float, default=5
        Calibration gas standard flow rate in sccm.
    mode : {"interp", "median"}, default="interp"
        Sensitivity interpolation mode across time:
        
        * ``"interp"``: Linearly interpolates sensitivity factors between calibration block midpoints over time.
        * ``"median"``: Applies a constant sensitivity factor using the global median of all calibration blocks.
    plot_flag : bool, default=False
        If ``True``, renders and exports diagnostic sensitivity calibration figures for each block.
    output_dir : str or None, default=None
        Export directory path for saved plots. Defaults to OpenTof default plot dir if ``None``.
    plot_subdir : str or None, default=None
        Optional sub-folder prefix appended to output plot filenames.

    Returns
    -------
    sensitivity_calibration_axis : numpy.ndarray
        1D array matching `background_corrected` length, containing dynamic sensitivity 
        multipliers used to convert instrument signal to concentration units (e.g., ppbV).

    Raises
    ------
    ValueError
        If `mode` is not one of ``"interp"`` or ``"median"``.
    """
    # Validate requested sensitivity interpolation mode
    if mode not in ["interp", "median"]:
        raise ValueError("mode must be either 'interp' or 'median'")

    array_length = len(background_corrected)
    unique_sens_cal_blocks = np.unique(sens_cal_blocks)
    
    # Isolate active sensitivity calibration block IDs (negative integers)
    neg_blocks = [b for b in unique_sens_cal_blocks if b < 0]

    center_indices = []
    sens_values = []
    block_sens_map = {}

    # --- PASS 1: Calculate Sensitivity Response Factor for Each Active Cal Block ---
    for b_id in neg_blocks:
        auto_sens_cal_idx = np.where(sens_cal_blocks == b_id)[0]
        if len(auto_sens_cal_idx) == 0:
            continue

        # Slice signal subset for current calibration block
        block_signal = background_corrected[auto_sens_cal_idx]

        # Cluster local block signal into low (baseline) and high (challenge) states
        low_mask, high_mask = _cluster_sensitivity_cal_signal(block_signal)
        high_indices_local = np.where(high_mask)[0]
        
        # Trim transient response edge at start of challenge step if sufficient points exist
        if len(high_indices_local) >= 2:
            split_point_local = high_indices_local[1]
            calc_signal_high = block_signal[split_point_local:]
            auto_sens_cal_idx_low = auto_sens_cal_idx[:split_point_local]
            auto_sens_cal_idx_high = auto_sens_cal_idx[split_point_local:]
        else:
            calc_signal_high = block_signal[high_mask]
            auto_sens_cal_idx_low = auto_sens_cal_idx[low_mask]
            auto_sens_cal_idx_high = auto_sens_cal_idx[high_mask]

        # Calculate sensitivity factor via standard gas dilution equation:
        # Diluted Conc = (cal_flow / (zero_flow + cal_flow)) * cal_tank_conc
        # Sensitivity = Diluted Conc / Mean Challenge Signal
        current_sensitivity = None
        if len(calc_signal_high) > 0:
            auto_sens_cal_avg = np.mean(calc_signal_high)
            if auto_sens_cal_avg != 0:
                current_sensitivity = ((cal_flow / (zero_flow + cal_flow)) * cal_tank_conc) / auto_sens_cal_avg

        # Fallback to previous valid sensitivity value if current block calculation failed
        if current_sensitivity is None:
            if len(sens_values) > 0:
                current_sensitivity = sens_values[-1]
            else:
                continue

        # Record block midpoint index for time-series interpolation
        center_idx = np.median(auto_sens_cal_idx)
        center_indices.append(center_idx)
        sens_values.append(current_sensitivity)

        # Cache block information for diagnostic plotting
        block_sens_map[b_id] = {
            'current_sensitivity': current_sensitivity,
            'auto_sens_cal_idx': auto_sens_cal_idx,
            'auto_sens_cal_idx_low': auto_sens_cal_idx_low,
            'auto_sens_cal_idx_high': auto_sens_cal_idx_high,
            'block_signal': block_signal
        }

    # --- PASS 2: Construct Continuous Time-Series Sensitivity Axis ---
    if not sens_values:
        # Fallback to zero axis if no sensitivity calibration blocks were found
        sensitivity_calibration_axis = np.zeros(array_length)
    elif mode == "interp":
        if len(sens_values) == 1:
            # Single calibration block: apply constant flat sensitivity factor across whole run
            sensitivity_calibration_axis = np.full(array_length, sens_values[0])
        else:
            # Multiple calibration blocks: linearly interpolate sensitivity factors across time
            sort_order = np.argsort(center_indices)
            sensitivity_calibration_axis = np.interp(
                np.arange(array_length),
                np.array(center_indices)[sort_order],
                np.array(sens_values)[sort_order]
            )
    elif mode == "median":
        # Apply constant global median sensitivity factor across entire dataset
        global_median_sens = np.median(sens_values)
        sensitivity_calibration_axis = np.full(array_length, global_median_sens)

    # --- PASS 3: Render Diagnostic Plots ---
    if plot_flag:
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)

        for block_id in unique_sens_cal_blocks:
            if block_id <= 0:
                continue
            neg_id = -block_id
            if neg_id in block_sens_map:
                prefix = f"{plot_subdir}_" if plot_subdir else ""
                plot_filename = f"{prefix}sens_cal_block_{block_id}.png"

                info = block_sens_map[neg_id]
                idx = np.where(sens_cal_blocks == block_id)[0]
                block_sensitivity = np.mean(sensitivity_calibration_axis[idx])
                
                plt.figure(figsize=(10, 4))
                plt.plot(info['auto_sens_cal_idx'], info['block_signal'], color='gray', alpha=0.5, label='Full Block')
                
                # Scatter clustered low (baseline) and high (challenge) points
                if len(info['auto_sens_cal_idx_low']) > 0:
                    plt.scatter(info['auto_sens_cal_idx_low'], background_corrected[info['auto_sens_cal_idx_low']], color='red', s=10)
                if len(info['auto_sens_cal_idx_high']) > 0:
                    plt.scatter(info['auto_sens_cal_idx_high'], background_corrected[info['auto_sens_cal_idx_high']], color='blue', s=10)
                    
                plt.title(f"Block {block_id} | Sens: {block_sensitivity:.4f} ({mode} mode)")
                plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
                plt.show()
                plt.close()

    return sensitivity_calibration_axis

def quantify_signal(peak_area_array, cycling_dict=None, 
                    cycling_zeros=True, cycling_sens_cals=True, normalization=False,
                    normalization_reference_values=None, static_target_value=1e6,
                    cal_tank_conc=1000, zero_flow=250, cal_flow=5, 
                    static_sens_factor=1.0,
                    mode="interp", plot_flag=False,
                    output_dir=None,
                    plot_subdir=None):
    """
    Quantify a peak area time-series into concentration units (ppb).

    Supports optional primary reagent ion normalization, dynamic auto-zero background 
    subtraction, and dynamic sensitivity calibration scaling (or static scaling factors) 
    modeled continuously across the acquisition timeline.

    Parameters
    ----------
    peak_area_array : numpy.ndarray
        1D array of writebuf-aligned peak area or amplitude signal values for a target compound.
    cycling_dict : dict or None, default=None
        Deployment instrument cycling dictionary containing a `'type'` array of status flags. 
        Required if `cycling_zeros=True` or `cycling_sens_cals=True`.
    cycling_zeros : bool, default=True
        If ``True``, locates auto-zero periods (type flag ``1.0``) to calculate and subtract a dynamic background.
    cycling_sens_cals : bool, default=True
        If ``True``, locates sensitivity calibration periods (type flag ``4.0``) to dynamically calculate response factors.
        If ``False``, applies `static_sens_factor` as a constant multiplier across all points.
    normalization : bool, default=False
        If ``True``, normalizes raw peak areas against primary reagent ion signals (`normalization_reference_values`).
    normalization_reference_values : numpy.ndarray or None, default=None
        1D array of primary reagent ion signal counts (e.g., $H_3O^+$ or $I^- + I(H_2O)^-$) 
        used as the normalization denominator. Required if `normalization=True`.
    static_target_value : float, default=1e6
        Standardized primary ion count limit used to scale normalized signals (e.g., $10^6$ counts).
    cal_tank_conc : float, default=1000
        Concentration of the calibration gas standard in the cylinder (e.g., ppbV).
    zero_flow : float, default=250
        Zero air dilution flow rate in standard cubic centimeters per minute (sccm).
    cal_flow : float, default=5
        Calibration standard gas flow rate in sccm.
    static_sens_factor : float, default=1.0
        Static sensitivity conversion multiplier used if `cycling_sens_cals=False`.
    mode : {"interp", "median"}, default="interp"
        Modeling mode across time for zero subtraction and sensitivity calibration:
        
        * ``"interp"``: Linearly interpolates background/sensitivity levels between event midpoints.
        * ``"median"``: Applies global median background/sensitivity levels across the dataset.
    plot_flag : bool, default=False
        If ``True``, renders and exports diagnostic background subtraction and sensitivity calibration plots.
    output_dir : str or None, default=None
        Export directory path for diagnostic plots. Defaults to OpenTof default plot dir if ``None``.
    plot_subdir : str or None, default=None
        Optional sub-folder prefix appended to output plot filenames.

    Returns
    -------
    ppb_bc : numpy.ndarray
        1D array of final background-corrected and sensitivity-calibrated concentrations in ppb.
    sensitivity_axis : numpy.ndarray
        1D array of dynamic or static sensitivity calibration factors applied across time.
    background_corrected : numpy.ndarray
        1D array of signal values after auto-zero background subtraction (prior to sensitivity scaling).
    working_array : numpy.ndarray
        1D array of normalized (or raw) signal values prior to background subtraction.

    Raises
    ------
    ValueError
        If `cycling_zeros` or `cycling_sens_cals` is ``True`` but `cycling_dict` is ``None``,
        or if `normalization=True` but `normalization_reference_values` is ``None``.
    """
    # --- Initialization and Safety Validation ---
    # Enforce requirement for cycling dictionary if dynamic zeros or sensitivity calibrations are enabled
    if (cycling_zeros or cycling_sens_cals) and cycling_dict is None:
        raise ValueError("`cycling_dict` must be provided if `cycling_zeros` or `cycling_sens_cals` is True.")

    # --- Step 1: Primary Reagent Ion Normalization ---
    if normalization:
        # Enforce requirement for reference primary ion values if normalization is requested
        if normalization_reference_values is None:
            raise ValueError("`normalization_reference_values` must be provided if `normalization` is True.")
        
        # Calculate per-spectrum normalization scaling factor: Target_Level / Reference_Ion_Counts
        correction_factor = static_target_value / normalization_reference_values
        
        # Multiply raw peak area array by primary ion normalization factor
        working_array = peak_area_array * correction_factor
    else:
        # Work on an un-normalized copy of the original raw peak area array
        working_array = peak_area_array.copy()

    # Extract cycling flag type array from deployment dictionary if cycling status is available
    if cycling_dict is not None:
        cycling_flags_type = cycling_dict['type']

    # --- Step 2: Background Subtraction (Auto-Zeros) ---
    if cycling_zeros:
        # Locate sample indices where instrument cycling status indicates an auto-zero (flag == 1.0)
        auto_zeros = np.where(cycling_flags_type == 1.)[0]
        
        # Partition contiguous index sequences into discrete event chunks
        zero_chunks = _locate_index_chunks(auto_zeros)
        
        # Assign negative IDs to auto-zero blocks and positive IDs to standard acquisition gaps
        zero_blocks = _assign_blocks_from_chunks(zero_chunks, working_array)
        
        # Compute dynamic/median baseline curve and subtract background from active sample blocks
        background_corrected = _perform_background_subtraction(
            zero_blocks, 
            working_array, 
            mode=mode, 
            plot_flag=plot_flag,
            output_dir=output_dir,
            plot_subdir=plot_subdir
        )
    else:
        # Bypass background subtraction and pass working array through directly
        background_corrected = working_array.copy()

    # --- Step 3: Sensitivity Calibration Modeling ---
    if cycling_sens_cals:
        # Locate sample indices where cycling status indicates a sensitivity calibration challenge (flag == 4.0)
        auto_sens_cals = np.where(cycling_flags_type == 4.)[0]
        
        # Partition contiguous index sequences into discrete event chunks
        sens_cal_chunks = _locate_index_chunks(auto_sens_cals)
        
        # Assign negative IDs to calibration blocks and positive IDs to sample acquisition gaps
        sens_cal_blocks = _assign_blocks_from_chunks(sens_cal_chunks, working_array)
        
        # Calculate block response factors via gas dilution equations and construct time-series sensitivity axis
        sensitivity_axis = _get_cyclic_sensitivity_calibrations(
            sens_cal_blocks,
            background_corrected,
            cal_tank_conc=cal_tank_conc,
            zero_flow=zero_flow,
            cal_flow=cal_flow,
            mode=mode,
            plot_flag=plot_flag,
            output_dir=output_dir,
            plot_subdir=plot_subdir
        )
    else:
        # Bypass dynamic calibration and populate axis with the static user-provided conversion factor
        sensitivity_axis = np.full_like(working_array, static_sens_factor, dtype=float)

    # --- Step 4: Final Concentration Calculation ---
    # Multiply background-subtracted signal by dynamic/static sensitivity calibration factors to get concentration in ppb
    ppb_bc = background_corrected * sensitivity_axis

    return ppb_bc, sensitivity_axis, background_corrected, working_array

# TODO is it possible to auto determine sensisitivity calibration periods somehow?