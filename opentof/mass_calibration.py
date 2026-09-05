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

# mass_calibration.py
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, minimum_filter
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time
import typing
from tqdm.auto import tqdm
import warnings
import dask

from opentof.utils import return_mass, truncate, calculate_rolling_average, ensure_dir, get_default_plot_dir, median_absolute_deviation
from opentof.peak_fitting import fit_unconstrained_peaks

# --- Forward Functions i(m) ---
# Modes 0 and 2 are recommended
def _mode0_fwd(m, p1, p2): return p1 * np.sqrt(m) + p2 # The theoretical function. Generally better for extrapolation to new m/z areas without a calibrant
def _mode1_fwd(m, p1, p2): return p1 / np.sqrt(m) + p2 # Not recommended for use as it can produce nonsensical results
def _mode2_fwd(m, p1, p2, p3): return p1 * m**p3 + p2 # Sometimes perfered over mode 0 as it can be more accurate for m/z areas with nearby calibrants
def _mode3_fwd(m, p1, p2, p3, p4): return p1 * np.sqrt(m) + p2 + p3 * (m - p4)**2 # Usually p3 and p4 are not used?
def _mode4_fwd(m, p1, p2, p3, p4, p5): return p1 * np.sqrt(m) + p2 + p3 * m**2 + p4 * m + p5 # Usually p2 and p5 are not used?
def _mode5_fwd(m, p1, p2, p3): return (-p2 + np.sqrt(p2**2 - 4 * p1 * (p3 - m))) / (2 * p1) # Not recommended as occasionally it does not converge

# --- Inverse Functions m(i) ---
def _mode0_inv(i, p1, p2): return ((i - p2) / p1)**2
def _mode1_inv(i, p1, p2): return (p1 / (i - p2))**2
def _mode2_inv(i, p1, p2, p3): return ((i - p2) / p1)**(1 / p3)
def _mode5_inv(i, p1, p2, p3): return p1 * i**2 + p2 * i + p3

def _numerical_inverse(i_arr, forward_func, params, m_min=1, m_max=2000, steps=10000):
    """Numerical inverse using interpolation."""
    m_grid = np.linspace(m_min, m_max, steps)
    i_grid = forward_func(m_grid, *params)
    # Ensure monotonic increasing for PCHIP
    sort_idx = np.argsort(i_grid)
    interp = PchipInterpolator(i_grid[sort_idx], m_grid[sort_idx])
    return interp(i_arr)

def _mode3_inv(i, p1, p2, p3, p4): return _numerical_inverse(i, _mode3_fwd, (p1, p2, p3, p4))
def _mode4_inv(i, p1, p2, p3, p4, p5): return _numerical_inverse(i, _mode4_fwd, (p1, p2, p3, p4, p5))

# --- Registry ---
MASS_CAL_MODES = {
    0: {'fwd': _mode0_fwd, 'inv': _mode0_inv, 'p0': [1769, -3644], 'eq': 'i(m) = p1 * sqrt(m) + p2'},
    1: {'fwd': _mode1_fwd, 'inv': _mode1_inv, 'p0': [1769, -3644], 'eq': 'i(m) = p1 / sqrt(m) + p2'},
    2: {'fwd': _mode2_fwd, 'inv': _mode2_inv, 'p0': [1769, -3644, 0.5], 'eq': 'i(m) = p1 * m^p3 + p2'},
    3: {'fwd': _mode3_fwd, 'inv': _mode3_inv, 'p0': [1769, -3644, 0.0, 100.0], 'eq': 'i(m) = p1 * sqrt(m) + p2 + p3 * (m - p4)^2'},
    4: {'fwd': _mode4_fwd, 'inv': _mode4_inv, 'p0': [1769, -3644, 0.0, 0.0, 0.0], 'eq': 'i(m) = p1 * sqrt(m) + p2 + p3 * m^2 + p4 * m + p5'},
    5: {'fwd': _mode5_fwd, 'inv': _mode5_inv, 'p0': [0.001, 0.1, 10.0], 'eq': 'm(i) = p1 * i^2 + p2 * i + p3'}
}

def apply_mass_calibration(i, params, mode=2):
    """Universal function to get the mass axis m(i)."""
    return MASS_CAL_MODES[mode]['inv'](i, *params)

def fit_mass_calibration(m, params, mode=2):
    """Universal function to get sample index i(m)."""
    return MASS_CAL_MODES[mode]['fwd'](m, *params)

def load_calibrants(filepath: str) -> dict:
    """
    Load mass calibration data from a .cal file.

    Parameters
    ----------
    filepath : str
        Path to the .cal file.

    Returns
    -------
    dict
        Mapping of peak name to exact m/z value, e.g. ``{"H3O+": 19.017841, ...}``.
        Order matches the indexed order in the file (n1/x1, n2/x2, …).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    names = []
    values = []
    peak_index_to_use = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('n_n='):
            count = int(line.split('=', 1)[1])
            for j in range(1, count + 1):
                names.append(lines[i + j].strip().split('=', 1)[1])
            i += count + 1
        elif line.startswith('n_x='):
            count = int(line.split('=', 1)[1])
            for j in range(1, count + 1):
                values.append(float(lines[i + j].strip().split('=', 1)[1]))
            i += count + 1
        elif line.startswith('UserNote'):
            to_use = int(line.split(';')[1].split(':')[1])
            peak_index_to_use = [i for i, bit in enumerate(bin(to_use)[:1:-1]) if bit == '1']
            i += 1
        else:
            i += 1

    if peak_index_to_use is not None:
        names = [names[i] for i in peak_index_to_use]
        values = [values[i] for i in peak_index_to_use]

    return dict(zip(names, values))

def load_calibration_mode(filepath: str) -> int:
    """
    Load mass calibration mode from a .cal file.

    Parameters
    ----------
    filepath : str
        Path to the .cal file.

    Returns
    -------
    int
        Calibration mode. Defaults to 2 if not found or unparseable.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('CalFunc='):
                    parts = line.split("MassCalibMode:")
                    if len(parts) > 1:
                        return int(parts[1].split(";")[0])
    except Exception as e:
        print(f"Warning: Could not parse calibration mode from {filepath} ({e}). Defaulting to Mode 2.")
    
    return 2


# Helper function executed in parallel on workers
@dask.delayed
def _multiply_chunk_sparse(W_sub, chunk_data):
    """
    Multiply a SciPy sparse sub-matrix by a dense NumPy data block in parallel.

    Parameters
    ----------
    W_sub : scipy.sparse.csr_matrix
        Sparse weight sub-matrix slice of shape ``(num_groups, chunk_len)``.
    chunk_data : numpy.ndarray
        Dense array block of spectral data of shape ``(chunk_len, num_samples)``.

    Returns
    -------
    numpy.ndarray
        Dense matrix of shape ``(num_groups, num_samples)`` representing the 
        partial matrix product for the given chunk.
    """
    # Perform sparse matrix multiplication between weight sub-block and data chunk
    return W_sub.dot(chunk_data)

def generate_averaged_dataset(tofdata, timestamps, 
                              standard_acq_data, averaging_interval=300,
                              chunk_size=1000, sample_perc=1.0):
    """
    Generate an averaged time-series dataset over specified time windows.

    Masks non-standard acquisition writebufs and partitions valid spectra into 
    discrete time intervals. Uses a sparse weight matrix multiplication pattern 
    parallelized across memory blocks via Dask delayed tasks.

    Parameters
    ----------
    tofdata : dask.array.Array or numpy.ndarray
        2D array of time-of-flight spectral data of shape ``(num_writebuf, num_samples)``.
    timestamps : numpy.ndarray
        1D array of datetime or timedelta timestamps corresponding to each writebuf.
    standard_acq_data : numpy.ndarray or None
        1D boolean mask where ``True`` indicates valid standard acquisition data.
        If ``None``, all writebufs are treated as standard acquisition data.
    averaging_interval : int or float, default=300
        Target time window duration in seconds over which to average spectra.
    chunk_size : int, default=1000
        Number of writebufs per chunk block for parallel computation.
    sample_perc : float, default=1.0
        Fraction of valid writebufs to randomly sample per averaging interval (0.0, 1.0].

    Returns
    -------
    averaged_results : numpy.ndarray
        Dense 2D NumPy array of time-averaged spectra of shape ``(num_groups, num_samples)``.
    midpoint_timestamps : numpy.ndarray
        1D array containing the calculated midpoint timestamp for each group interval.
    final_indices : numpy.ndarray
        1D integer array mapping each original writebuf index to its corresponding 
        averaged group index (with non-standard writebuf gaps interpolated).
    """
    # Convert scalar interval (in seconds) to a NumPy timedelta64 object for duration comparisons
    averaging_interval = np.timedelta64(averaging_interval, 's')
    num_writebuf = tofdata.shape[0]
    
    # Default to treating all writebufs as standard acquisition if no boolean mask is provided
    if standard_acq_data is None:
        standard_acq_data = np.ones(num_writebuf, dtype=bool)

    # ---------------------------------------------------------
    # STEP 1: Determine Intervals & Sampled Indices
    # ---------------------------------------------------------
    # Initialize index mapping array filled with NaNs for non-standard writebufs
    interval_indices = np.full(num_writebuf, np.nan)
    group_timestamps = []
    
    current_window_start = None
    current_group_ts, current_group_idx = [], []
    mi = 0  # Group index counter
    all_selected_indices = []
    group_member_counts = []

    # Inner helper function to compute group statistics and perform subsampling
    def process_group(idx_list, ts_list):
        # Compute the group midpoint timestamp from start and end points
        avg_ts = ts_list[0] + (ts_list[-1] - ts_list[0]) / 2
        group_timestamps.append(avg_ts)
        
        # Determine number of samples to select based on requested sample percentage
        n_samples = max(1, int(len(idx_list) * sample_perc))
        sampled_idx = np.random.choice(idx_list, size=n_samples, replace=False)
        all_selected_indices.append(sampled_idx)
        group_member_counts.append(len(sampled_idx))

    if sample_perc < 1.0:
        print(f"Sampling {sample_perc*100:.1f}% of data within each MC interval...")

    # Iterate through writebufs and group valid acquisition points into discrete time windows
    for i in range(num_writebuf):
        # Skip writebufs flagged as non-standard acquisition
        if not standard_acq_data[i]:
            continue
            
        ts = timestamps[i]
        # Establish start boundary for the initial window
        if current_window_start is None:
            current_window_start = ts
            
        # Accumulate writebuf into current interval if within averaging window
        if (ts - current_window_start) <= averaging_interval:
            interval_indices[i] = mi
            current_group_ts.append(ts)
            current_group_idx.append(i)
        else:
            # Close out active group and advance window index counter
            process_group(current_group_idx, current_group_ts)
            mi += 1
            current_window_start = ts
            interval_indices[i] = mi
            current_group_ts, current_group_idx = [ts], [i]

    # Process final remaining writebuf group if non-empty
    if current_group_idx:
        process_group(current_group_idx, current_group_ts)

    num_groups = mi + 1

    # ---------------------------------------------------------
    # STEP 2: Sparse Weight Matrix Construction
    # ---------------------------------------------------------
    # Build CSR sparse aggregation matrix W where each row computes an interval arithmetic mean
    row_indices = []
    col_indices = []
    weights = []

    for g_id, (indices, count) in enumerate(zip(all_selected_indices, group_member_counts)):
        row_indices.extend([g_id] * count)     # Row = Group ID
        col_indices.extend(indices)            # Column = Original writebuf index
        weights.extend([1.0 / count] * count)  # Weight = 1 / N_samples for averaging

    # Assemble Compressed Sparse Row (CSR) matrix of shape (num_groups, num_writebuf)
    W_sparse = sp.csr_matrix(
        (weights, (row_indices, col_indices)), 
        shape=(num_groups, num_writebuf), 
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # STEP 3: Chunk-Wise Parallel Reduction via Dask Delayed
    # ---------------------------------------------------------
    tofdata_2d = tofdata

    # Check if dataset is an out-of-core Dask array
    if hasattr(tofdata_2d, 'to_delayed'):
        print("Computing averaged dataset...")
        # Flatten multi-dimensional Dask chunks into a 1D list of delayed block objects
        blocks = tofdata_2d.to_delayed().flatten()
        delayed_results = []
        start_idx = 0

        # Loop through chunks and slice corresponding sparse weight matrix blocks
        for block, chunk_len in zip(blocks, tofdata_2d.chunks[0]):
            end_idx = start_idx + chunk_len
            # Slice weight columns matching writebuf range in current Dask block
            W_sub = W_sparse[:, start_idx:end_idx]
            
            # Queue delayed sparse matrix multiplication task for worker execution
            res = _multiply_chunk_sparse(W_sub, block)
            delayed_results.append(res)
            
            start_idx = end_idx

        # Sum and evaluate partial matrix multiplication results in a single parallel pass
        averaged_results = dask.compute(sum(delayed_results))[0]
    else:
        # Fallback for in-memory NumPy arrays: direct sparse dot product
        averaged_results = W_sparse.dot(tofdata_2d)

    # ---------------------------------------------------------
    # STEP 4: Compute Averages and Format
    # ---------------------------------------------------------
    midpoint_timestamps = np.array(group_timestamps)

    # Ensure midpoint timestamps and averaged matrix are sorted in chronological order
    sort_idx = np.argsort(midpoint_timestamps)
    averaged_results = averaged_results[sort_idx]
    midpoint_timestamps = midpoint_timestamps[sort_idx]

    # Re-map writebuf group indices to reflect chronological sorting and interpolate gaps
    not_nan = ~np.isnan(interval_indices)
    if np.any(not_nan):
        # Map original group indices to new chronologically sorted position
        inv_map = np.zeros(len(sort_idx), dtype=int)
        inv_map[sort_idx] = np.arange(len(sort_idx))
        interval_indices[not_nan] = inv_map[interval_indices[not_nan].astype(int)]
        
        # Linearly interpolate indices across non-standard acquisition gaps and round
        x = np.arange(len(interval_indices))
        interpolated = np.interp(x, x[not_nan], interval_indices[not_nan])
        final_indices = np.round(interpolated).astype(int)
    else:
        final_indices = interval_indices

    return averaged_results, midpoint_timestamps, final_indices


def run_mass_calibration(tofdata, 
                         timestamps, 
                         standard_acq_data, 
                         massaxis_first_guess, 
                         calibrants,
                         tof_axis,
                         gad_sample_perc=1.0,
                         precomputed_gad=None,
                         search_range=None, 
                         averaging_interval=300,
                         mass_cal_mode=2,
                         track_sip_drift=True,
                         auto_reject=False,            
                         auto_reject_threshold=10.0, 
                         auto_reject_weights={'ppm_90pct': 0.4, 'ppm_std': 0.2, 'sip_drift': 0.4},
                         plot_every_iteration=False,
                         min_calibrants=None,
                         interval_auto_reject=False,
                         interval_auto_reject_threshold=50.0,
                         interval_max_rejects=2,
                         supersaturated_calibrants=False,
                         mz_protection_window=10.0,
                         mz_anchor=False,
                         mz_anchor_spacing=50.0,
                         peak_type='gaussian',
                         custom_shape=None,
                         output_dir=None,
                         plot_flag=True,
                         show_plot_flag=True,
                         show_tof_drift_plot=False,            
                         mc_summary_plot_name='mc_summary.png',
                         tof_drift_plot_name='tof_drift.png',
                         plot_dpi=150,
                         chunk_size=1000):
    """
    Perform time-dependent mass calibration on Time-of-Flight (TOF) mass spectrometry data.

    Calibrates acquired TOF spectra data using an averaged dataset, fitting known calibrant 
    peak targets in Sample Index Position (SIP) space and resolving non-linear calibration 
    coefficients. Supports multi-metric global auto-rejection, interval-level outlier removal, 
    supersaturated calibrant filtering, m/z anchor gap protection, and baseline drift tracking.

    Parameters
    ----------
    tofdata : dask.array.Array or numpy.ndarray
        2D matrix of time-of-flight spectral data of shape ``(num_writebuf, num_samples)``.
    timestamps : numpy.ndarray
        1D array of acquisition timestamps (datetime or timedelta) for each writebuf.
    standard_acq_data : numpy.ndarray
        1D boolean array indicating standard acquisition writebufs (``True`` = valid sample data).
    massaxis_first_guess : numpy.ndarray
        1D array representing the initial estimated mass-to-charge (m/z) axis.
    calibrants : dict
        Mapping of compound names to theoretical m/z values (e.g., ``{"H3O+": 19.01784}``).
    tof_axis : numpy.ndarray
        1D array representing the physical time-of-flight axis in nanoseconds.
    gad_sample_perc : float, default=1.0
        Fraction of valid spectra to randomly sample per averaging interval (0.0 to 1.0].
    precomputed_gad : tuple or None, default=None
        Pre-calculated Global Averaged Dataset tuple ``(averaged_data, midpoint_timestamps, interval_indices)``
        to bypass compute step.
    search_range : int or None, default=None
        Width (in sample bins) of the fitting window around each calibrant target.
        If ``None``, defaults to 1/1000th of the total spectrum bin length.
    averaging_interval : int or float, default=300
        Time window duration in seconds over which spectra are averaged for calibration.
    mass_cal_mode : int, default=2
        Mathematical calibration equation model index from ``MASS_CAL_MODES`` 
        (e.g., Mode 0: i(m) = p_1 sqrt{m} + p_2, Mode 2: i(m) = p_1 m^{p_3} + p_2).
    track_sip_drift : bool, str, or list of str, default=True
        If ``True``, tracks sample index position drift for all active calibrants. If a string or list,
        tracks drift specifically for the requested compound keys.
    auto_reject : bool, default=False
        If ``True``, iteratively calculates compound quality scores across time and removes 
        the worst-performing calibrants exceeding performance thresholds.
    auto_reject_threshold : float, default=10.0
        Target 90th percentile PPM error limit for auto-rejection.
    auto_reject_weights : dict, default={'ppm_90pct': 0.4, 'ppm_std': 0.2, 'sip_drift': 0.4}
        Weight dictionary used to scale min-max normalized metrics for compound auto-rejection scoring.
    plot_every_iteration : bool, default=False
        If ``True``, renders intermediate deviation plots during every auto-rejection iteration loop.
    min_calibrants : int or None, default=None
        Minimum number of active calibrants required to continue calibration optimization.
        If ``None``, defaults to ``num_params + 1``.
    interval_auto_reject : bool, default=False
        If ``True``, enables localized auto-rejection of transient outlier peaks within individual time intervals.
    interval_auto_reject_threshold : float, default=50.0
        PPM deviation threshold above which localized interval peaks are rejected.
    interval_max_rejects : int, default=2
        Maximum number of calibrants that can be locally rejected within a single time interval.
    supersaturated_calibrants : bool, default=False
        If ``True``, evaluates dense/overlapping calibrant lists within m/z windows, selecting only 
        the most stable, non-latching calibrant per window.
    mz_protection_window : float, default=10.0
        Window width in m/z units used to partition calibrants during supersaturated filtering.
    mz_anchor : bool, default=False
        If ``True``, detects large uncalibrated m/z gaps and restores previously auto-rejected 
        calibrants within those gaps to maintain structural stability.
    mz_anchor_spacing : float, default=50.0
        Maximum allowable gap size in m/z units between adjacent calibrants before trigger.
    peak_type : str, default='gaussian'
        Peak shape model to fit ('gaussian', 'lorentzian', 'pseudo_voigt', or 'custom').
    custom_shape : callable or None, default=None
        Callable custom peak shape function required if ``peak_type='custom'``.
    output_dir : str or None, default=None
        Directory path to save diagnostic plots and summaries. Defaults to OpenTof default plot dir.
    plot_flag : bool, default=True
        If ``True``, generates and exports summary diagnostic figures.
    show_plot_flag : bool, default=True
        If ``True``, displays generated diagnostic plots interactively.
    show_tof_drift_plot : bool, default=False
        If ``True``, generates a multi-panel plot showing absolute TOF drift (ns) over time.
    mc_summary_plot_name : str, default='mc_summary.png'
        Filename for the saved calibration summary plot.
    tof_drift_plot_name : str, default='tof_drift.png'
        Filename for the saved TOF drift summary plot.
    plot_dpi : int, default=150
        DPI resolution for exported diagnostic plots.
    chunk_size : int, default=1000
        Number of writebufs per chunk block for parallel computation.

    Returns
    -------
    calibration_results : dict
        A mass calibration results dictionary containing:
            * ``"calibrants"`` : pandas.DataFrame of final active calibrants and theoretical m/z values.
            * ``"batch_params"`` : 2D numpy.ndarray of fitted calibration parameters per interval.
            * ``"batch_massaxes"`` : 2D numpy.ndarray of reconstructed mass axes per interval.
            * ``"batch_deviations"`` : 2D numpy.ndarray of PPM deviations per calibrant per interval.
            * ``"batch_min_err"`` : 2D numpy.ndarray of absolute mass errors per calibrant per interval.
            * ``"batch_measured_sip"`` : 2D numpy.ndarray of fitted Sample Index Positions per interval.
            * ``"batch_fwhm_values"`` : 2D numpy.ndarray of fitted peak FWHMs in sample index space.
            * ``"batch_fwhm_squared_values"`` : 2D numpy.ndarray of squared FWHM values per interval.
            * ``"batch_resolution_values"`` : 2D numpy.ndarray of calculated mass resolutions (m / dm).
            * ``"mass_cal_mode"`` : int representing the calibration mode equation used.
            * ``"midpoint_timestamps"`` : 1D numpy.ndarray of time interval midpoints.
            * ``"interval_indices"`` : 1D numpy.ndarray mapping original writebufs to interval indices.
            * ``"mean_absolute_min_mass_error"`` : 1D numpy.ndarray of mean absolute mass errors per calibrant.
            * ``"median_absolute_min_mass_error"`` : 1D numpy.ndarray of median absolute mass errors.
            * ``"std_absolute_min_mass_error"`` : 1D numpy.ndarray of error standard deviations per calibrant.
            * ``"max_absolute_min_mass_error"`` : 1D numpy.ndarray of maximum absolute mass errors per calibrant.
            * ``"mean_resolution"`` : float representing the overall dataset mean mass resolution.
            * ``"per_calibrant_mean_resolution"`` : 1D numpy.ndarray of mean resolution per calibrant.
            * ``"calibrant_individual_sip_drifts"`` : dict of sample index position drift arrays per compound.
            * ``"average_sip_drift"`` : 1D numpy.ndarray of mean sample index position drift over time.
            * ``"median_sip_drift"`` : 1D numpy.ndarray of median sample index position drift over time.
            * ``"tof_axis"`` : 1D numpy.ndarray of the physical TOF axis in nanoseconds.
    """
    # Start timer
    s_start = time.time()

    # Default output directory setup
    if output_dir is None:
        output_dir = get_default_plot_dir()
        
    ensure_dir(output_dir)

    # Print the active mass calibration function layout equation
    equation_str = MASS_CAL_MODES[mass_cal_mode]['eq']
    print(f"Using mass calibration mode {mass_cal_mode}: {equation_str}")
    print("-" * 50)

    # --- Pre-process sample index position (SIP) drift tracking selection ---
    if track_sip_drift is True:
        # Pull all keys directly from the calibrants tracking map if True
        track_sip_drift = list(calibrants.keys())
    elif isinstance(track_sip_drift, str):
        # Wrap single string argument entries into an iterable list
        track_sip_drift = [track_sip_drift]

    # Pre-inject any requested drift compounds missing from the core dictionary
    if track_sip_drift:
        for comp in track_sip_drift:
            if comp not in calibrants:
                calibrants[comp] = return_mass(comp)

    # ---------------------------------------------------------
    # STEP 1: Generate Averaged Dataset (GAD)
    # ---------------------------------------------------------
    # Check standard acquisition mask coverage; fallback to all spectra if < 60% standard data
    is_60_percent_true = np.mean(standard_acq_data) >= 0.60
    if not is_60_percent_true:
        print("WARNING - Either cycling was not active during acquisition or less than 60% of data is standard acquisition data.")
        print("Using all available writebufs to generate averaged dataset.")
        standard_acq_data = np.full_like(standard_acq_data, True, dtype=bool)

    # Compute or reuse pre-calculated Global Averaged Dataset (GAD)
    if precomputed_gad is None:
        print("\nGenerating Averaged Dataset...")
        averaged_data, midpoint_timestamps, interval_indices = generate_averaged_dataset(
            tofdata, timestamps, standard_acq_data=standard_acq_data,
            averaging_interval=averaging_interval, chunk_size=chunk_size, sample_perc=gad_sample_perc,
        )
        print("Complete.\n")
    else:
        print("Using Stored Averaged Dataset.")
        averaged_data, midpoint_timestamps, interval_indices = precomputed_gad

    print(f"Number of intervals with {averaging_interval} sec averaging: {averaged_data.shape[0]}")
    print(f"Calibrant list: {list(calibrants.keys())}")

    # Generate sample index axis array: [0, 1, 2, ..., num_samples - 1]
    sample_index_axis = np.arange(tofdata.shape[-1])

    # Initialize reference anchors outside retry loops so baseline origins remain fixed across iterations
    if track_sip_drift:
        reference_lock_sips = {}

    # ---------------------------------------------------------
    # SUPERSATURATED CALIBRANTS FILTER
    # ---------------------------------------------------------
    if supersaturated_calibrants:
        print(f"Supersaturated mode active. Evaluating {len(calibrants)} calibrants over all {len(averaged_data)} intervals...")

        # Group input calibrants into discrete m/z protection windows
        windowed_cals = {}
        for name, mz in calibrants.items():
            window_idx = int(mz // mz_protection_window)
            if window_idx not in windowed_cals:
                windowed_cals[window_idx] = {}
            windowed_cals[window_idx][name] = mz

        # Dictionary to store fitted SIP arrays for each candidate calibrant
        cal_sips = {name: [] for name in calibrants.keys()}
        search_range_pre = int(tofdata.shape[-1] / 1000) if search_range is None else search_range

        # Perform preliminary peak fits for ALL candidates across ALL averaged time intervals
        for interval in tqdm(range(len(averaged_data)), desc="Supersaturated peak fits"):
            for name, mz in calibrants.items():
                # Estimate sample index position using initial first-guess mass axis
                idx_guess = np.argmin(np.abs(massaxis_first_guess[:] - mz))
                idx_min = int(max(idx_guess - search_range_pre/2, 0))
                idx_max = int(min(idx_guess + search_range_pre/2, tofdata.shape[-1]))
                
                intensityaxis = averaged_data[interval, idx_min:idx_max]
                si_axis = np.arange(idx_min, idx_max)
                
                if len(intensityaxis) == 0: 
                    continue
                intensity_max = np.max(intensityaxis)
                if intensity_max == 0: 
                    continue
                norm_intensity = intensityaxis / intensity_max
                
                try:
                    xc_guess = si_axis[np.argmax(norm_intensity)]
                    fwhm_guess = max(np.sum(norm_intensity > 0.5), 1.0)
                    popt = fit_unconstrained_peaks(
                        x_axis=si_axis, signal=norm_intensity,
                        centers_guess=[xc_guess], fwhms_guess=[fwhm_guess], amplitudes_guess=[1.0],
                        peak_type=peak_type, custom_shape=custom_shape, center_wiggle=2.0, max_iter=1000
                    )
                    cal_sips[name].append(popt[1])
                except Exception:
                    pass

        # Filter candidates to select the single best calibrant per window via drift and deduplication
        filtered_calibrants = {}
        min_required_fits = len(averaged_data) * 0.5  # Require successful fit in at least 50% of intervals
        
        for win_idx, cals_in_win in windowed_cals.items():
            valid_cals = {}
            for name, mz in cals_in_win.items():
                sips = cal_sips[name]
                if len(sips) >= min_required_fits:
                    mean_sip = np.mean(sips)
                    drift = max(sips) - min(sips)  # Measure timeseries positional drift
                    estimated_mz = np.interp(mean_sip, sample_index_axis, massaxis_first_guess)
                    mass_err = abs(mz - estimated_mz)
                    valid_cals[name] = {"mz": mz, "mean_sip": mean_sip, "drift": drift, "mass_err": mass_err, "fits": len(sips)}
            
            best_cal_name = None
            if valid_cals:
                # --- PEAK LATCHING DEDUPLICATION ---
                # Sort candidates by mass error to prioritize targets closest to theoretical values
                sorted_by_err = sorted(valid_cals.items(), key=lambda item: item[1]["mass_err"])
                unique_peaks = {}
                sip_tolerance = 1.0  # Index threshold to determine if two fits latched onto the same physical peak
                
                for name, data in sorted_by_err:
                    is_duplicate = False
                    for unique_name, unique_data in unique_peaks.items():
                        if abs(data["mean_sip"] - unique_data["mean_sip"]) < sip_tolerance:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_peaks[name] = data

                # Select the candidate with the lowest timeseries drift among unique physical peaks
                best_cal_name = min(unique_peaks.keys(), key=lambda k: unique_peaks[k]["drift"])
            else:
                # --- FALLBACK SELECTION ---
                # Salvage by picking the candidate with the highest fit count (break ties via mass error)
                fallback_cals = []
                for name, mz in cals_in_win.items():
                    sips = cal_sips[name]
                    if len(sips) > 0:
                        mean_sip = np.mean(sips)
                        estimated_mz = np.interp(mean_sip, sample_index_axis, massaxis_first_guess)
                        mass_err = abs(mz - estimated_mz)
                        fallback_cals.append((name, len(sips), mass_err))
                if fallback_cals:
                    fallback_cals.sort(key=lambda x: (-x[1], x[2]))
                    best_cal_name = fallback_cals[0][0]

            if best_cal_name:
                filtered_calibrants[best_cal_name] = cals_in_win[best_cal_name]
                
        calibrants = filtered_calibrants
        print(f"Supersaturated filtering complete. Retained {len(calibrants)} calibrants.")
        print("-" * 50)

    # Determine required parameter count based on active calibration equation mode
    num_params = len(MASS_CAL_MODES[mass_cal_mode]['p0'])

    # Enforce minimum calibrants safety floor to avoid overfitting
    if min_calibrants is None:
        calculated_min_calibrants = num_params + 1
    else:
        if min_calibrants < num_params + 1:
            print(f"Minimum number of calibrants entered: {min_calibrants} is too low "
                  f"for current mass calibration mode. Optimization may overfit. "
                  f"Either set min_calibrants to None or {num_params + 1}")
        calculated_min_calibrants = min_calibrants

    # Initialize auto-rejection state trackers
    active_calibrants = calibrants.copy()
    auto_rejected = []
    rejection_history = {}  # Store worst error at time of rejection for recovery
    protected_peaks = set() # Peaks restored by m/z anchor gap protection cannot be re-rejected

    # ---------------------------------------------------------
    # STEPS 2-4: Global Calibration & Auto-Rejection Loop
    # ---------------------------------------------------------
    while True:
        # Match current active calibrant m/z values to initial sample index targets
        calibrant_info = {}
        for name, m_q in active_calibrants.items():
            closest_index = np.argmin(np.abs(massaxis_first_guess[:] - np.round(m_q, 6)))
            calibrant_info[name] = {"m/Q": m_q, "index": closest_index}

        # Initialize temporal batch arrays for accumulating interval results
        batch_measured_sip = []
        batch_calibrant_mz_values = []
        batch_fwhm_values = []
        batch_massaxes = []
        batch_deviations = []
        batch_min_err = []
        batch_params = []
        batch_fwhm_squared_values = []
        batch_resolution_values = []

        # Reset drift tracking containers inside retry loop boundaries
        if track_sip_drift:
            batch_individual_drifts = {comp: [] for comp in track_sip_drift}

        # Set search range default if unassigned
        if search_range is None:
            total_si_length = tofdata.shape[-1]
            search_range = int(total_si_length / 1000)
            print(f"Using search range of {search_range}")

        # --- Interval Loop (where the calibration occurs) ---
        # Iterate over each averaged time interval block
        for interval in tqdm(range(len(averaged_data)), desc="Running mass calibration"):
            interval_sips, interval_fwhms = {}, {}

            # Fit each active calibrant peak within its local index window
            for compound, cal_info in calibrant_info.items():
                idx = cal_info["index"]
                idx_min = int(idx - search_range/2)
                idx_max = int(idx + search_range/2)
                intensityaxis = averaged_data[interval, idx_min:idx_max]
                si_axis = np.arange(idx_min, idx_max)

                # Normalize local peak segment intensity
                intensity_max = np.max(intensityaxis)
                if intensity_max == 0: 
                    continue
                norm_intensity = intensityaxis / intensity_max

                try:
                    # Formulate initial guesses for unconstrained peak fit
                    xc_guess = si_axis[np.argmax(norm_intensity)]
                    fwhm_guess = max(np.sum(norm_intensity > 0.5), 1.0)

                    # Fit peak shape in sample index space
                    popt = fit_unconstrained_peaks(
                        x_axis=si_axis, signal=norm_intensity,
                        centers_guess=[xc_guess], fwhms_guess=[fwhm_guess], amplitudes_guess=[1.0],
                        peak_type=peak_type, custom_shape=custom_shape, center_wiggle=2.0, max_iter=1000
                    )

                    interval_sips[compound] = popt[1]   # Fitted center (SIP)
                    interval_fwhms[compound] = popt[2]  # Fitted FWHM (sample index units)
                except Exception:
                    continue

            # --- Local Interval Auto-Improvement Loop ---
            interval_active_cals = list(interval_sips.keys())
            interval_rejects_count = 0
            
            while True:
                # Extract fitted SIPs and theoretical m/z values for active interval calibrants
                measured_sip = np.array([interval_sips[c] for c in interval_active_cals])
                calibrant_mz_values = np.array([calibrant_info[c]["m/Q"] for c in interval_active_cals])

                if len(measured_sip) < calculated_min_calibrants:
                    break

                # Fit mass calibration function coefficients for current time interval
                params = calibrate_mass(measured_sip, calibrant_mz_values, mode=mass_cal_mode)

                # Reconstruct full mass axis for current interval
                sample_indices = np.arange(averaged_data.shape[1])
                mass_axis = apply_mass_calibration(sample_indices, params, mode=mass_cal_mode)

                # Build PCHIP interpolator for high-precision SIP-to-mass translation
                pchip_interp = PchipInterpolator(sample_index_axis, mass_axis)

                # Exit local loop if interval auto-rejection is disabled
                if not interval_auto_reject:
                    break

                # Calculate local PPM deviations for outlier detection
                current_deviations = {}
                for compound in interval_active_cals:
                    exact_mass = calibrant_info[compound]["m/Q"]
                    current_deviations[compound] = np.abs(((exact_mass - pchip_interp(interval_sips[compound])) / exact_mass) * 1e6)

                # Identify worst calibrant in current time interval
                worst_cal_interval = max(current_deviations, key=current_deviations.get)

                # Reject local outlier if deviation exceeds threshold and floor constraints
                if (current_deviations[worst_cal_interval] > interval_auto_reject_threshold and 
                    len(interval_active_cals) > calculated_min_calibrants and 
                    interval_rejects_count < interval_max_rejects):
                    interval_active_cals.remove(worst_cal_interval)
                    interval_rejects_count += 1
                else:
                    break
            
            try:
                batch_params.append(params)
            except UnboundLocalError:
                raise UnboundLocalError("Number of calibrants must be 2 + number of mass calibration parameters")

            # Track sample index position (SIP) drift over time
            if track_sip_drift:
                for comp in track_sip_drift:
                    if comp in interval_sips:
                        current_sip = interval_sips[comp]
                        if comp not in reference_lock_sips:
                            reference_lock_sips[comp] = current_sip
                        # Store drift relative to initial anchor position
                        batch_individual_drifts[comp].append(current_sip - reference_lock_sips[comp])
                    else:
                        batch_individual_drifts[comp].append(np.nan)
            
            # Store reconstructed mass axis
            batch_massaxes.append(mass_axis)

            # Calculate and align error metrics (pad missing peaks with NaNs)
            deviations, min_err, aligned_sips = [], [], []
            aligned_fwhms, aligned_fwhm_sq, resolution_values = [], [], []
            aligned_mz = [] 

            for compound in active_calibrants.keys():
                if compound in interval_active_cals:
                    exact_mass = calibrant_info[compound]["m/Q"]
                    sip_center = interval_sips[compound]
                    fwhm_si = interval_fwhms[compound]

                    # Calculate mass error and PPM deviation
                    fit_mass_interp = pchip_interp(sip_center)
                    deviations.append(((exact_mass - fit_mass_interp) / exact_mass) * 1e6)
                    min_err.append(exact_mass - fit_mass_interp)

                    aligned_sips.append(sip_center)
                    aligned_fwhms.append(fwhm_si)
                    aligned_fwhm_sq.append(fwhm_si ** 2)
                    aligned_mz.append(exact_mass)

                    # Calculate mass resolution (m / Delta_m)
                    fwhm_half = fwhm_si / 2
                    indices = np.arange(len(mass_axis))
                    mz_min_precise = np.interp(sip_center - fwhm_half, indices, mass_axis)
                    mz_max_precise = np.interp(sip_center + fwhm_half, indices, mass_axis)
                    delta_m = mz_max_precise - mz_min_precise
                    resolution_values.append(exact_mass / delta_m if delta_m > 0 else np.nan)
                else:
                    # Pad missing calibrants with NaNs
                    deviations.append(np.nan); min_err.append(np.nan); aligned_sips.append(np.nan)
                    aligned_fwhms.append(np.nan); aligned_fwhm_sq.append(np.nan); resolution_values.append(np.nan)
                    aligned_mz.append(np.nan)

            # Accumulate interval arrays into batch master containers
            batch_measured_sip.append(aligned_sips)
            batch_fwhm_values.append(aligned_fwhms)
            batch_fwhm_squared_values.append(aligned_fwhm_sq)
            batch_deviations.append(deviations)
            batch_min_err.append(min_err)
            batch_resolution_values.append(resolution_values)
            batch_calibrant_mz_values.append(aligned_mz)

        # =========================================================================
        # MULTI-METRIC AUTO-REJECT SCORING
        # =========================================================================
        if not auto_reject:
            break

        optimization_history_errors = []
        optimization_history_labels = []

        dev_arr = np.array(batch_deviations, dtype=float)
        sip_arr = np.array(batch_measured_sip, dtype=float)
        total_intervals = len(averaged_data)

        # Compute raw physical performance metrics across time for each active calibrant
        cal_metrics = {}
        for i, cal in enumerate(active_calibrants.keys()):
            if i < dev_arr.shape[1]:
                v_ppm = dev_arr[:, i][~np.isnan(dev_arr[:, i])]
                v_sip = sip_arr[:, i][~np.isnan(sip_arr[:, i])]

                ppm_90 = np.percentile(np.abs(v_ppm), 90) if len(v_ppm) > 0 else np.nan
                ppm_max = np.max(np.abs(v_ppm)) if len(v_ppm) > 0 else np.nan
                ppm_std = np.std(v_ppm) if len(v_ppm) > 0 else np.nan
                sip_drift = (np.max(v_sip) - np.min(v_sip)) if len(v_sip) > 1 else np.nan
                fit_fraction = len(v_ppm) / total_intervals
            else:
                ppm_90, ppm_max, ppm_std, sip_drift, fit_fraction = np.nan, np.nan, np.nan, np.nan, 0.0
                
            cal_metrics[cal] = {
                'ppm_90pct': ppm_90,
                'ppm_max': ppm_max,
                'ppm_std': ppm_std,
                'sip_drift': sip_drift,
                'fit_fraction': fit_fraction
            }
        
        # Min-Max Normalization across active calibrants (Scales metrics from 0.0 to 1.0)
        scaled_metrics = {m: {} for m in auto_reject_weights.keys()}
        for m in auto_reject_weights.keys():
            active_vals = [cal_metrics[c][m] for c in active_calibrants.keys() if not np.isnan(cal_metrics[c][m])]
            if len(active_vals) > 0:
                v_min, v_max = min(active_vals), max(active_vals)
                v_range = v_max - v_min if v_max != v_min else 1.0
                for c in active_calibrants.keys():
                    val = cal_metrics[c][m]
                    if not np.isnan(val):
                        # Invert scale for fit_fraction (lower fraction = higher penalty)
                        scaled_metrics[m][c] = (v_max - val) / v_range if m == 'fit_fraction' else (val - v_min) / v_range
                    else:
                        scaled_metrics[m][c] = 1.0  # Apply maximum penalty for failure
            else:
                for c in active_calibrants.keys(): 
                    scaled_metrics[m][c] = 1.0

        # Calculate weighted compound penalty score for each candidate
        compound_scores = {}
        total_weight = sum(auto_reject_weights.values())

        for c in active_calibrants.keys():
            if np.isnan(cal_metrics[c]['ppm_90pct']):
                compound_scores[c] = float('inf')  # Priority removal for complete fit failures
                continue

            weighted_sum = 0.0
            for m, weight in auto_reject_weights.items():
                weighted_sum += scaled_metrics[m][c] * weight
            compound_scores[c] = weighted_sum / total_weight

        # Filter out protected peaks from rejection evaluation
        rejectable_candidates = [c for c in active_calibrants.keys() if c not in protected_peaks]
        
        if not rejectable_candidates:
            print("--> All remaining calibrants are protected. Stopping auto-reject.")
            break

        # Identify candidate with the worst compound score
        worst_cal = max(rejectable_candidates, key=lambda k: compound_scores[k])
        worst_error = cal_metrics[worst_cal]['ppm_90pct']
        worst_max_err = cal_metrics[worst_cal]['ppm_max']

        optimization_history_errors.append(worst_error)

        # Check for severe transient outliers (max deviation > 5x threshold)
        is_outlier = False
        if not np.isnan(auto_reject_threshold) and not np.isnan(worst_max_err):
            if worst_max_err > (5.0 * auto_reject_threshold):
                is_outlier = True

        # Trigger rejection if error threshold is exceeded, fit failed, or outlier is detected
        reject_condition = (np.isnan(worst_error) or worst_error > auto_reject_threshold or is_outlier)

        if reject_condition and len(active_calibrants) > calculated_min_calibrants:
            err_str = "FIT FAILED" if np.isnan(worst_error) else f"{worst_error:.1f} ppm"
            if is_outlier and worst_error <= auto_reject_threshold:
                print(f"--> Auto-rejecting '{worst_cal}' due to MAX outlier ({worst_max_err:.1f} > 5 * {auto_reject_threshold:.1f} ppm). Retrying...")
            else:
                print(f"--> Auto-rejecting '{worst_cal}' (Error: {err_str}). Retrying calibration...")

            optimization_history_labels.append(f"Rejected\n{worst_cal}")
            rejection_history[worst_cal] = worst_error if not np.isnan(worst_error) else float('inf')
            
            # --- OPTIONAL INTERMEDIATE ITERATION PLOTTING ---
            if plot_flag and plot_every_iteration:
                print(f"--- Intermediate Plot: {len(active_calibrants)} calibrants (Booting '{worst_cal}') ---")
                
                interm_timestamps = timestamps if averaging_interval == 0 else midpoint_timestamps

                fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
                np.random.seed(55)
                interm_colors = np.random.rand(len(active_calibrants), 3)
                x_positions = np.linspace(0, len(interm_timestamps) - 1, len(active_calibrants), dtype=int)

                for i, (cal, color) in enumerate(zip(active_calibrants.keys(), interm_colors)):
                    if i < dev_arr.shape[1]:
                        ax.plot(interm_timestamps, dev_arr[:, i], color=color)

                ax.set_ylabel('ppm')
                ax.grid(True, linestyle='--', alpha=0.5)

                ylim_top = ax.get_ylim()[1]
                for i, (cal, color) in enumerate(zip(active_calibrants.keys(), interm_colors)):
                    if i < len(x_positions) and len(interm_timestamps) > 0:
                        x_pos = interm_timestamps[x_positions[i]]
                        ax.text(x_pos, ylim_top * 1.05, cal, color=color, fontsize=8, ha='center')

                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.ticklabel_format(style='plain', axis='y')
                plt.tight_layout()
                if show_plot_flag:
                    plt.show()
                plt.close()

            auto_rejected.append(worst_cal)
            del active_calibrants[worst_cal]
            continue  # Repeat loop without the rejected calibrant
            
        else:
            # --- m/z Anchor Gap Protection ---
            optimization_history_labels.append("Final Set")
            peaks_restored = False
            
            if mz_anchor:
                # Sort active calibrants by m/z to evaluate gap sizes
                active_mzs = sorted([(c, calibrant_info[c]["m/Q"]) for c in active_calibrants], key=lambda x: x[1])
                
                for i in range(len(active_mzs) - 1):
                    mz1 = active_mzs[i][1]
                    mz2 = active_mzs[i+1][1]
                    
                    # Detect uncalibrated gaps exceeding requested spacing threshold
                    if mz2 - mz1 > mz_anchor_spacing:
                        gap_candidates = [c for c in auto_rejected if mz1 < calibrants[c] < mz2]
                        if gap_candidates:
                            # Restore candidate with lowest recorded error prior to rejection
                            best_candidate = min(gap_candidates, key=lambda c: rejection_history.get(c, float('inf')))
                            print(f"--> MZ Anchor: Gap > {mz_anchor_spacing} m/z detected between {mz1:.1f} and {mz2:.1f}. Restoring '{best_candidate}'.")
                            
                            active_calibrants[best_candidate] = calibrants[best_candidate]
                            protected_peaks.add(best_candidate)
                            auto_rejected.remove(best_candidate)
                            peaks_restored = True
                            
            if peaks_restored:
                continue  # Repeat loop with restored anchor peak

            if len(active_calibrants) <= calculated_min_calibrants and (np.isnan(worst_error) or worst_error > auto_reject_threshold or is_outlier):
                print(f"--> Reached minimum calibrants ({calculated_min_calibrants}). Stopping auto-reject.")
            else:
                print(f"--> All remaining calibrants meet auto-reject criteria. Optimization complete!")
            break

    # --- Clean up completely failed calibrants ---
    dev_arr = np.array(batch_deviations, dtype=float)
    failed_compounds = [cal for i, cal in enumerate(active_calibrants.keys()) if np.all(np.isnan(dev_arr[:, i]))]
    if failed_compounds:
        print(f"\nCleaning up: Removing completely failed peaks from final output: {failed_compounds}")
        for cal in failed_compounds:
            del active_calibrants[cal]

        # Filter batch matrices to exclude failed columns
        valid_indices = [i for i, cal in enumerate(list(active_calibrants.keys()) + failed_compounds) if cal not in failed_compounds]
        batch_deviations = dev_arr[:, valid_indices].tolist()
        batch_min_err = np.array(batch_min_err, dtype=float)[:, valid_indices].tolist()
        batch_measured_sip = np.array(batch_measured_sip, dtype=float)[:, valid_indices].tolist()
        batch_fwhm_values = np.array(batch_fwhm_values, dtype=float)[:, valid_indices].tolist()
        batch_fwhm_squared_values = np.array(batch_fwhm_squared_values, dtype=float)[:, valid_indices].tolist()
        batch_resolution_values = np.array(batch_resolution_values, dtype=float)[:, valid_indices].tolist()
        batch_calibrant_mz_values = np.array(batch_calibrant_mz_values, dtype=float)[:, valid_indices].tolist()

    # Calculate summary statistics across time
    abs_min_err_arr = np.abs(np.array(batch_min_err, dtype=float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_err = np.nanmean(abs_min_err_arr, axis=0)
        median_err = np.nanmedian(abs_min_err_arr, axis=0)
        std_err = np.nanstd(abs_min_err_arr, axis=0)
        max_err = np.nanmax(abs_min_err_arr, axis=0)

    # Calculate mean and median SIP drift curves
    if track_sip_drift:
        drifts_matrix = np.array([batch_individual_drifts[comp] for comp in track_sip_drift])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_drift_curve = np.nanmean(drifts_matrix, axis=0)
            median_drift_curve = np.nanmedian(drifts_matrix, axis=0)
    else:
        average_drift_curve = None

    # ---------------------------------------------------------
    # STEP 5: Package Results
    # ---------------------------------------------------------
    calibration_results = {
        "calibrants": pd.DataFrame(active_calibrants.items(), columns=['calibrants', 'mass']),
        "batch_measured_sip": np.array(batch_measured_sip),
        "batch_calibrant_mz_values": np.array(batch_calibrant_mz_values),
        "batch_fwhm_values": np.array(batch_fwhm_values),
        "batch_fwhm_squared_values": np.array(batch_fwhm_squared_values),
        "batch_massaxes": np.array(batch_massaxes),
        "batch_deviations": np.array(batch_deviations),
        "batch_min_err": np.array(batch_min_err),
        "batch_params": np.array(batch_params),
        "mass_cal_mode": mass_cal_mode,
        "midpoint_timestamps": midpoint_timestamps,
        "interval_indices": interval_indices,
        "mean_absolute_min_mass_error": mean_err,
        "median_absolute_min_mass_error": median_err,
        "std_absolute_min_mass_error": std_err,
        "max_absolute_min_mass_error": max_err,
        "batch_resolution_values": batch_resolution_values,
        "mean_resolution": np.nanmean([r for vals in batch_resolution_values for r in vals]),
        "per_calibrant_mean_resolution": np.nanmean(np.array(batch_resolution_values, dtype=float), axis=0),
        "calibrant_individual_sip_drifts": {
            k: v for k, v in batch_individual_drifts.items() 
            if not np.isnan(v).all()
        } if track_sip_drift else None,
        "average_sip_drift": np.array(average_drift_curve.tolist() if track_sip_drift else None),
        "median_sip_drift": np.array(median_drift_curve.tolist() if track_sip_drift else None),
        "tof_axis": tof_axis
    }

    # ---------------------------------------------------------
    # STEP 6: Render Calibration Summary Plot
    # ---------------------------------------------------------
    if plot_flag:
        masscal_timestamps = timestamps if averaging_interval == 0 else midpoint_timestamps
        batch_masscal_df = {"timestamps": masscal_timestamps}

        # Extract parameters into time-series dictionary
        num_params = len(batch_params[0])
        for p_idx in range(num_params):
            batch_masscal_df[f'p{p_idx+1}'] = [params[p_idx] for params in batch_params]

        # Add per-calibrant PPM deviations and mass errors
        for i, cal in enumerate(active_calibrants.keys()):
            batch_masscal_df[f"{cal}_ppm"] = [deviation[i] if i < len(deviation) else np.nan for deviation in batch_deviations]
            batch_masscal_df[f"{cal}_mass_err"] = [m_err[i] if i < len(m_err) else np.nan for m_err in batch_min_err]

        extra_plots = 3 if track_sip_drift else 2
        total_plots = num_params + extra_plots
        fig, axs = plt.subplots(total_plots, 1, figsize=(12, 2 * total_plots), sharex=True, dpi=plot_dpi)

        param_colors = ["red", "black", "blue", "orange", "green"]

        # Plot parameters dynamically in reverse order
        for p_idx in range(num_params - 1, -1, -1):
            ax_idx = (num_params - 1) - p_idx
            color = param_colors[p_idx % len(param_colors)]
            axs[ax_idx].plot(batch_masscal_df['timestamps'], batch_masscal_df[f'p{p_idx+1}'], color=color)
            axs[ax_idx].set_ylabel(f'p{p_idx+1}')
            axs[ax_idx].grid(True, linestyle='--', alpha=0.5)

        dev_ax_idx = num_params
        err_ax_idx = num_params + 1
        lock_ax_idx = num_params + 2 if track_sip_drift else None

        num_calibrants = len(active_calibrants)
        np.random.seed(43)
        calpeak_colors = np.random.rand(num_calibrants, 3)
        x_positions = np.linspace(0, len(batch_masscal_df['timestamps']) - 1, num_calibrants, dtype=int)
        active_keys_list = list(active_calibrants.keys())

        # Subplot: PPM Deviations
        for i, (cal, color) in enumerate(zip(active_calibrants.keys(), calpeak_colors)):
            axs[dev_ax_idx].plot(batch_masscal_df['timestamps'], batch_masscal_df[f"{cal}_ppm"], color=color)
        axs[dev_ax_idx].set_ylabel('ppm'); axs[dev_ax_idx].grid(True)

        ylim_top = axs[dev_ax_idx].get_ylim()[1]
        for i, (cal, color) in enumerate(zip(active_calibrants.keys(), calpeak_colors)):
            if len(batch_masscal_df['timestamps']) > 0:
                axs[dev_ax_idx].text(batch_masscal_df['timestamps'][x_positions[i]], ylim_top * 1.05, cal, color=color, fontsize=8, ha='center')

        # Subplot: Absolute Mass Errors
        for i, (cal, color) in enumerate(zip(active_calibrants.keys(), calpeak_colors)):
            axs[err_ax_idx].plot(batch_masscal_df['timestamps'], batch_masscal_df[f"{cal}_mass_err"], color=color)
        axs[err_ax_idx].set_ylabel('min mass error'); axs[err_ax_idx].grid(True)

        ylim_top = axs[err_ax_idx].get_ylim()[1]
        for i, (cal, color) in enumerate(zip(active_calibrants.keys(), calpeak_colors)):
            if len(batch_masscal_df['timestamps']) > 0:
                axs[err_ax_idx].text(batch_masscal_df['timestamps'][x_positions[i]], ylim_top * 1.05, cal, color=color, fontsize=8, ha='center')

        # Subplot: SIP Drift Tracking
        if track_sip_drift:
            surviving_tracked = [comp for comp in track_sip_drift if comp in active_keys_list]

            # Iterate over each surviving calibrant plotting SIP drift
            for comp in surviving_tracked:
                color = calpeak_colors[active_keys_list.index(comp)]
                axs[lock_ax_idx].plot(
                    batch_masscal_df['timestamps'], batch_individual_drifts[comp], 
                    color=color, alpha=0.4, linestyle='-'
                )

            # Plot the average SIP drift in black
            axs[lock_ax_idx].plot(
                batch_masscal_df['timestamps'], calibration_results["average_sip_drift"], 
                color='black', lw=1.0, linestyle='-'
            )
            axs[lock_ax_idx].set_ylabel('Δ SIP (indices)')
            axs[lock_ax_idx].grid(True, linestyle='--', alpha=0.5)

            ylim_top_lock = axs[lock_ax_idx].get_ylim()[1]
            x_positions_lock = np.linspace(0, len(batch_masscal_df['timestamps']) - 1, len(surviving_tracked), dtype=int)
            
            for i, comp in enumerate(surviving_tracked):
                color = calpeak_colors[active_keys_list.index(comp)]
                if len(batch_masscal_df['timestamps']) > 0:
                    axs[lock_ax_idx].text(
                        batch_masscal_df['timestamps'][x_positions_lock[i]], 
                        ylim_top_lock * 1.05, comp, color=color, fontsize=8, ha='center'
                    )

        # Apply plain y-axis formatting (suppresses scientific offset notation)
        for ax in axs:
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, mc_summary_plot_name))
        if show_plot_flag:
            plt.show()
        plt.close()

        # ---------------------------------------------------------
        # STEP 7: Render TOF Drift Plot (different from the condensed "SIP drift" plot)
        # ---------------------------------------------------------
        if show_tof_drift_plot:
            sample_indices = np.arange(len(tof_axis))
            for i, cal in enumerate(active_calibrants.keys()):
                # Interpolate SIP positions to absolute TOF in nanoseconds
                batch_masscal_df[f'{cal}_tof'] = [
                    np.interp(measured_sip[i], sample_indices, tof_axis) 
                    if i < len(measured_sip) and not np.isnan(measured_sip[i]) else np.nan 
                    for measured_sip in batch_measured_sip
                ]

            fig, axs_drift = plt.subplots(len(active_calibrants), 1, figsize=(12, 2 * len(active_calibrants)), sharex=True, dpi=plot_dpi)
            if len(active_calibrants) == 1: 
                axs_drift = [axs_drift]

            # Loop through calibrants and calpeak_colors
            for i, (cal, color) in enumerate(zip(active_calibrants.keys(), calpeak_colors)):
                axs_drift[i].plot(batch_masscal_df['timestamps'], batch_masscal_df[f'{cal}_tof'], color=color)
                axs_drift[i].set_ylabel(f'{cal}\nTOF (ns)'); axs_drift[i].grid(True, linestyle='--', alpha=0.5)
                axs_drift[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                axs_drift[i].ticklabel_format(style='plain', axis='y')

            axs_drift[-1].set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, tof_drift_plot_name), dpi=300)
            if show_plot_flag:
                plt.show()
            plt.close()

    # ---------------------------------------------------------
    # STEP 8: Console Calibrant Summary Table
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("Calibrant Performance Summary (Ranked by Mean Abs PPM Error)")
    print("="*70)
    print(f"{'Calibrant':<15} | {'Mean Abs PPM':<15} | {'PPM Std Dev':<15} | {'Mean Abs Mass Err':<20}")
    print("-" * 70)

    cal_stats = []
    dev_arr = np.array(batch_deviations)
    err_arr = np.array(batch_min_err)

    # Aggregate stats for each calibrant
    for original_cal in calibrants.keys():
        if original_cal in auto_rejected and original_cal not in protected_peaks:
            cal_stats.append((original_cal, "AUTO REJECT", "AUTO REJECT", "AUTO REJECT"))
        elif original_cal in failed_compounds:
            cal_stats.append((original_cal, "FIT FAILED", "FIT FAILED", "FIT FAILED"))
        else:
            active_keys = list(active_calibrants.keys())
            if original_cal in active_keys:
                i = active_keys.index(original_cal)
                valid_ppm = dev_arr[:, i].astype(float)[~np.isnan(dev_arr[:, i])]
                valid_err = err_arr[:, i].astype(float)[~np.isnan(err_arr[:, i])]
                
                if len(valid_ppm) > 0:
                    cal_stats.append((original_cal, np.mean(np.abs(valid_ppm)), np.std(valid_ppm), np.mean(np.abs(valid_err))))
                else:
                    cal_stats.append((original_cal, np.nan, np.nan, np.nan))

    # Custom sorting: Numerical errors first (ascending), then AUTO REJECT, then FIT FAILED / NaNs
    def sort_key(x):
        val = x[1]
        if isinstance(val, str):
            return float('inf') - 1 if val == "AUTO REJECT" else float('inf')
        if np.isnan(val):
            return float('inf')
        return val

    # Apply the sorting
    cal_stats.sort(key=sort_key)

    # Make a pretty table of the results
    for stat in cal_stats:
        cal, m_ppm, s_ppm, m_err = stat
        if isinstance(m_ppm, str):
            print(f"{cal:<15} | {m_ppm:<15} | {s_ppm:<15} | {m_err:<20}")
        elif np.isnan(m_ppm):
            print(f"{cal:<15} | {'FIT FAILED':<15} | {'FIT FAILED':<15} | {'FIT FAILED':<20}")
        else:
            print(f"{cal:<15} | {m_ppm:<15.2f} | {s_ppm:<15.2f} | {m_err:<20.5f}")
    print("="*70 + "\n")

    # Stop timer and report runtime
    s_end = time.time()
    print(f"Mass calibration with {len(calibrants)} calibrants complete in {truncate(s_end - s_start, 3)} seconds.")

    return calibration_results

def combined_mc_aux_plots(calibration_results,
                          output_dir=None,
                          plot_mode="average",
                          show_plot_flag=True):
    """
    Generate combined diagnostic figures for mass calibration performance.

    Renders a two-panel layout for each calibration interval (or a single time-averaged 
    summary figure):
      1. **Deviation Plot (2/3 width)**: Time-of-Flight (TOF) vs. m/z curve overlaid 
         with calibrant position markers and a secondary y-axis displaying per-calibrant 
         PPM deviations as vertical stems.
      2. **FWHM² vs. m/z Plot (1/3 width)**: Squared peak width in sample index space 
         plotted against m/z along with a linear regression trendline.

    Parameters
    ----------
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`. Must contain keys:
        ``"batch_massaxes"``, ``"batch_measured_sip"``, ``"batch_calibrant_mz_values"``,
        ``"batch_deviations"``, ``"batch_fwhm_squared_values"``, and ``"tof_axis"``.
    output_dir : str or None, default=None
        Directory path where output plot images will be saved. Defaults to the 
        standard OpenTof plot directory if ``None``.
    plot_mode : {"average", "full"}, default="average"
        Display and export mode:
        
        * ``"average"``: Computes the mean across all time intervals and exports a 
          single summary figure (``mc_aux_plots_average.png``).
        * ``"full"``: Exports individual figures for every single calibration interval 
          into an ``interval_results/`` subfolder.
    show_plot_flag : bool, default=True
        If ``True``, renders the generated figures interactively in the active display.

    Raises
    ------
    ValueError
        If `plot_mode` is unrecognized or if `plot_mode="average"` encounters jagged 
        (non-rectangular) arrays due to failed or missing calibrant fits across intervals.
    """
    # Default to the global OpenTof plotting path if no custom directory is specified
    if output_dir is None:
        output_dir = get_default_plot_dir()
        
    ensure_dir(output_dir)

    # Retrieve calibration batch matrices and physical axes from the results dictionary
    batch_massaxes = calibration_results["batch_massaxes"]
    batch_measured_sip = calibration_results["batch_measured_sip"]
    batch_calibrant_mz_values = calibration_results["batch_calibrant_mz_values"]
    batch_deviations = calibration_results["batch_deviations"]
    batch_fwhm_squared_values = calibration_results["batch_fwhm_squared_values"]

    # --- MODE 1: TIME-AVERAGED SUMMARY PRE-PROCESSING ---
    if plot_mode.lower() == "average":
        # Helper function to ensure arrays are rectangular before attempting matrix reduction
        def check_uniform_length(batch_data, data_name):
            if not batch_data.all(): 
                return  # Skip validation if dataset array is empty
            base_len = len(batch_data[0])
            # Verify that every interval contains an identical number of calibrant entries
            if any(len(arr) != base_len for arr in batch_data):
                raise ValueError(
                    f"Jagged array detected in '{data_name}'! "
                    f"Intervals have varying numbers of data points (e.g., a missing calibration peak/fit failed). "
                    f"Averaging requires uniform lengths across all intervals. "
                    f"Plotting with plot_mode='full' should reveal this."
                )

        # Validate dimensional uniformity across all batch arrays before averaging
        check_uniform_length(batch_massaxes, "batch_massaxes")
        check_uniform_length(batch_measured_sip, "batch_measured_sip")
        check_uniform_length(batch_calibrant_mz_values, "batch_calibrant_mz_values")
        check_uniform_length(batch_deviations, "batch_deviations")
        check_uniform_length(batch_fwhm_squared_values, "batch_fwhm_squared_values")

        # Collapse multi-interval 2D matrices into 1-row averaged representations (ignoring NaNs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress 'Mean of empty slice' warnings
            batch_massaxes = [np.nanmean(batch_massaxes, axis=0)]
            batch_measured_sip = [np.nanmean(batch_measured_sip, axis=0)]
            batch_calibrant_mz_values = [np.nanmean(batch_calibrant_mz_values, axis=0)]
            batch_deviations = [np.nanmean(batch_deviations, axis=0)]
            batch_fwhm_squared_values = [np.nanmean(batch_fwhm_squared_values, axis=0)]

    elif plot_mode.lower() != "full":
        raise ValueError(f"Invalid plot_mode '{plot_mode}'. Use 'full' or 'average'.")

    # --- LOOP OVER CALIBRATION INTERVALS ---
    # Will execute exactly once if plot_mode="average", or N times if plot_mode="full"
    for masscal_interval in range(len(batch_massaxes)):
        # Construct figure with a 2:1 aspect ratio split between the left and right subplots
        fig, (ax_dev, ax_fwhm) = plt.subplots(
            1, 2, 
            figsize=(15, 5), 
            gridspec_kw={'width_ratios': [2, 1]}
        )

        # =========================================================================
        # SUBPLOT 1: TOF vs. m/z Curve & PPM Deviations (Dual Y-Axis)
        # =========================================================================
        # Plot physical TOF (ns) against the calibrated m/z axis (Primary Y-Axis)
        ax_dev.plot(batch_massaxes[masscal_interval],
                    calibration_results['tof_axis'], 
                    label='TOF (ns)', color='red')
        ax_dev.set_xlabel('Mass-to-charge (m/z)')
        ax_dev.set_ylabel('Time-of-flight (ns)', color='red')
        ax_dev.tick_params(axis='y', labelcolor='red')

        # Retrieve the master physical TOF axis in nanoseconds
        tof_axis = calibration_results['tof_axis']
        sample_indices = np.arange(len(tof_axis))

        # Convert fitted Sample Index Positions (SIP) to physical TOF (ns) via linear interpolation
        measured_tof = np.interp(batch_measured_sip[masscal_interval], sample_indices, tof_axis)

        # Identify nearest m/z values on the calibrated mass axis corresponding to measured TOF positions
        closest_mass_values = [
            batch_massaxes[masscal_interval][
                np.argmin(np.abs(calibration_results['tof_axis'] - t_val))
            ]
            for t_val in measured_tof
        ]

        # Draw open circles marking where calibrants were localized on the TOF curve
        ax_dev.plot(closest_mass_values,
                    measured_tof,     
                    marker='o', color='black', markerfacecolor='none',
                    linestyle='None')

        # Create secondary twin Y-axis for rendering PPM deviations
        ax_dev_twin = ax_dev.twinx()
        ax_dev_twin.set_ylabel('Deviation (ppm)', color='blue')
        ax_dev_twin.tick_params(axis='y', labelcolor='blue')

        # Draw vertical stem lines from y=0 to each calibrant's PPM deviation
        for i, exact_mass in enumerate(batch_calibrant_mz_values[masscal_interval]):
            ax_dev_twin.plot([exact_mass, exact_mass],
                             [0, batch_deviations[masscal_interval][i]],
                             color='blue', lw=2)
            
        valid_devs = np.array(batch_deviations[masscal_interval], dtype=float)

        # Determine y-axis boundaries for the deviation twin axis
        if np.all(np.isnan(valid_devs)):
            max_dev = 1.0  # Fallback scale if all deviations are NaN
        else:
            # Find maximum absolute deviation, ignoring NaNs from rejected peaks
            max_dev = max(abs(np.nanmin(valid_devs)), abs(np.nanmax(valid_devs)))
            if max_dev == 0: 
                max_dev = 1.0  # Prevent singular scale limit errors if deviation is 0.0

        # Apply a 2.5% padding buffer so deviation stems do not touch graph borders
        ax_dev_twin.set_ylim(-max_dev * 1.025, max_dev * 1.025)
        ax_dev_twin.axhline(0, color='black', linestyle='--', lw=1)
        ax_dev.set_title('Deviation Plot')

        # =========================================================================
        # SUBPLOT 2: Squared Peak Width (FWHM²) vs. m/z
        # =========================================================================
        mz_values = np.array(batch_calibrant_mz_values[masscal_interval], dtype=float)
        fwhm2 = np.array(batch_fwhm_squared_values[masscal_interval], dtype=float)

        # Filter out NaNs (rejected or failed peak fits)
        valid_mask = ~np.isnan(fwhm2) & ~np.isnan(mz_values)
        mz_valid = mz_values[valid_mask]
        fwhm2_valid = fwhm2[valid_mask]

        # Fit a linear regression model if at least 2 active calibrants exist
        if len(mz_valid) >= 2:
            slope, intercept = np.polyfit(mz_valid, fwhm2_valid, 1)
            fitted_line = slope * mz_valid + intercept

            ax_fwhm.plot(mz_valid, fitted_line, color='red',
                         label=f'Linear fit: y = {slope:.3f}x + {intercept:.3f}')
        else:
            # Fallback legend entry if auto-rejection left fewer than 2 points
            ax_fwhm.plot([], [], color='red', label='Not enough points for fit')

        # Draw scatter points for squared peak widths of surviving calibrants
        ax_fwhm.scatter(mz_valid, fwhm2_valid,
                        marker='o', facecolors='none', color='teal', s=20)

        ax_fwhm.set_xlabel('Mass-to-charge (m/z)')
        ax_fwhm.set_ylabel('Squared Peak Width (FWHM²)')
        ax_fwhm.set_title('Squared Fitted Peak Width vs m/z')
        ax_fwhm.legend()

        # =========================================================================
        # EXPORT AND DISPLAY
        # =========================================================================
        plt.tight_layout()  # Adjust subplot margins to eliminate overlapping labels
        
        if plot_mode == "average":
            fig.suptitle('Averaged Calibration Results', fontsize=14, y=1.02)
            filename = "mc_aux_plots_average.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            if show_plot_flag:
                plt.show()
        else:
            fig.suptitle(f'Calibration Results - Interval {masscal_interval}', fontsize=14, y=1.02)
            filename = f"mc_aux_plots_{masscal_interval}.png"
            individual_interval_path = os.path.join(output_dir, "interval_results")
            ensure_dir(individual_interval_path)
            plt.savefig(os.path.join(individual_interval_path, filename), bbox_inches='tight')

        plt.close()

def mass_dependent_error_plot(calibration_results,
                              output_dir=None,
                              show_plot_flag=True):
    """
    Generate cubic spline interpolations of mass-dependent calibration error across time.

    Fits cubic splines (:class:`scipy.interpolate.BSpline`) to absolute mass errors 
    across calibrant m/z positions for each time interval, alongside a master median 
    spline trendline. Extrapolates trends to the edges of the full mass axis using 
    dashed lines and exports ``mass_dependent_error.png``.

    This plot should still largly be seen as experimental, but I think eventually something
    similar could be used to also apply some sort of mass-dependent correction. In any case,
    the cubic spline interpolation does an excellent job displaying regions of the mass
    axis without a nearby calibrant. If I do find myself looking at this plot it usually
    is comparing the magnetude of the maxiumum "absolute mass error" from the spline with
    the "min_mass_error" subplot in the mc_summary figure. Ideally I'd think the maximum 
    spline error is the same order of magnitude as the min_mass_error bounds---which is 
    often the case if there are sufficently dense calibrants along the full range of the 
    mass axis---but this often is not the case as sometimes a suitible calibrant within 
    certain m/z windows is not found... in these cases I'd think its best to look at this
    plot as more of a "qualitative" result, becuase it may not effect results if you are 
    only calibrating in a narrow m/z range, but also only fitting/integrating peaks
    within that same narrow range.


    Parameters
    ----------
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`. Must contain keys:
        ``"calibrants"``, ``"batch_min_err"``, and ``"batch_massaxes"``.
    output_dir : str or None, default=None
        Directory path where the output figure will be saved. Defaults to the 
        standard OpenTof plot directory if ``None``.
    show_plot_flag : bool, default=True
        If ``True``, displays the plot interactively upon completion.

    Returns
    -------
    median_spline : scipy.interpolate.BSpline
        Fitted cubic spline object representing the median mass-dependent error curve across time.
    time_splines : list of (scipy.interpolate.BSpline or None)
        List of cubic spline objects corresponding to each time interval. Elements for 
        failed or NaN-containing intervals are set to ``None``.

    Raises
    ------
    ValueError
        If all time intervals contain NaNs/Infs or if the median error vector cannot 
        be computed due to missing calibrant data across all intervals.
    """
    from scipy.interpolate import make_interp_spline

    if output_dir is None:
        output_dir = get_default_plot_dir()
        
    ensure_dir(output_dir)

    # Extract calibrant m/z values and the interval error matrix (Intervals x Calibrants)
    mz = calibration_results['calibrants']['mass'].values
    min_err_matrix = calibration_results['batch_min_err']

    # Sort calibrants and corresponding matrix columns in strictly ascending m/z order
    sort_idx = np.argsort(mz)
    mz = mz[sort_idx]
    min_err_matrix = min_err_matrix[:, sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine full spectrum boundaries for extrapolation visualization
    massaxes = calibration_results['batch_massaxes'][0]
    plot_min, plot_max = np.min(massaxes), np.max(massaxes)

    # Construct dense grid spaces for interpolation (within calibrant range) and extrapolation (outer tails)
    mz_dense = np.linspace(mz.min(), mz.max(), 300)
    mz_dense_left = np.linspace(plot_min, mz.min(), 50)
    mz_dense_right = np.linspace(mz.max(), plot_max, 50)

    # Configure time-gradient color mapping using the Spectral colormap
    cmap = plt.get_cmap('Spectral')
    num_times = min_err_matrix.shape[0]
    
    individual_time_splines = []

    # --- STEP 1: Fit and Plot Individual Interval Splines ---
    for i in range(num_times):
        # Calculate normalized color index based on time progression
        color_val = i / max(1, (num_times - 1))
        time_color = cmap(color_val)
        
        slice_err = min_err_matrix[i]
        
        # Skip time slices that contain non-finite numbers (NaNs from failed/rejected peaks)
        if not np.all(np.isfinite(slice_err)):
            individual_time_splines.append(None)
            continue

        # Scatter raw error points for this interval
        ax.scatter(mz, slice_err, color=time_color, alpha=0.4, s=15, zorder=1)
        
        # Fit cubic spline (degree k=3) across m/z for current interval
        individual_time_spline = make_interp_spline(mz, slice_err, k=3)
        individual_time_splines.append(individual_time_spline)
        
        # Plot smooth interpolated spline curve
        time_smooth = individual_time_spline(mz_dense)
        ax.plot(mz_dense, time_smooth, color=time_color, alpha=0.3, linewidth=1, zorder=1)

    # Fail gracefully if no valid intervals were available for spline fitting
    if all(spline is None for spline in individual_time_splines):
        plt.close(fig)
        raise ValueError("All time intervals contain NaNs/Infs. Cannot fit mass-dependent error splines.")

    # --- STEP 2: Compute and Plot Master Median Spline ---
    median_err = np.nanmedian(min_err_matrix, axis=0)

    if not np.all(np.isfinite(median_err)):
        plt.close(fig)
        raise ValueError("Median error contains NaNs/Infs due to missing calibrant data across intervals.")

    # Fit cubic spline to median mass error points
    median_spline = make_interp_spline(mz, median_err, k=3)
    median_smooth = median_spline(mz_dense)

    # Render median anchor points and master trendline
    ax.scatter(mz, median_err, color='black', s=50, label='calibrant m/z', zorder=2)
    ax.plot(mz_dense, median_smooth, color='black', linewidth=4, label='median spline', zorder=3)

    # --- STEP 3: Render Extrapolated Tails ---
    # Store initial y-limits prior to tail plotting to prevent extrapolation runaway from distorting scale
    original_ylim = ax.get_ylim()

    # Draw dashed extrapolation curves beyond the min and max calibrant range
    for i, time_spline in enumerate(individual_time_splines):
        if time_spline is None:
            continue
        color_val = i / max(1, (num_times - 1))
        time_color = cmap(color_val)
        ax.plot(mz_dense_left, time_spline(mz_dense_left), color=time_color, alpha=0.3, linewidth=1, linestyle='--', zorder=1)
        ax.plot(mz_dense_right, time_spline(mz_dense_right), color=time_color, alpha=0.3, linewidth=1, linestyle='--', zorder=1)

    # Plot master median extrapolated dashed tails
    ax.plot(mz_dense_left, median_spline(mz_dense_left), color='black', linewidth=4, linestyle='--', zorder=3)
    ax.plot(mz_dense_right, median_spline(mz_dense_right), color='black', linewidth=4, linestyle='--', zorder=3)
    
    # Restore Y-axis scale to bound view tightly around interpolations
    ax.set_ylim(original_ylim)

    # Formatting and aesthetics
    ax.set_xlabel('m/z')
    ax.set_ylabel('absolute mass error (largely qualitative?)')
    ax.set_title('cubic spline interpolation of calibrant m/z and min_mass_error at every averaging interval (and median)')
    ax.legend()
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle="--")

    # Save and display figure
    plt.tight_layout()
    filename = "mass_dependent_error.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    if show_plot_flag:
        plt.show()
    plt.close()

    # Return master median spline and interval spline list
    return median_spline, individual_time_splines

# Fit the mass calibration function to known m/Q values and measured TOF values
def calibrate_mass(s_i_values, mz_values, mode=2):
    """
    Fit mass calibration equation parameters to measured TOF/SIP and theoretical m/z values.

    Uses non-linear least squares regression (:func:`scipy.optimize.curve_fit`) to optimize
    the parameters of the selected forward calibration function :math:`i(m)` relating mass-to-charge 
    ratio to Sample Index Position (SIP) or Time-of-Flight (TOF) channels.

    Parameters
    ----------
    s_i_values : numpy.ndarray
        1D array of measured Sample Index Positions (SIP) or Time-of-Flight values for known calibrants.
    mz_values : numpy.ndarray
        1D array of exact theoretical mass-to-charge ratios (m/z) corresponding to `s_i_values`.
    mode : int, default=2
        Calibration equation mode index from ``MASS_CAL_MODES`` defining the model 
        architecture and initial parameter guesses (e.g., Mode 0: i(m) = p_1 sqrt{m} + p_2, 
        Mode 2: i(m) = p_1 m^{p_3} + p_2).

    Returns
    -------
    params : numpy.ndarray
        1D array of optimized calibration equation parameters (e.g., :math:`[p_1, p_2]` or :math:`[p_1, p_2, p_3]`).
    """
    # Extract configuration dictionary and initial parameter guesses for requested calibration mode
    mode_dict = MASS_CAL_MODES[mode]
    initial_guess = mode_dict['p0']
    
    # Formulate wrapper function to pass variable-length parameter vectors to curve_fit
    def fit_func(m, *params):
        return mode_dict['fwd'](m, *params)
    
    # Execute non-linear least squares optimization to solve forward function coefficients i(m)
    params, covariance = curve_fit(
        fit_func, 
        mz_values, 
        s_i_values, 
        p0=initial_guess, 
        maxfev=100000
    )
    
    return params

def reference_spectrum_window(timestamps, 
                              ppm_deviation, 
                              window_seconds=600, 
                              methods_weights_dict={'std': 0.4, 'zero': 0.6},
                              min_fraction=0.9):
    """
    Identify the optimal time window for defining a representative reference spectrum.

    Evaluates rolling time windows across a mass accuracy time-series according to user-defined 
    stability and closeness metrics (e.g., standard deviation, deviation from zero, MAD). 
    Normalizes and ranks each rolling window metric, calculating a weighted composite rank score 
    to select the most representative, stable acquisition window.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D array of acquisition timestamps (datetime or timedelta format).
    ppm_deviation : numpy.ndarray
        1D array of mass accuracy error values in PPM across acquisition time.
    window_seconds : int, default=600
        Rolling evaluation window duration in seconds.
    methods_weights_dict : dict, default={'std': 0.4, 'zero': 0.6}
        Dictionary mapping evaluation metric keys to relative weightings.
        Supported keys:
        
        * ``'std'``: Rolling standard deviation (measures window stability).
        * ``'var'``: Rolling variance.
        * ``'range'``: Peak-to-peak span (max - min).
        * ``'mad'``: Median Absolute Deviation (robust variance).
        * ``'zero'``: Absolute value of rolling mean (closeness to zero PPM error).
        * ``'median'``: Absolute distance between rolling mean and global median.
    min_fraction : float, default=0.9
        Minimum required fraction of valid data points inside a rolling window 
        to be eligible for selection.

    Returns
    -------
    mask : numpy.ndarray of bool
        1D boolean array aligned with the original input timestamps array, 
        where ``True`` marks writebufs falling within the chosen reference window.
    best_start : pandas.Timestamp
        Timestamp marking the start of the selected optimal window.
    best_end : pandas.Timestamp
        Timestamp marking the end of the selected optimal window.
    best_metrics : dict
        Dictionary containing raw metric values and the final composite rank score 
        evaluated at the optimal window location.

    Raises
    ------
    ValueError
        If no rolling window meets the minimum valid data fraction constraint or 
        if an unrecognized metric method key is supplied in `methods_weights_dict`.
    """
    # Verify and notify if user-provided method weights do not sum to 1.0
    total_weight = sum(methods_weights_dict.values())
    if not np.isclose(total_weight, 1.0):
        print(f"NOTE: Weights sum to {total_weight:.2f}. "
              f"Normalizing automatically to 1.0 for score calculation.")

    # Convert timestamps to pandas DatetimeIndex and sort chronologically
    timestamps = pd.to_datetime(timestamps)
    sort_idx = np.argsort(timestamps)
    ts_sorted = timestamps[sort_idx]
    ppm_sorted = ppm_deviation[sort_idx]
    ts = pd.Series(ppm_sorted, index=ts_sorted)

    # Determine data completeness inside rolling time windows
    valid_counts = ts.rolling(f"{window_seconds}s").count()
    window_size = valid_counts.max()
    valid_fraction = valid_counts / window_size
    valid_mask = valid_fraction >= min_fraction

    # Containers for storing raw metrics and percentile ranks
    raw_metrics = {}
    ranked_metrics = {}

    # Evaluate each requested stability / accuracy metric over rolling time windows
    for method in methods_weights_dict.keys():
        if method == "std":
            # Measure standard deviation (stability) within window
            metric_series = ts.rolling(f"{window_seconds}s").std()
        elif method == "var":
            # Measure rolling variance within window
            metric_series = ts.rolling(f"{window_seconds}s").var()
        elif method == "range":
            # Measure peak-to-peak span (max - min) within window
            metric_series = ts.rolling(f"{window_seconds}s").apply(
                lambda x: np.nanmax(x) - np.nanmin(x), raw=True
            )
        elif method == "mad":
            # Measure Median Absolute Deviation within window
            metric_series = ts.rolling(f"{window_seconds}s").apply(
                lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
            )
        elif method == "zero":
            # Measure closeness of window rolling mean to zero PPM error
            rolling_mean = ts.rolling(f"{window_seconds}s").mean()
            metric_series = rolling_mean.abs()
        elif method == "median":
            # Measure closeness of window rolling mean to global median PPM error
            med = np.nanmedian(ppm_deviation)
            rolling_mean = ts.rolling(f"{window_seconds}s").mean()
            metric_series = (rolling_mean - med).abs()
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Mask out windows failing min_fraction data coverage requirements
        metric_series = metric_series.where(valid_mask, np.nan)
        raw_metrics[method] = metric_series
        
        # Rank valid windows via percentile ranking (0.0 = best/lowest error, 1.0 = worst)
        ranked_metrics[method] = metric_series.rank(pct=True, ascending=True)

    # Compute weighted composite score across all evaluated metrics
    combined_score = pd.Series(0.0, index=ts.index)
    total_weight = sum(methods_weights_dict.values())
    
    for method, weight in methods_weights_dict.items():
        combined_score += (ranked_metrics[method].fillna(1.0) * weight)

    combined_score = combined_score / total_weight

    # Identify timestamp corresponding to the lowest combined rank score
    best_idx = combined_score.idxmin()
    if pd.isna(best_idx):
        raise ValueError("No valid window found that meets min_fraction requirement.")

    # Extract raw metric values and final score for winning window
    best_metrics = {m: raw_metrics[m].loc[best_idx] for m in methods_weights_dict.keys()}
    best_metrics['combined_score'] = combined_score.loc[best_idx]

    # Calculate start and end boundaries for the optimal window
    best_start = best_idx - pd.Timedelta(seconds=window_seconds)
    best_end = best_idx

    # Build boolean selection mask matching original unsorted input array layout
    mask_sorted = (ts.index >= best_start) & (ts.index <= best_end)
    mask = np.zeros_like(mask_sorted, dtype=bool)
    mask[sort_idx] = mask_sorted

    return mask, best_start, best_end, best_metrics

def define_reference_spectrum(calibration_results, 
                              tofdata,
                              standard_acq_data,
                              timestamps,
                              reference_peak_mass=None,
                              window_seconds=600,                  
                              methods_weights_dict={'std': 0.4, 'zero': 0.6},
                              min_fraction=0.9,
                              peak_type='gaussian',
                              custom_shape=None,
                              cursor_starttime=None,
                              cursor_endtime=None,
                              search_range=None,
                              fast_mode=False,
                              fit_every_n=3,
                              precomputed_global_average=None,
                              plot_visual_trim=False,
                              interval_cmap='prism',
                              plot_flag=True,
                              output_dir=None,
                              show_plot_flag=True,
                              plt_y_max=1,
                              plt_y_min=-0.05,
                              chunk_size=1000,
                              max_iter=100):
    """
    Identifies an 'optimal' time window and computes a high-quality reference mass spectrum.

    Tracks the positional mass drift (PPM deviation) of a prominent reference mass peak
    across time. Evaluates rolling time windows to locate the most stable and ideally
    least artificially broadened window (or uses user-specified time boundaries), then 
    computes the average spectrum and aggregate calibrated mass axis across that window.

    Parameters
    ----------
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`. Must contain keys:
        ``"batch_params"``, ``"interval_indices"``, and ``"mass_cal_mode"``.
    tofdata : dask.array.Array or numpy.ndarray
        2D matrix of time-of-flight spectral data of shape ``(num_writebuf, num_samples)``.
    standard_acq_data : numpy.ndarray
        1D boolean array indicating standard acquisition writebufs (``True`` = valid sample data).
    timestamps : numpy.ndarray
        1D array of acquisition timestamps (datetime or timedelta) for each writebuf.
    reference_peak_mass : float or None, default=None
        Theoretical m/z of the anchor peak used for positional tracking. If ``None``, 
        automatically identifies the highest intensity peak in an estimated average spectrum.
    window_seconds : int, default=600
        Duration in seconds of the rolling window used to select the reference spectrum.
    methods_weights_dict : dict, default={'std': 0.4, 'zero': 0.6}
        Dictionary mapping evaluation metric keys to relative weightings passed to 
        :func:`reference_spectrum_window`.
    min_fraction : float, default=0.9
        Minimum fraction of valid data points required inside a rolling window.
    peak_type : str, default='gaussian'
        Peak shape model to fit ('gaussian', 'lorentzian', 'pseudo_voigt', or 'custom').
    custom_shape : callable or None, default=None
        Callable custom peak shape function required if ``peak_type='custom'``.
    cursor_starttime : pandas.Timestamp, str, or None, default=None
        Manual override start timestamp for the reference window.
    cursor_endtime : pandas.Timestamp, str, or None, default=None
        Manual override end timestamp for the reference window.
    search_range : int or None, default=None
        Width (in sample bins) of the search window around the anchor peak. Defaults to 
        1/1000th of total spectrum length if ``None``.
    fast_mode : bool, default=False
        If ``True``, uses 3-point parabolic peak interpolation instead of non-linear 
        least squares fitting (approx 2-3x faster, slightly less precise).
    fit_every_n : int, default=3
        Subsampling interval for peak tracking (e.g., ``3`` fits every 3rd spectrum and 
        linearly interpolates intermediate offsets).
    precomputed_global_average : numpy.ndarray or None, default=None
        Pre-calculated 1D average intensity spectrum across all writebufs to bypass 
        auto-detection sampling.
    plot_visual_trim : bool, default=False
        If ``True``, trims the y-axis of the PPM deviation time-series plot to the 5th-95th 
        percentiles.
    interval_cmap : str, default='prism'
        Matplotlib colormap name used for background shading of calibration intervals.
    plot_flag : bool, default=True
        If ``True``, exports time-series drift and reference spectrum diagnostic plots.
    output_dir : str or None, default=None
        Directory path to save output plots. Defaults to OpenTof default plot dir.
    show_plot_flag : bool, default=True
        If ``True``, displays generated figures interactively.
    plt_y_max : float, default=1
        Upper y-axis limit for the zoomed spectrum diagnostic plot.
    plt_y_min : float, default=-0.05
        Lower y-axis limit for the zoomed spectrum diagnostic plot.
    chunk_size : int, default=1000
        Number of spectra per chunk block for memory-safe HDF5/Dask reads.
    max_iter : int, default=100
        Maximum iterations allowed for peak fitting optimization.

    Returns
    -------
    reference_spectrum : numpy.ndarray
        1D array containing the time-averaged intensity spectrum across the reference window.
    rs_mass_axis : numpy.ndarray
        1D array representing the calibrated mass-to-charge axis for the reference spectrum.
    ppm_positional_offset_np : numpy.ndarray
        1D array of PPM mass deviations for the anchor peak across all valid writebufs.
    absolute_positional_offset_np : numpy.ndarray
        1D array of absolute m/z positional offsets for the anchor peak.
    rolling_avg : numpy.ndarray
        1D array representing the rolling average of the PPM deviation time-series.
    rolling_time_mask : numpy.ndarray
        1D boolean array indicating valid timestamps included in the rolling average.
    """
    # Verify standard acquisition data coverage; fallback to all spectra if coverage < 50%
    is_50_percent_true = np.mean(standard_acq_data) >= 0.5
    if not is_50_percent_true:
        print("WARNING! Less than 60% of data is standard acquisition data. Using all available writebufs to generate averaged dataset...")
        standard_acq_data = np.full_like(standard_acq_data, True, dtype=bool)

    # Isolate global valid writebuf indices and filter timestamps/interval mapping
    global_valid_indices = np.where(standard_acq_data)[0]
    interval_indicies = calibration_results['interval_indices'][standard_acq_data]
    timestamps = timestamps[standard_acq_data]

    # Retrieve calibration coefficients and equation mode from results dictionary
    batch_params = calibration_results['batch_params']
    mass_cal_mode = calibration_results.get('mass_cal_mode', 2)

    # Compute mean calibration parameters across all intervals to build a baseline axis
    params_avg = np.average(batch_params, axis=0)
    sample_indicies = np.arange(tofdata.shape[-1])

    # --- AUTO-DETECT REFERENCE ANCHOR PEAK ---
    # If no explicit reference mass is provided, find the maximum intensity peak in an average spectrum
    if reference_peak_mass is None:
        if precomputed_global_average is not None:
            average_spectra = precomputed_global_average
        else:
            print(f"reference_peak_mass not provided! Estimating average spectrum using [{chunk_size}] spectra... (controlled by Deployment.chunk_size)")
            total_spectra = tofdata.shape[0]
            
            # Sample a manageable block of spectra to estimate global average profile
            if total_spectra > chunk_size:
                random_start = np.random.randint(0, total_spectra - chunk_size)
                random_end = random_start + chunk_size
            else:
                random_start = 0
                random_end = total_spectra
                
            chunk_data = tofdata[random_start:random_end, :]
            if hasattr(chunk_data, 'compute'):
                chunk_data = chunk_data.compute()
                
            average_spectra = np.mean(chunk_data, axis=0)

        # Locate global intensity maximum and translate sample index to m/z
        print("Automatically determining reference peak mass using the location\nof max intensity within the sampled average spectra...")
        avg_mass_axis = apply_mass_calibration(sample_indicies, params_avg, mode=mass_cal_mode)
        max_intensity_index = np.argmax(average_spectra)
        reference_peak_mass = avg_mass_axis[max_intensity_index]
        print(f"Reference peak determined to be at m/z: {truncate(reference_peak_mass, 2)}...")

    # Initialize tracking arrays for time-series positional offset values
    n_writebufs = len(timestamps) 
    ppm_positional_offset_np = np.full(n_writebufs, np.nan)
    absolute_positional_offset_np = np.full(n_writebufs, np.nan)

    # Extract unique interval identifiers for calibration mapping
    unique_intervals = np.unique(interval_indicies)
    unique_intervals = unique_intervals[~np.isnan(unique_intervals)]

    print(f"Defining reference spectrum across {len(unique_intervals)} calibration intervals and {n_writebufs} writebufs.")

    # Determine default sample index search window (1/1000th of total spectrum length)
    if search_range is None:
        total_si_length = tofdata.shape[-1]
        search_range = int(total_si_length / 1000)

    # --- STEP 1: Establish Global Safe Bounding Box ---
    avg_mass_axis = apply_mass_calibration(sample_indicies, params_avg, mode=mass_cal_mode)
    global_peak_idx = np.argmin(np.abs(avg_mass_axis - reference_peak_mass))

    # Apply 3x safety buffer to accommodate thermal TOF drift without slicing edge clipping
    safety_buffer = int(search_range * 3)
    global_start_idx = max(0, global_peak_idx - safety_buffer)
    global_end_idx = min(tofdata.shape[-1], global_peak_idx + safety_buffer)

    print(f"Loading reference peak signal region [{global_start_idx}:{global_end_idx}] across all writebufs into RAM...")

    # Load bounded slice matrix into RAM: shape (N_valid_spectra, Slice_Width)
    ref_data_matrix = tofdata[global_valid_indices, global_start_idx:global_end_idx]
    if hasattr(ref_data_matrix, 'compute'):
        ref_data_matrix = ref_data_matrix.compute()

    slice_si_axis = sample_indicies[global_start_idx:global_end_idx]

    # --- STEP 2: Fit Reference Peak Across Writebufs ---
    for i, global_i in enumerate(tqdm(global_valid_indices, desc='Fitting reference peak')):
        # Subsample fitting according to fit_every_n step parameter
        if global_i % fit_every_n != 0:
            continue

        interval_idx = int(interval_indicies[i])
        params = batch_params[interval_idx]
        
        # Reconstruct local calibrated mass axis for active interval
        current_mass_axis = apply_mass_calibration(slice_si_axis, params, mode=mass_cal_mode)
        pchip_interp = PchipInterpolator(slice_si_axis, current_mass_axis)

        # Isolate local peak target window inside memory slice
        peak_local_idx = np.argmin(np.abs(current_mass_axis - reference_peak_mass))
        w_start = max(0, int(peak_local_idx - (search_range / 2)))
        w_end = min(len(slice_si_axis), int(peak_local_idx + (search_range / 2)))

        intensityaxis = ref_data_matrix[i, w_start:w_end]
        sub_si_axis = slice_si_axis[w_start:w_end]
        
        intensity_max = np.max(intensityaxis)
        if intensity_max <= 0: 
            continue

        try:
            if fast_mode:
                # 3-Point Parabolic Sub-Bin Peak Estimation
                max_local_idx = np.argmax(intensityaxis)
                if 0 < max_local_idx < len(intensityaxis) - 1:
                    y0, y1, y2 = intensityaxis[max_local_idx - 1], intensityaxis[max_local_idx], intensityaxis[max_local_idx + 1]
                    denom = (y0 - 2 * y1 + y2)
                    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
                    peak_samp_idx_pos = sub_si_axis[max_local_idx] + delta
                else:
                    peak_samp_idx_pos = sub_si_axis[max_local_idx]
            else:
                # Non-Linear Least Squares Peak Fitting
                norm_intensity = intensityaxis / intensity_max
                xc_guess = sub_si_axis[np.argmax(norm_intensity)]
                fwhm_guess = max(np.sum(norm_intensity > 0.5), 1.0)
                
                popt = fit_unconstrained_peaks(
                    x_axis=sub_si_axis,
                    signal=norm_intensity,
                    centers_guess=[xc_guess],
                    fwhms_guess=[fwhm_guess],
                    amplitudes_guess=[1.0],
                    peak_type=peak_type,
                    custom_shape=custom_shape,
                    center_wiggle=2.0,
                    max_iter=max_iter,
                )
                peak_samp_idx_pos = popt[1]

            # Translate fitted index position to m/z and calculate mass offsets
            interpolated_mass = pchip_interp(peak_samp_idx_pos)
            ppm_positional_offset_np[i] = ((reference_peak_mass - interpolated_mass) / reference_peak_mass) * 1e6
            absolute_positional_offset_np[i] = reference_peak_mass - interpolated_mass

        except Exception:
            pass

    # Linearly interpolate missing offset points if subsampling (fit_every_n > 1) was used
    if fit_every_n > 1:
        print(f"Interpolating reference peak position between fitted spectra (fit_every_n={fit_every_n})...")
        
        def fill_nans_linear(arr):
            valid = ~np.isnan(arr)
            if valid.any():
                x_valid = np.where(valid)[0]
                x_missing = np.where(~valid)[0]
                arr[~valid] = np.interp(x_missing, x_valid, arr[valid])
            return arr

        ppm_positional_offset_np = fill_nans_linear(ppm_positional_offset_np)
        absolute_positional_offset_np = fill_nans_linear(absolute_positional_offset_np)

    print(f"Reference peak positional offset determined! Selecting best [{window_seconds}] second window...")

    # --- STEP 3: Reference Window Selection ---
    if (cursor_starttime is None) and (cursor_endtime is None):
        # Automatically determine optimal reference window via stability ranking
        rs_mask, start, end, metrics_dict = reference_spectrum_window(
            timestamps, 
            ppm_positional_offset_np, 
            window_seconds=window_seconds,
            methods_weights_dict=methods_weights_dict,
            min_fraction=min_fraction,
        )

        print()
        print("Defining reference spectrum between timestamps:")
        print(f"{start} -> {end}")
        print("Metrics for best window:")
        for m_name, m_val in metrics_dict.items():
            print(f"  - {m_name}: {m_val:.4f}")
        print("Number of MS inside window:", rs_mask.sum())
    else:
        # Use user-defined manual time window boundaries
        start = cursor_starttime
        end = cursor_endtime
        rs_mask = (timestamps >= start) & (timestamps <= end)
        print(f"User defined reference spectrum interval between:")
        print(f"{start} -> {end}")

    # Compute rolling average of PPM deviations for trendline visualization
    rolling_avg, rolling_time_mask = calculate_rolling_average(
        ppm_positional_offset_np, 
        timestamps,
        rolling_average_window=window_seconds
    )

    # --- STEP 4: Render Drift Plot ---
    if plot_flag:
        plt.figure(figsize=(18, 6))
        
        # Shade background according to calibration interval indices
        cmap = plt.get_cmap(interval_cmap)
        
        start_idx = 0
        # Background interval coloring!
        for i in range(1, len(interval_indicies)):
            if interval_indicies[i] != interval_indicies[i-1] or np.isnan(interval_indicies[i]) != np.isnan(interval_indicies[i-1]):
                interval_val = interval_indicies[start_idx]
                if not np.isnan(interval_val):
                    color = cmap(int(interval_val) % cmap.N)
                    plt.axvspan(timestamps[start_idx], timestamps[i-1], color=color, alpha=0.3, zorder=1, lw=0.3, linestyle=":")
                start_idx = i
                
        interval_val = interval_indicies[start_idx]
        if not np.isnan(interval_val):
            color = cmap(int(interval_val) % cmap.N)
            plt.axvspan(timestamps[start_idx], timestamps[-1], color=color, alpha=0.3, zorder=1, lw=0.3, linestyle=":")

        # Plot raw PPM deviations and rolling average curve
        plt.plot(timestamps[10:], ppm_positional_offset_np[10:], color='steelblue', label='PPM Deviation', zorder=5)
        plt.plot(timestamps[rolling_time_mask][10:], rolling_avg[rolling_time_mask][10:], color='black', label=f'Rolling {window_seconds}s average', zorder=5)

        # Apply visual trim to percentile bounds if requested
        if plot_visual_trim:
            low_end_vis = np.percentile(ppm_positional_offset_np[~np.isnan(ppm_positional_offset_np)], 5)
            high_end_vis = np.percentile(ppm_positional_offset_np[~np.isnan(ppm_positional_offset_np)], 95)
            plt.ylim(low_end_vis, high_end_vis)        

        plt.title(f"Time Series of [{truncate(reference_peak_mass, 3)}] Positional Offset")
        plt.xlabel("DateTime")
        plt.ylabel("PPM Deviation")
        
        # Draw green start and red end reference window boundary lines
        plt.axvline(start, color='green', linestyle='--', linewidth=3, label='Reference Spectrum Start', zorder=6)
        plt.axvline(end, color='red', linestyle='--', linewidth=3, label='Reference Spectrum End', zorder=6)

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        filename = "rs_ppm_deviation.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        if show_plot_flag:
            plt.show()
        plt.close()

    # --- STEP 5: Compute Average Reference Spectrum ---
    if rs_mask.sum() == 0:
        print("[ERROR] No spectra found in reference window. Returning empty spectrum.")
        return np.zeros(tofdata.shape[-1]), np.zeros(tofdata.shape[-1]), ppm_positional_offset_np, absolute_positional_offset_np

    # Map reference window mask back to global writebuf indices
    global_rs_mask = np.zeros(len(tofdata), dtype=bool)
    global_rs_mask[global_valid_indices] = rs_mask
    valid_rs_indices = np.where(global_rs_mask)[0]

    if len(valid_rs_indices) == 0:
        print("[ERROR] No spectra found in reference window. Returning empty spectrum.")
        return np.zeros(tofdata.shape[-1]), np.zeros(tofdata.shape[-1]), ppm_positional_offset_np, absolute_positional_offset_np
    
    unique_mass_cal_intervals = interval_indicies[rs_mask].astype(int)

    # Accumulate spectra across reference window in memory-safe chunks
    sum_spectrum = np.zeros(tofdata.shape[-1], dtype=np.float64)
    for chunk_start in range(0, len(valid_rs_indices), chunk_size):
        chunk_idx = valid_rs_indices[chunk_start:chunk_start + chunk_size]
        chunk_idx = np.sort(chunk_idx)  # Sorting optimizes HDF5/Dask disk access
        
        chunk_data = tofdata[chunk_idx, :]
        if hasattr(chunk_data, 'compute'):
            chunk_data = chunk_data.compute()
            
        sum_spectrum += np.sum(chunk_data, axis=0)

    # Compute mean intensity vector
    tofdata_average = sum_spectrum / len(valid_rs_indices)
    reference_spectrum = np.squeeze(tofdata_average)
    
    print(f"Reference Spectrum shape: {reference_spectrum.shape}")
    print(f"Unique mass calibrations intervals spanning reference spectrum window: {np.unique(unique_mass_cal_intervals)}")

    # Calculate average calibration parameters over intervals spanning reference window
    start_idx = unique_mass_cal_intervals[0]
    end_idx = unique_mass_cal_intervals[-1] + 1
    rs_params = np.nanmean(np.array(batch_params)[start_idx:end_idx], axis=0)

    print()
    print("Average mass calibration parameters across reference spectrum window are:")
    for i, p_val in enumerate(rs_params, 1):
        print(f"p{i}: {truncate(p_val, 6)}")

    # Construct final calibrated reference mass axis
    rs_mass_axis = apply_mass_calibration(sample_indicies, rs_params, mode=mass_cal_mode)

    # --- STEP 6: Render Reference Spectrum Plot ---
    if plot_flag:
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, dpi=300)
        fig.suptitle("Reference Spectrum")

        # Top panel: Full vertical intensity scale
        axs[0].plot(rs_mass_axis, reference_spectrum, color='purple', linewidth=1)
        axs[0].set_ylabel("ions/s")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # Bottom panel: Zoomed vertical scale (plt_y_min to plt_y_max) for baseline detail
        axs[1].plot(rs_mass_axis, reference_spectrum, color='mediumvioletred', linewidth=1)
        axs[1].set_ylabel("ions/s")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].set_ylim(plt_y_min, plt_y_max)

        plt.xlabel("m/z")
        plt.tight_layout()

        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        filename = "reference_spectrum.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        if show_plot_flag:
            plt.show()
        plt.close()

    return reference_spectrum, rs_mass_axis, ppm_positional_offset_np, absolute_positional_offset_np, rolling_avg, rolling_time_mask

def low_pass_filter(data, cutoff_freq, fs, order):
    """
    Apply a zero-phase low-pass Butterworth filter to remove high-frequency noise.

    Converts the requested physical cutoff frequency to normalized Nyquist units 
    and constructs a Butterworth digital filter. Uses forward-backward filtering 
    (:func:`scipy.signal.filtfilt`) to eliminate phase distortion and phase shift.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of spectral or time-series intensity data.
    cutoff_freq : float
        Filter cutoff frequency in Hertz (Hz).
    fs : float
        Sampling frequency in Hertz (Hz).
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    filtered : numpy.ndarray
        1D array of low-pass filtered signal data matching the length of `data`.

    Raises
    ------
    ValueError
        If the normalized cutoff frequency falls outside the open interval (0, 1).
    """
    # Calculate Nyquist frequency limit (half the sampling frequency)
    nyquist = 0.5 * fs

    # Normalize cutoff frequency relative to the Nyquist limit (must be 0.0 < Wn < 1.0)
    normal_cutoff = cutoff_freq / nyquist
    
    # Validate normalized cutoff frequency bounds
    if not (0 < normal_cutoff < 1):
        raise ValueError(f"Invalid cutoff frequency. normal_cutoff={normal_cutoff}")

    # Design Butterworth digital filter coefficients (numerator 'b' and denominator 'a')
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Perform zero-phase forward-backward digital filtering across the signal
    filtered = signal.filtfilt(b, a, data, axis=-1, padtype='odd')
    
    # Sanity check for computational overflow resulting in NaNs
    if np.isnan(filtered).any():
        print("NaNs detected in low_pass_filter output!")

    return filtered

def running_box_filter(data, window_size):
    """
    Isolate baseline minima using a running minimum sliding window filter.

    Applies a 1D minimum filter (:func:`scipy.ndimage.minimum_filter`) across a 1D 
    signal array to track local non-peak intensity troughs.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of spectral intensity data (typically low-pass filtered).
    window_size : int
        Width of the sliding box filter window in sample bins.

    Returns
    -------
    numpy.ndarray
        1D array containing the minimum value found within each sliding window.
    """
    # Apply 1D minimum filter using nearest-neighbor boundary padding
    return minimum_filter(data, size=window_size, mode='nearest')


def smooth_baseline(baseline, smoothing_window):
    """
    Smooth an estimated baseline profile using a 1D Gaussian filter.

    Parameters
    ----------
    baseline : numpy.ndarray
        1D array representing an un-smoothed baseline estimate (e.g., from a minimum filter).
    smoothing_window : float or int
        Standard deviation (:math:`\\sigma`) for the Gaussian kernel in sample bins.

    Returns
    -------
    numpy.ndarray
        1D Gaussian-smoothed baseline array.
    """
    # Apply 1D Gaussian convolution filter across the un-smoothed baseline profile
    return gaussian_filter1d(baseline, sigma=smoothing_window)


def estimate_global_noise_level(data, window_size, noise_percentile=10):
    """
    Estimate the background noise floor from sliding standard deviation percentiles.

    Constructs a sliding window view of the 1D signal, computes local standard 
    deviations across all windows, and returns a robust low percentile value to isolate 
    true background noise from high-variance peak regions.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of spectral intensity data.
    window_size : int
        Width of the sliding window in sample bins.
    noise_percentile : float, default=10
        Percentile threshold (0 to 100) evaluated across local standard deviation windows.

    Returns
    -------
    noise_level : float
        Estimated background noise standard deviation.
    """
    # Create zero-copy 2D sliding window view across the 1D signal
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
    
    # Compute local standard deviation for each window along axis 1
    rolling_std = np.std(windows, axis=1)

    # Pad left-hand boundary to match the original input array length
    padding = np.full(window_size - 1, rolling_std[0])
    rolling_std_aligned = np.concatenate([padding, rolling_std])

    # Calculate low percentile threshold to extract noise floor from non-peak regions
    noise_level = np.percentile(rolling_std_aligned, noise_percentile)

    return noise_level

def adjust_baseline(baseline, noise_level, scaling_factor=1.0):
    """
    Push the baseline upwards by adding a scaled noise floor offset.

    Parameters
    ----------
    baseline : numpy.ndarray
        1D array representing a smoothed baseline estimate.
    noise_level : float
        Estimated background noise standard deviation.
    scaling_factor : float, default=1.0
        Multiplier applied to `noise_level` before adding to the baseline.

    Returns
    -------
    numpy.ndarray
        1D elevated baseline array.
    """
    # Shift baseline upwards by a scaled noise offset to prevent noise clipping near zero
    return baseline + (scaling_factor * noise_level)

# TODO: adaptive window_size decreasing from lower to higher mz? 
# perhaps add some sort of decay parameter?
def determine_baseline(intensity_spectrum, 
                       cutoff_freq=5e7, 
                       fs=1e9, 
                       order=1, 
                       window_size=None, 
                       smoothing_window=10, 
                       noise_percentile=10,
                       noise_scaling=1.0,
                       correct_offset=False,
                       average_good_baseline=None,
                       offset_threshold=0.0025,
                       output_dir=None,
                       plot_flag=True,
                       show_plot_flag=True,
                       plt_log_flag=False,
                       plt_y_max=1,
                       plt_y_min=-0.05,
                       plt_si_vis_center=None,
                       plt_si_vis_window=None):
    """
    Execute a full multi-stage baseline estimation and subtraction pipeline on a single spectrum.

    Processes a raw intensity spectrum through low-pass filtering, minimum running box filtering, 
    Gaussian baseline smoothing, noise floor estimation, baseline elevation, optional DC offset 
    correction, and baseline subtraction. Optionally exports a 3-panel diagnostic visualization.

    Parameters
    ----------
    intensity_spectrum : numpy.ndarray
        1D array of raw spectral intensity values.
    cutoff_freq : float, default=5e7
        Cutoff frequency in Hz for low-pass Butterworth filtering.
    fs : float, default=1e9
        Sampling frequency in Hz.
    order : int, default=1
        Order of the Butterworth low-pass filter.
    window_size : int or None, default=None
        Width of the running minimum box filter window in sample bins. Defaults to 0.1% 
        of the spectrum length if ``None``.
    smoothing_window : float, default=10
        Standard deviation (:math:`\\sigma`) for Gaussian baseline smoothing.
    noise_percentile : float, default=10
        Percentile threshold of rolling standard deviation used to estimate noise.
    noise_scaling : float, default=1.0
        Multiplier applied to the estimated noise floor when elevating the baseline.
    correct_offset : bool, default=False
        If ``True``, evaluates and subtracts DC intercept shifts relative to `average_good_baseline`.
    average_good_baseline : numpy.ndarray or None, default=None
        Reference baseline array required if ``correct_offset=True``.
    offset_threshold : float, default=0.0025
        Minimum absolute intercept shift required to trigger DC offset correction.
    output_dir : str or None, default=None
        Directory path to save diagnostic plots. Defaults to OpenTof default plot dir if ``None``.
    plot_flag : bool, default=True
        If ``True``, exports a 3-panel diagnostic figure (``baseline_subtraction.png``).
    show_plot_flag : bool, default=True
        If ``True``, displays the generated plot interactively.
    plt_log_flag : bool, default=False
        If ``True``, renders the y-axis of diagnostic plots in log scale.
    plt_y_max : float, default=1
        Upper y-axis limit for diagnostic plots.
    plt_y_min : float, default=-0.05
        Lower y-axis limit for diagnostic plots.
    plt_si_vis_center : int or None, default=None
        Center sample index bin for zoomed diagnostic visualization. Defaults to 1/3 of total bins.
    plt_si_vis_window : int or None, default=None
        Half-width window (in sample bins) around `plt_si_vis_center` for visualization.

    Returns
    -------
    baseline_adjusted_spectrum : numpy.ndarray
        1D array containing the final baseline-subtracted intensity spectrum.
    low_pass_data : numpy.ndarray
        1D array of low-pass filtered spectral data.
    smoothed_baseline : numpy.ndarray
        1D array of smoothed baseline prior to noise scaling.
    adjusted_baseline : numpy.ndarray
        1D array of final elevated baseline subtracted from the raw spectrum.
    noise_level : float
        Calculated background noise standard deviation.
    med_diff : float or None
        Median DC offset correction applied, or ``None`` if `correct_offset=False`.

    Raises
    ------
    ValueError
        If ``correct_offset=True`` but `average_good_baseline` is ``None``.
    """
    # Update guess for window size based on the total length of the intensity spectrum
    if window_size is None:
        window_size = int(len(intensity_spectrum) * 0.001)
        print(f"Window size is: {window_size}")

    # Step 1: Suppress high-frequency noise via zero-phase Butterworth filter
    low_pass_data = low_pass_filter(intensity_spectrum, cutoff_freq, fs, order)
    if np.isnan(low_pass_data).any():
        print("NaNs detected in filtered_data!")

    # Step 2: Track local intensity minima using running minimum filter
    baseline = running_box_filter(low_pass_data, window_size)
    if np.isnan(baseline).any():
        print("NaNs detected in baseline!")

    # Step 3: Smooth raw minimum envelope with a Gaussian kernel
    smoothed_baseline = smooth_baseline(baseline, smoothing_window)
    if np.isnan(smoothed_baseline).any():
        print("NaNs detected in smoothed_baseline!")

    # Step 4: Estimate background noise level from rolling standard deviation percentiles
    global_noise_level = estimate_global_noise_level(intensity_spectrum, window_size, noise_percentile)
    if np.isnan(global_noise_level):
        print("NaN detected in noise_level!")

    # Step 5: Elevate baseline by adding scaled noise floor offset
    adjusted_baseline = adjust_baseline(smoothed_baseline, global_noise_level, noise_scaling)
    if np.isnan(adjusted_baseline).any():
        print("NaNs detected in adjusted_baseline!")

    # Step 6: Perform initial baseline subtraction
    baseline_adjusted_spectrum = intensity_spectrum - adjusted_baseline

    # Step 7: Optional DC intercept offset correction (RF interference shift mitigation) (not typical)
    if correct_offset:
        if average_good_baseline is None:
            raise ValueError("'average_good_baseline must be provided if correct_offset=True")
        
        # Calculate average intercept from baseline edges of the reference baseline
        first_point_baseline_avg = average_good_baseline[:10]
        last_point_baseline_avg = average_good_baseline[-10:]
        intercept_avg = (first_point_baseline_avg + last_point_baseline_avg) / 2

        # Calculate average intercept from baseline edges of current spectrum
        first_point_baseline_indv = adjusted_baseline[:10]
        last_point_baseline_indv = adjusted_baseline[-10:]
        intercept_indv = (first_point_baseline_indv + last_point_baseline_indv) / 2

        # Compute median DC intercept shift
        intercept_diff = intercept_indv - intercept_avg
        med_diff = np.median(intercept_diff)
        print(f"Intercept diff {med_diff}")

        # Subtract flat DC shift if threshold is exceeded
        if np.absolute(med_diff) > offset_threshold:
            baseline_adjusted_spectrum = baseline_adjusted_spectrum - med_diff

        avg_abs_distance = np.mean(np.abs(adjusted_baseline - average_good_baseline))
        print(f"maen diff: {avg_abs_distance}")
    else:
        med_diff = None
   
    # Step 8: Render 3-panel diagnostic visualization
    if plot_flag:
        if plt_si_vis_center is None:
            plt_si_vis_center = int(len(intensity_spectrum) / 3) # some point generally earlier in the spectrum...
            print(f"Centering visualization around sample index: {plt_si_vis_center}")
        if plt_si_vis_window is None:
            plt_si_vis_window = 1000
            print(f"Using a +-sample index window of: {plt_si_vis_window}")

        plt_si_low = plt_si_vis_center - plt_si_vis_window
        plt_si_high = plt_si_vis_center + plt_si_vis_window

        # Helper to toggle log or linear scaling across all subplots
        def toggle_log(axs, use_log=True):
            for ax in axs:
                ax.set_yscale('log' if use_log else 'linear')
                ax.set_ylim(plt_y_min,plt_y_max)

        # Visualization of the MS baseline definition procedure
        fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True, dpi=300)
        fig.suptitle("MS Baseline Subtraction")

        # Subplot 1: Low-Pass Filtered Signal
        axs[0].plot(intensity_spectrum, color='purple', linewidth=1, alpha=0.5)
        axs[0].plot(low_pass_data, color='indigo', linewidth=1)
        axs[0].set_ylabel("ions/s")
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].set_ylim(plt_y_min, plt_y_max)
        axs[0].set_title("(1) Low Pass Filter")

        # Subplot 2: Smoothed and Adjusted Baselines
        axs[1].plot(intensity_spectrum, color='purple', linewidth=1, alpha=0.5)
        axs[1].plot(smoothed_baseline, color='green', linewidth=1)
        axs[1].plot(adjusted_baseline, color='blue', linewidth=1)
        axs[1].set_ylabel("ions/s")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].set_ylim(plt_y_min, plt_y_max)
        axs[1].set_title("(2) Smoothed Baseline + Noise Scaling")

        # Subplot 3: Final Baseline Subtracted Spectrum
        axs[2].plot(baseline_adjusted_spectrum, color='darkgreen', linewidth=1)
        axs[2].plot(intensity_spectrum, color='purple', linewidth=1, alpha=0.5)
        axs[2].set_ylabel("ions/s")
        axs[2].grid(True, linestyle='--', alpha=0.5)
        axs[2].set_ylim(plt_y_min, plt_y_max)
        axs[2].set_title("(3) Subtraction Applied")

        plt.xlim(plt_si_low, plt_si_high)

        # Turn log scale ON for all three
        toggle_log(axs, plt_log_flag)

        plt.xlabel("sample index")
        plt.tight_layout()

        if output_dir is None:
            output_dir = get_default_plot_dir()
            ensure_dir(output_dir)
            
            filename = "baseline_subtraction.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        
        if show_plot_flag:
            plt.show()
        plt.close()

    return baseline_adjusted_spectrum, low_pass_data, smoothed_baseline, adjusted_baseline, global_noise_level, med_diff, #avg_abs_distance

def get_mass_axis_for_ms_i(ms_i, calibration_results, sample_indexes):
    """
    Reconstruct the calibrated m/z axis for a specific writebuf (mass spectrum) index.

    Retrieves the calibration interval index and corresponding parameter vector for 
    a given writebuf index `ms_i`, then evaluates the calibration equation 
    :func:`apply_mass_calibration` across `sample_indexes`.

    Parameters
    ----------
    ms_i : int
        0-based index of the target mass spectrum / writebuf.
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`. Must contain keys:
        ``"interval_indices"``, ``"batch_params"``, and ``"mass_cal_mode"``.
    sample_indexes : numpy.ndarray
        1D array of sample index channel positions (e.g., ``np.arange(num_samples)``).

    Returns
    -------
    spectra_mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge (m/z) values aligned with `sample_indexes`.
    """
    # Extract interval index corresponding to this writebuf
    mass_cal_interval = calibration_results['interval_indices'][ms_i]
    
    # Retrieve calibration equation parameters for this interval
    params = calibration_results['batch_params'][mass_cal_interval]
    mode = calibration_results.get('mass_cal_mode', 2)

    # Compute calibrated m/z axis across sample index positions
    spectra_mass_axis = apply_mass_calibration(sample_indexes, params, mode=mode)
    return spectra_mass_axis