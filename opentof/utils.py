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

# utils.py
from chemparse import parse_formula
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import re
import os
from sklearn.metrics import (
    root_mean_squared_error, r2_score, 
    d2_absolute_error_score, mean_absolute_error, 
    explained_variance_score, max_error
)
from sklearn.cluster import DBSCAN

class GlobalMinMaxScaler:
    """
    Global Min-Max Scaler for array-wide feature normalization.

    Scales an entire array or dataset to the range [0, 1] using a single global 
    minimum and maximum value. This scaler mimics the API of Scikit-Learn's 
    ``MinMaxScaler``, but operates globally across the entire input matrix rather 
    than feature-wise (column-wise). 

    It is particularly useful for variables like mass axes or time-of-flight 
    axes where consistent global scaling across all samples is desired and relative 
    coordinate spacing must be preserved.

    Attributes
    ----------
    global_min_ : float or None
        The global minimum value computed during fitting.
    global_max_ : float or None
        The global maximum value computed during fitting.
    """

    def __init__(self):
        """Initialize the GlobalMinMaxScaler with unassigned extrema attributes."""
        # Initialize internal placeholders for global dataset extrema
        self.global_min_ = None
        self.global_max_ = None

    def fit(self, arr: np.ndarray):
        """
        Compute the global minimum and maximum values of the input array.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array containing values to fit.

        Returns
        -------
        self : GlobalMinMaxScaler
            Fitted scaler instance.
        """
        # Find and store the absolute scalar minimum across all elements
        self.global_min_ = arr.min()
        # Find and store the absolute scalar maximum across all elements
        self.global_max_ = arr.max()
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Scale the input array to the range [0, 1] using fitted global extrema.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array to scale.

        Returns
        -------
        arr_scaled : numpy.ndarray
            Globally scaled array bounded in the range [0, 1].

        Raises
        ------
        ValueError
            If the scaler has not been fitted prior to calling `transform`.
        """
        # Guard clause: Ensure fit() has been executed before transforming
        if self.global_min_ is None or self.global_max_ is None:
            raise ValueError("GlobalMinMaxScaler must be fitted before calling transform.")

        # Compute range denominator
        denominator = self.global_max_ - self.global_min_
        
        # Avoid division-by-zero if the array contains flat/constant values
        if denominator == 0:
            denominator = 1.0

        # Apply global min-max scaling equation: (X - min) / (max - min)
        return (arr - self.global_min_) / denominator

    def inverse_transform(self, arr_scaled: np.ndarray) -> np.ndarray:
        """
        Reconstruct original unscaled array values from scaled data [0, 1].

        Parameters
        ----------
        arr_scaled : numpy.ndarray
            Scaled array bounded in [0, 1] to revert.

        Returns
        -------
        arr_reconstructed : numpy.ndarray
            Reconstructed array scaled back to original data limits.

        Raises
        ------
        ValueError
            If the scaler has not been fitted prior to calling `inverse_transform`.
        """
        # Guard clause: Ensure fit() has been executed before attempting inverse transform
        if self.global_min_ is None or self.global_max_ is None:
            raise ValueError("GlobalMinMaxScaler must be fitted before calling inverse_transform.")

        # Compute range denominator
        denominator = self.global_max_ - self.global_min_
        
        # Avoid zero-range multiplication issue on flat datasets
        if denominator == 0:
            denominator = 1.0

        # Revert scaling using inverse equation: (X_scaled * range) + min
        return arr_scaled * denominator + self.global_min_

    
def load_peak_list(filepath, make_calibrant_list=False):
    """
    Load a peak list TSV file and extract target chemical formulas/ions.

    Parses a tab-delimited peak table file, filters out invalid or unknown 
    compound entries (such as place-holders or unknown markers), and optionally 
    constructs a calibrant dictionary mapping valid formulas to exact masses.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the tab-separated (.txt or .tsv) peak list file containing an 'ion' column.
    make_calibrant_list : bool, default=False
        If True, calculates exact theoretical mass values for each valid compound 
        and returns a calibrants dictionary.

    Returns
    -------
    df : pandas.DataFrame
        The full parsed DataFrame loaded from the target file.
    compounds_or_calibrants : list of str or dict
        If `make_calibrant_list=False`, returns a list of valid compound formula strings.
        If `make_calibrant_list=True`, returns a dictionary mapping valid formulas to exact $m/z$.
    skipped_compounds : list of str
        List of formula strings that were ignored or filtered out during parsing.
    """
    # Read tab-separated peak table file into a pandas DataFrame
    df = pd.read_csv(filepath, sep="\t")

    compounds = []
    skipped_compounds = []

    # Iterate through all values in the 'ion' column
    for s in df['ion'].values:
        # Exclude placeholders containing brackets like '<unknown0001>'
        if "<" in s:
            skipped_compounds.append(s)
            continue
        # Exclude any generic 'Unknown' entries
        if "Unknown" in s:
            skipped_compounds.append(s)
            continue
        else:
            # Retain valid formula string
            compounds.append(s)

    # Optionally construct a calibrant dictionary mapping formula -> theoretical exact mass
    if make_calibrant_list:
        calibrants = {}

        # Look up exact theoretical mass for each valid compound formula
        for c in compounds:
            calibrants[c] = return_mass(c)
            
        return df, calibrants, skipped_compounds
    else:
        return df, compounds, skipped_compounds

def get_nm_segment_data(nm, spectra, mass_axis, tof_axis, nm_search_range=0.5):
    """
    Extract spectral, mass, and time-of-flight slices around a target nominal mass.

    Slices a localized window centered at a nominal mass integer ($m/z$) 
    across intensity, mass-to-charge, and time-of-flight coordinate arrays.

    Parameters
    ----------
    nm : int or float
        Target nominal mass center in $m/z$ units (e.g., 78).
    spectra : numpy.ndarray
        1D array of spectral intensity values.
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge ($m/z$) coordinates.
    tof_axis : numpy.ndarray
        1D array of physical time-of-flight coordinates in nanoseconds.
    nm_search_range : float, default=0.5
        Half-width window size in $m/z$ units ($[nm - Δ, nm + Δ]$).

    Returns
    -------
    int_segment : numpy.ndarray
        1D array slice of spectral intensities within the search window.
    mz_segment : numpy.ndarray
        1D array slice of $m/z$ coordinates within the search window.
    tof_segment : numpy.ndarray
        1D array slice of time-of-flight coordinates within the search window.
    """
    # Create boolean slicing mask for coordinates falling inside [nm - search_range, nm + search_range]
    mask = (mass_axis >= (nm - nm_search_range)) & (mass_axis <= (nm + nm_search_range))

    # Slice each axis using the unified boolean mask
    mz_segment = mass_axis[mask]
    tof_segment = tof_axis[mask]
    int_segment = spectra[mask]

    # Return sliced intensity, m/z, and TOF segments
    return int_segment, mz_segment, tof_segment

def spectra_nm_split(spectra, mass_axis, tof_axis, sr=0.5):
    """
    Partition a single spectrum into nominal mass dictionary segments.

    Identifies all unique nominal mass integers present along the mass axis, 
    slices local search windows around each nominal mass, and returns a dictionary 
    keyed by integer nominal mass values.

    Parameters
    ----------
    spectra : numpy.ndarray
        1D array of spectral intensity values.
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge ($m/z$) coordinates.
    tof_axis : numpy.ndarray
        1D array of time-of-flight coordinates in nanoseconds.
    sr : float, default=0.5
        Half-width nominal mass search range in $m/z$ units ($Δ m/z$).

    Returns
    -------
    nm_dict : dict
        Dictionary keyed by integer nominal masses (e.g., `78`), where each value 
        is a sub-dictionary containing:
        
        * ``'int_segment'``: 1D intensity array slice.
        * ``'ma_segment'``: 1D mass axis array slice.
        * ``'tof_segment'``: 1D time-of-flight axis array slice.
    """
    nm_dict = {}

    # Find all unique nominal mass integers present on the mass axis
    nominal_masses = np.unique(np.round(mass_axis)).astype(int)

    # Slice axes for each nominal mass integer
    for nm in nominal_masses:
        # Create boolean mask for current nominal mass window
        mask = (mass_axis > (nm - sr)) & (mass_axis < (nm + sr))
        
        # Store sliced array segments under integer key
        nm_dict[nm] = {
            'int_segment': spectra[mask],
            'ma_segment': mass_axis[mask],
            'tof_segment': tof_axis[mask]
        }

    return nm_dict

def std_list_array_lengths(all_segments, mode="pad"):
    """
    Standardize a list of 1D arrays of varying lengths to uniform dimensions.

    Aligns uneven 1D array segments (such as nominal mass slices across multiple 
    scans) so they can be stacked into a uniform 2D matrix. Supports edge padding, 
    truncation, and linear trend extrapolation.

    Parameters
    ----------
    all_segments : sequence of numpy.ndarray
        List or sequence of 1D arrays with potentially varying element lengths.
    mode : {'pad', 'truncate', 'extrap'}, default='pad'
        Standardization strategy:
        
        * ``'pad'``: Extend shorter arrays to the maximum length by repeating boundary edge values.
        * ``'truncate'``: Slice longer arrays down to the minimum array length.
        * ``'extrap'``: Extend shorter arrays by linearly extrapolating the trajectory of the final two elements.

    Returns
    -------
    numpy.ndarray
        Uniform 2D NumPy array of shape ``(len(all_segments), target_length)``.

    Raises
    ------
    ValueError
        If an unsupported `mode` string option is supplied.
    """
    # Extract element lengths across all input segment arrays
    lengths = [len(seg) for seg in all_segments]

    # Fast path: If all input arrays already have identical lengths, stack and return immediately
    if len(set(lengths)) == 1:
        return np.array(all_segments)

    # Determine target matrix column dimension based on active mode
    if mode == "truncate":
        target_len = np.min(lengths)  # Truncate down to shortest array length
    else:
        target_len = np.max(lengths)  # Extend up to longest array length

    # Container for standardized uniform arrays
    standardized = []
    
    # Process each segment array individually
    for seg in all_segments:
        n = len(seg)
        
        # Case 1: Array already matches target length
        if n == target_len:
            seg_std = seg
            
        # Case 2: Array is shorter than target length (requires extension)
        elif n < target_len:
            if mode in ["pad", "truncate"]:
                # Determine missing element count
                pad_width = target_len - n
                # Repeat the final array boundary value (edge mode padding)
                seg_std = np.pad(seg, (0, pad_width), mode="edge")
                
            elif mode == "extrap":
                # Linearly extrapolate trajectory using the slope of the final two elements
                extra = np.linspace(seg[-2], seg[-1], target_len - n + 1)[1:]
                # Concatenate original segment with linear extrapolation extension
                seg_std = np.concatenate([seg, extra])
                
            else:
                raise ValueError(f"Unsupported mode: {mode}")
                
        # Case 3: Array is longer than target length (requires truncation)
        else:
            # Slice array from start up to target length, discarding trailing bins
            seg_std = seg[:target_len]
        
        # Append standardized array to container
        standardized.append(seg_std)

    # Stack list of uniform 1D arrays into a 2D NumPy array
    return np.array(standardized)

def get_raw_nm_data(nm, tofdata, tof_axis, calibration_results, mode='extrap'):
    """
    Extract and standardize spectral segment time series around a nominal mass window.

    Iterates through all spectra (writebufs) in memory-safe chunks, reconstructs the 
    calibrated mass axis for each scan, and slices out localized segment profiles 
    around the requested nominal mass integer ($nm +- 0.5 m/z$). Computes 
    the negative second derivative to highlight peak curvature and standardizes 
    all extracted arrays to uniform lengths.

    Parameters
    ----------
    nm : int or float
        Target nominal mass integer ($m/z$) to slice across all spectra.
    tofdata : dask.array.Array or numpy.ndarray
        2D matrix of time-of-flight spectral data of shape ``(num_spectra, num_samples)``.
    tof_axis : numpy.ndarray
        1D array of physical time-of-flight coordinates in nanoseconds.
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`. Must contain keys:
        ``"interval_indices"``, ``"batch_params"``, and ``"mass_cal_mode"``.
    mode : {'extrap', 'pad', 'truncate'}, default='extrap'
        Length standardization strategy passed to :func:`std_list_array_lengths`.

    Returns
    -------
    all_int_segments_std : numpy.ndarray
        Standardized 2D array of intensity segments of shape ``(num_spectra, segment_length)``.
    all_neg2d_segments_std : numpy.ndarray
        Standardized 2D array of negative second derivative values ($-d^2y/dx^2$).
    all_ma_segments_std : numpy.ndarray
        Standardized 2D array of calibrated mass-to-charge ($m/z$) axes.
    all_tof_segments_std : numpy.ndarray
        Standardized 2D array of time-of-flight axes in nanoseconds.
    """
    # Import calibration utility locally to break circular module dependencies
    from opentof.mass_calibration import get_mass_axis_for_ms_i

    # Initialize master accumulation lists for extracted nominal mass segments
    all_int_segments = []    # Intensity vectors (y-axis)
    all_ma_segments = []     # Mass-to-charge vectors (x-axis)
    all_neg2d_segments = []  # Negative second derivative vectors
    all_tof_segments = []    # Time-of-flight vectors (ns)

    chunk_size = 1000
    num_spectra = len(tofdata)

    # Process dataset in memory-safe chunks to prevent RAM overflow on large Dask arrays
    for chunk_start in range(0, num_spectra, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_spectra)
        
        # Pull current block slice into RAM and compute Dask computational graph if required
        working_chunk = tofdata[chunk_start:chunk_end, :]
        if hasattr(working_chunk, 'compute'):
            working_chunk = working_chunk.compute()
            
        # Iterate through individual spectra within the active memory chunk
        for local_i in range(chunk_end - chunk_start):
            ms_i = chunk_start + local_i  # Calculate global spectrum index
            
            # Extract 1D raw spectrum intensity vector
            spectra = working_chunk[local_i]

            # Generate sample index channel array [0, 1, 2, ..., num_samples - 1]
            sample_indexes = np.arange(len(spectra))

            # Reconstruct the calibrated mass-to-charge (m/z) axis for this specific scan
            spectra_mass_axis = get_mass_axis_for_ms_i(ms_i, calibration_results, sample_indexes)

            # Slice out the local segment window centered around requested nominal mass (nm +/- 0.5 m/z)
            int_segment, mz_segment, tof_segment = get_nm_segment_data(nm, spectra, spectra_mass_axis, tof_axis)

            # --- Compute Negative Second Derivative (-d^2y / dx^2) ---
            # 1st derivative (d1) finds slope / rate of change
            d1 = np.gradient(int_segment)
            # 2nd derivative (d2) finds curvature / peak tops
            d2 = np.gradient(d1)
            # Negating d2 turns concave-down peak maxima into positive concave-up spikes
            neg2d = -d2

            # Accumulate extracted scan segments into master lists
            all_int_segments.append(int_segment)
            all_ma_segments.append(mz_segment)
            all_neg2d_segments.append(neg2d)
            all_tof_segments.append(tof_segment)

    # Standardize segment vector lengths to uniform matrix dimensions across all scans
    all_int_segments_std = std_list_array_lengths(all_int_segments, mode=mode)
    all_ma_segments_std = std_list_array_lengths(all_ma_segments, mode=mode)
    all_neg2d_segments_std = std_list_array_lengths(all_neg2d_segments, mode=mode)
    all_tof_segments_std = std_list_array_lengths(all_tof_segments, mode=mode)

    # Return standardized 2D segment matrices aligned with global spectrum index (ms_i)
    return all_int_segments_std, all_neg2d_segments_std, all_ma_segments_std, all_tof_segments_std

def plot_peak_fits(ms_i, nm, tofdata_subtracted, peak_list_df, 
                   calibration_results, sample_index_axis,
                   peak_ts, peak_list, peak_width_function,
                   custom_shape,
                   output_dir=None,
                   plot_filename=None,
                   save_plot=True):
    """
    Plot raw spectrum data, deconvoluted peak fits, and total isotopic response for a nominal mass window.

    Extracts a target spectrum by index `ms_i`, reconstructs its calibrated mass axis, 
    retrieves fitted peak amplitudes from `peak_list_df`, evaluates individual empirical 
    peak curves, reconstructs the total theoretical isotopic background signal, and renders 
    a diagnostic plot.

    Parameters
    ----------
    ms_i : int
        0-based index of the target mass spectrum / writebuf to plot.
    nm : int or float
        Target nominal mass integer ($m/z$) defining the visualization window ($nm +- 0.5$).
    tofdata_subtracted : dask.array.Array or numpy.ndarray
        2D baseline-subtracted time-of-flight spectral data array of shape ``(num_spectra, num_samples)``.
    peak_list_df : pandas.DataFrame
        Wide-format fitted peak results DataFrame containing a `MS_index` column and 
        per-peak amplitude columns ``f"{peak}_amplitude"``.
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`.
    sample_index_axis : numpy.ndarray
        1D array of sample index channel positions.
    peak_ts : pandas.DataFrame or dict
        Fitted peak time-series DataFrame passed to :func:`reconstruct_total_isotope_signal`.
    peak_list : list of (str or float)
        Sequence of target chemical formulas or exact $m/z$ floats.
    peak_width_function : callable
        Callable function mapping $m/z$ to expected FWHM resolution.
    custom_shape : callable
        Callable empirical peak shape function.
    output_dir : str or None, default=None
        Export directory path for saved plots. Defaults to OpenTof default plot dir if ``None``.
    plot_filename : str or None, default=None
        Custom filename for the exported image file. Defaults to ``diagnostic_fit_ms{ms_i}_nm{nm}.png`` if ``None``.
    save_plot : bool, default=True
        If ``True``, saves the generated figure to disk.
    """
    # Import utilities locally to prevent circular module dependencies
    from opentof.mass_calibration import get_mass_axis_for_ms_i
    from opentof.isotopes import reconstruct_total_isotope_signal

    # Extract single target spectrum row from 2D array and compute Dask graph if required
    spectra = tofdata_subtracted[ms_i, :]
    if hasattr(spectra, 'compute'):
        spectra = spectra.compute()
        
    # Reconstruct calibrated mass axis for this spectrum index
    spectra_mass_axis = get_mass_axis_for_ms_i(ms_i, calibration_results, sample_index_axis)

    # Extract target row for ms_i from wide-format peak data DataFrame
    try:
        ms_i_row = peak_list_df.loc[peak_list_df['MS_index'] == ms_i].iloc[0]
    except IndexError:
        print(f"MS Index {ms_i} not found in peak_data.")
        return

    # Filter master peak list for targets falling within active nominal mass window (nm +/- 0.5 m/z)
    nm_peaks = []
    for peak in peak_list:
        # Resolve theoretical exact mass for formula string or float
        c_mass = return_mass(peak) if isinstance(peak, str) else float(peak)
        
        # Check if peak center falls inside target nominal mass window
        if (nm - 0.5) < c_mass < (nm + 0.5):
            amp_col = f"{peak}_amplitude"
            
            # Record peak details if amplitude column exists and contains non-NaN value
            if amp_col in ms_i_row.index and not pd.isna(ms_i_row[amp_col]):
                nm_peaks.append({
                    'name': peak,
                    'center_mass': c_mass,
                    'amplitude': ms_i_row[amp_col]
                })

    # Reconstruct cumulative theoretical isotopic signal response from lower-mass parents
    reconstructed_isotope_signal = reconstruct_total_isotope_signal(
        ms_i, 
        peak_ts, 
        peak_list, 
        spectra_mass_axis, 
        peak_width_function, 
        custom_shape=custom_shape
    )

    # Initialize figure canvas
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(spectra_mass_axis, spectra, color='gray', linewidth=1.5, label='RAW Data', alpha=0.7)
    
    # Iterate through each active peak found in nominal mass window and render fitted curve
    max_amp = 0
    for p_info in nm_peaks:
        c_mass = p_info['center_mass']
        amp = p_info['amplitude']
        max_amp = max(max_amp, amp)
        
        # Compute expected FWHM width and scale by empirical custom shape factor if present
        fwhm = peak_width_function(c_mass)
        if custom_shape is not None and hasattr(custom_shape, 'gauss_to_ps'):
            fwhm *= custom_shape.gauss_to_ps
            
        # Render deconvoluted individual peak profile
        peak_curve = custom_shape(spectra_mass_axis, amp, c_mass, fwhm)
        plt.plot(spectra_mass_axis, peak_curve, linestyle="--", linewidth=1.5, alpha=0.8, label=f"Fit: {p_info['name']}")

    # Plot total reconstructed isotopic background response
    plt.plot(spectra_mass_axis, reconstructed_isotope_signal, color='red', linewidth=1.5, linestyle=":", label='Total Isotope Response')
    
    plt.title(f"Diagnostic Fit - MS Index: {ms_i} (m/z ~ {nm})")
    plt.xlabel("m/z")
    plt.ylabel("Intensity (ions/s)")
    
    # Dynamically scale y-axis bounds based on visible window peak heights
    mask = (spectra_mass_axis > nm - 0.5) & (spectra_mass_axis < nm + 0.5)
    if np.any(mask):
        plot_max = max(np.max(spectra[mask]), max_amp)
    else:
        plot_max = max_amp
        
    plt.ylim(-1, plot_max * 1.1 if plot_max > 0 else 10)
    plt.xlim(nm - 0.5, nm + 0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    
    # Export figure to disk if requested
    if save_plot:
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        if plot_filename is None:
            plot_filename = f"diagnostic_fit_ms{ms_i}_nm{nm}.png"
            
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=150)
        
    plt.show()
    plt.close()

def _explore_h5(obj, path=''):
    """
    Recursively inspect and print the internal object hierarchy of an HDF5 container.

    Iterates through groups and datasets within an HDF5 structure, printing object paths, 
    associated attributes, dataset dimensions, data types, and specific dataset contents 
    (such as Tofware metadata arrays like ``TwInfo``).

    Parameters
    ----------
    obj : h5py.File, h5py.Group, or h5py.Dataset
        Active HDF5 object node to inspect.
    path : str, default=''
        Current structural path string tracking recursion depth.

    Returns
    -------
    None
        Prints structural inspection output directly to stdout.
    """
    # Print root level attributes if currently inspecting the root node '/'
    if obj.name == '/':
        for attr_key in obj.attrs:
            print(f"Attribute at {path or '/'}: {attr_key} = {obj.attrs[attr_key]}")

    # Recursively iterate over all member items within the active group
    for key in obj:
        item = obj[key]
        # Construct absolute HDF5 dataset path string
        full_path = f"{path}/{key}".replace('//', '/')
        print(f"\n--- Path: {full_path} ---")

        # Print all metadata attributes bound to the current item
        for attr_key in item.attrs:
            print(f"Attribute at {full_path}: {attr_key} = {item.attrs[attr_key]}")

        # Recurse if current item is a Group node
        if isinstance(item, h5py.Group):
            _explore_h5(item, full_path)
        # Display shape and dtype if current item is a Dataset node
        elif isinstance(item, h5py.Dataset):
            print(f"Dataset shape: {item.shape}, dtype: {item.dtype}")
            # Print raw contents for Tofware information arrays ('TwInfo')
            if key == "TwInfo":
                try:
                    print(f"Contents of {full_path}:")
                    print(item[()])
                except Exception as e:
                    print(f"Could not read dataset {full_path}: {e}")
        else:
            print("Unknown item type")

def print_h5_structure(h5_file_path):
    """
    Open an HDF5 file from disk and display its complete internal group hierarchy.

    Convenience context-manager wrapper around :func:`_explore_h5` that handles file opening 
    and closing automatically.

    Parameters
    ----------
    h5_file_path : str or pathlib.Path
        File path to the target HDF5 file on disk.

    Returns
    -------
    None
        Prints structural inspection output directly to stdout.
    """
    # Open target file in read-only mode and initiate recursive hierarchy traversal
    with h5py.File(h5_file_path, 'r') as f:
        print("--- Root Group: / ---")
        _explore_h5(f)

def median_absolute_deviation(arr):
    """
    Calculate the Median Absolute Deviation (MAD) of an array.

    Computes a robust measure of statistical dispersion defined as:

    .. math::

        MAD = median(|X - median(X)|)

    Parameters
    ----------
    arr : numpy.ndarray or sequence of float
        Input array containing numeric values.

    Returns
    -------
    mad_val : float
        Calculated Median Absolute Deviation.
    """
    # Compute the median value of the input array
    median_val = np.median(arr)
    
    # Calculate absolute deviations from the median
    abs_deviations = np.absolute(arr - median_val)
    
    # Calculate the median of the absolute deviations
    mad_val = np.median(abs_deviations)
    
    return mad_val

def calculate_rolling_average(data, timestamps, rolling_average_window=300):
    """
    Compute a rolling average over irregular time-series data with gap handling.

    Smooths input data using 1D convolution filtering. Automatically identifies acquisition gaps: 
    short gaps below `rolling_average_window` are interpolated through convolution, while long 
    gaps are masked out with :data:`numpy.nan` to prevent edge contamination bleed across missing data.

    Parameters
    ----------
    data : numpy.ndarray
        1D or 2D array of signal intensity values.
    timestamps : numpy.ndarray
        1D array of acquisition timestamps (datetime64, object, or timedelta format).
    rolling_average_window : int or float, default=300
        Duration in seconds of the rolling average convolution window.

    Returns
    -------
    final_results : numpy.ndarray
        1D or 2D array of rolling-averaged signal values matching `data` dimensions, 
        with gap-contaminated regions masked as :data:`numpy.nan`.
    valid_mask : numpy.ndarray of bool
        1D boolean array indicating valid data channels (``True``) versus masked gap edge regions (``False``).
    """
    # Ensure input array is at least 1D
    working_data = np.atleast_1d(data)
    ts_arr = np.asarray(timestamps)
    
    # Convert timestamps to float seconds relative to UNIX epoch depending on input dtype
    if np.issubdtype(ts_arr.dtype, np.datetime64):
        ts_seconds = ts_arr.astype('datetime64[ns]').astype(np.int64) / 1e9
    elif ts_arr.dtype == object:
        ts_seconds = pd.to_datetime(ts_arr).astype(np.int64) / 1e9
    else:
        ts_seconds = ts_arr.astype('timedelta64[ns]').astype(float) / 1e9

    # Calculate median sample time step (dt) and determine convolution kernel size
    diffs = np.diff(ts_seconds)
    dt_median = np.median(diffs)
    window_size = max(1, int(rolling_average_window / dt_median))
    kernel = np.ones(window_size) / window_size

    # Identify time gaps exceeding 1.5x expected sample rate
    gap_threshold = dt_median * 1.5
    gap_indices = np.where(diffs > gap_threshold)[0]
    
    # Initialize valid data mask (True = uncontaminated data)
    valid_mask = np.ones(len(timestamps), dtype=bool)
    imputed_data = working_data.copy().astype(float)

    # Evaluate each identified gap to classify as short or long
    for idx in gap_indices:
        gap_duration = ts_seconds[idx+1] - ts_seconds[idx]
        
        if gap_duration <= rolling_average_window:
            # SHORT GAP: Allow convolution window to bridge gap using mode='same'
            continue 
        else:
            # LONG GAP: Mask out half-window edge buffer to remove boundary bleed artifacts
            edge_buffer = window_size // 2
            start = max(0, idx - edge_buffer + 1)
            end = min(len(timestamps), idx + edge_buffer + 1)
            valid_mask[start:end] = False

    # Execute 1D uniform convolution rolling average along array axis 0
    rolled_results = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), 
        axis=0, 
        arr=imputed_data
    )

    # Mask out contaminated edge regions around long gaps with NaNs
    final_results = rolled_results.astype(float)
    final_results[~valid_mask] = np.nan

    return final_results, valid_mask

def activity_classifier(data, timestamps, window_size=10, lp_cutoff=0.01, lp_fs=1, lp_order=2, 
                         lp_percentile_threshold=90, 
                         plot_flag=False, 
                         output_dir=None,
                         plot_filename="activity_classification.png"):
    """
    Classify active plume or elevation events in time-series data using dual-path detection.

    Combines two complementary detection criteria:
    1. **Prolonged Events**: Low-pass filtered rolling standard deviation exceeding a high percentile threshold.
    2. **Sharp Spikes**: Median-filtered signal difference exceeding a combined p99/p5 noise threshold.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of time-series signal values (e.g., peak area or amplitude).
    timestamps : numpy.ndarray
        1D array of datetime64 timestamps corresponding to `data`.
    window_size : int, default=10
        Sliding window size in sample channels used to compute rolling standard deviation.
    lp_cutoff : float, default=0.01
        Cutoff frequency in Hz for low-pass Butterworth filtering.
    lp_fs : float, default=1
        Sampling frequency in Hz.
    lp_order : int, default=2
        Order of the low-pass Butterworth filter.
    lp_percentile_threshold : float, default=90
        Percentile threshold (0-100) applied to low-pass rolling standard deviation.
    plot_flag : bool, default=False
        If ``True``, renders and exports a 2-panel diagnostic classification figure.
    output_dir : str or None, default=None
        Export directory path for saved plots. Defaults to OpenTof default plot dir if ``None``.
    plot_filename : str, default="activity_classification.png"
        Filename for the saved diagnostic plot.

    Returns
    -------
    results : dict
        A classification dictionary containing:
            * ``"is_active"`` : 1D boolean array (``True`` = Active event).
            * ``"active_data"`` : 1D array containing signal values during active events and ``NaN`` elsewhere.
            * ``"rolling_std"`` : 1D array of rolling standard deviation values.
            * ``"low_pass"`` : 1D array of low-pass filtered rolling standard deviation values.
            * ``"lp_threshold"`` : Percentile cutoff value used for prolonged event detection.
            * ``"spike_threshold"`` : Heuristic noise threshold used for spike override detection.
            * ``"metrics"`` : Sub-dictionary containing total count and percentage of active spectra.
    """
    # Import filtering dependencies locally
    from opentof.mass_calibration import low_pass_filter
    from scipy.signal import medfilt
    from opentof.utils import ensure_dir, get_default_plot_dir

    data = np.asarray(data)
    
    # --- STEP 1: Rolling Standard Deviation ---
    # Construct zero-copy sliding window view to calculate local variability
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
    rolling_std = np.std(windows, axis=1)
    
    # Pad left boundary to align array length with original time-series
    padding = np.full(window_size - 1, rolling_std[0])
    rolling_std_aligned = np.concatenate([padding, rolling_std])

    # --- STEP 2: Low-Pass Filter (Prolonged Event Detection) ---
    # Low-pass filter rolling standard deviation to isolate broad variance elevations
    low_pass_data = low_pass_filter(rolling_std_aligned, lp_cutoff, lp_fs, lp_order)
    
    # Determine percentile threshold for classifying prolonged elevations
    std_p_lp = np.percentile(rolling_std_aligned, lp_percentile_threshold)
    is_prolonged_event = low_pass_data > std_p_lp

    # --- STEP 3: Median Filter (Spike Override Detection) ---
    # Apply 5-point median filter to isolate high-frequency transient spikes
    local_median = medfilt(data, kernel_size=5) 
    difference = data - local_median
    
    # Calculate heuristic spike threshold: p99 (high end spike limit) + p5 (noise floor offset)
    spike_threshold = np.percentile(np.abs(difference), 99) + np.percentile(np.abs(difference), 5)
    
    # Classify sharp transient bursts where rolling standard deviation exceeds spike threshold
    is_spike_override = rolling_std_aligned > spike_threshold

    # --- STEP 4: Combined Classification Logic ---
    # Active if: (Low-Pass variance is high AND raw signal > global median) OR (Spike override triggered)
    is_active = (is_prolonged_event & (data > np.median(data))) | is_spike_override

    # Create masked active signal array (NaN for normal baseline data)
    active_data = np.where(is_active, data, np.nan)
    perc_active = round(((np.sum(is_active) / len(is_active)) * 100), 2)

    # --- STEP 5: Diagnostic Plotting ---
    if plot_flag:
        fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Top Panel: Raw signal trace with active periods highlighted
        ax[0].plot(timestamps, data, color='darkorchid', label='Baseline/Normal', linewidth=0.75, alpha=0.75)
        ax[0].plot(timestamps, active_data, color='yellowgreen', label='Detected Active', linewidth=1.0)
        ax[0].set_title(f'Raw Time Series [{perc_active}% active]')
        ax[0].set_ylabel("ions/s")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc='upper right')

        # Bottom Panel: Filter components and decision thresholds
        spike_indices = timestamps[is_spike_override]
        if len(spike_indices) > 0:
            ax[1].vlines(spike_indices, ymin=0, ymax=np.max(low_pass_data), 
                        color='red', alpha=0.3, label='Spike Triggers', linewidth=0.5)

        ax[1].plot(timestamps, rolling_std_aligned, color='green', label='Rolling Std Dev', linewidth=0.75, alpha=0.4)
        ax[1].plot(timestamps, low_pass_data, color='navy', label='LP Filter', linewidth=1.2)
        ax[1].axhline(spike_threshold, linestyle="--", color='red', label='Spike Threshold (p99+p5)')

        ax[1].axhline(std_p_lp, linestyle=":", color='indianred', label='LP Threshold')
        ax[1].set_title('Filter Components & Detection Logic')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc='upper right', fontsize='small', ncol=2)
        ax[1].set_xlabel("Time")

        plt.tight_layout()
        
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.clf()
        plt.close()

    # Package results into output dictionary
    return {
        "is_active": is_active,
        "active_data": active_data,
        "rolling_std": rolling_std_aligned,
        "low_pass": low_pass_data,
        "lp_threshold": std_p_lp,
        "spike_threshold": spike_threshold,
        "metrics": {
            "sum_active": np.sum(is_active),
            "percent_active": perc_active
        }
    }


def cluster_discovered_peaks(
    peak_df,
    peak_width_coeffs=None,
    peak_width_func=None,
    eps_fwhm=0.3,
    min_spectra_fraction=0.1,
):
    """
    Cluster unassigned peaks found across multiple spectra into a consolidated peak list.

    Uses DBSCAN clustering in either a resolution-agnostic $z$-space coordinate system 
    or localized nominal mass (NM) search windows. $z$-space maps mass values such that 
    a distance of 1.0 unit corresponds to 1.0 FWHM, allowing a single distance threshold 
    (`eps_fwhm`) to apply uniformly across the entire mass spectrum.

    Parameters
    ----------
    peak_df : pandas.DataFrame
        DataFrame containing peak discovery results. Must include columns `'center_mass'` 
        and `'spectra_index'`.
    peak_width_coeffs : tuple or list of float, optional
        Calibration coefficients $(S, B)$ for linear peak width parameterization 
        $FWHM(m) = S * m + B$. Required for global $z$-space clustering (Strategy 1).
    peak_width_func : callable, optional
        Fallback function evaluating expected FWHM per nominal mass ($m/z$). Used for local 
        chunking (Strategy 2) if `peak_width_coeffs` are absent or invalid.
    eps_fwhm : float, default=0.3
        DBSCAN clustering distance threshold in FWHM units (e.g., $0.3 = 0.3 * \text{FWHM}$ radius).
    min_spectra_fraction : float, default=0.1
        Minimum fraction of total unique spectra required to form a valid DBSCAN cluster.

    Returns
    -------
    consensus_peaks : list of float
        Sorted list of consensus cluster centers (computed as the median mass per cluster).
    peak_df : pandas.DataFrame
        Updated DataFrame containing added `'NM'` (nominal mass) and `'dbscan_labels'` columns.

    Raises
    ------
    ValueError
        If both `peak_width_coeffs` and `peak_width_func` are missing or invalid.
    """
    # Guard clause: Return immediately if input peak DataFrame is empty
    if peak_df.empty:
        return [], peak_df

    # Create a explicit copy to avoid modifying the caller's original DataFrame
    peak_df = peak_df.copy()
    
    # Assign nominal mass integer labels by rounding center mass values
    peak_df["NM"] = np.round(peak_df["center_mass"]).astype(int)
    
    # Initialize all DBSCAN cluster labels to -1 (default noise label)
    peak_df["dbscan_labels"] = -1

    # Determine DBSCAN min_samples threshold based on unique spectra presence
    num_spectra = peak_df["spectra_index"].nunique()
    min_samples = max(2, int(np.round(num_spectra * min_spectra_fraction)))
    masses = peak_df["center_mass"].values

    # =========================================================================
    # STRATEGY 1: Global FWHM z-Space Clustering
    # =========================================================================
    # Transform coordinates into resolution-agnostic z-space where dz = dm / FWHM(m)
    if peak_width_coeffs is not None and len(peak_width_coeffs) >= 2:
        S, B = peak_width_coeffs[0], peak_width_coeffs[1]
        log_arg = 1.0 + (S / B) * masses

        # Ensure mathematical validity of the logarithmic argument before transforming
        if np.all(np.isfinite(log_arg)) and np.all(log_arg > 0):
            # Evaluate analytical integral: z(m) = (1/S) * ln(1 + (S/B)*m)
            z_coords = (1.0 / S) * np.log(log_arg)

            # Fit 1D DBSCAN clustering on transformed z-coordinates
            clustering = DBSCAN(eps=eps_fwhm, min_samples=min_samples).fit(
                z_coords.reshape(-1, 1)
            )
            labels = clustering.labels_
            peak_df["dbscan_labels"] = labels

            # Compute median mass center for each valid cluster (excluding noise label -1)
            consensus_peaks = [
                float(np.median(masses[labels == lbl]))
                for lbl in np.unique(labels)
                if lbl != -1
            ]
            
            # Return sorted consensus peaks and annotated DataFrame
            return sorted(consensus_peaks), peak_df

    # =========================================================================
    # STRATEGY 2: Local Nominal Mass (NM) Chunking Fallback
    # =========================================================================
    # Enforce requirement for fallback peak width function if global coefficients are missing
    if peak_width_func is None:
        raise ValueError(
            "Must provide either valid `peak_width_coeffs` (S, B) or `peak_width_func`."
        )

    consensus_peaks = []
    
    # Iterate over each nominal mass group independently
    for nm, group in peak_df.groupby("NM"):
        X = group["center_mass"].values.reshape(-1, 1)
        
        # Calculate local distance threshold in mass units for this nominal mass: eps = FWHM(NM) * eps_fwhm
        eps_local = np.abs(peak_width_func(nm)) * eps_fwhm

        # Run DBSCAN on localized nominal mass peak candidates
        clustering = DBSCAN(eps=eps_local, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        
        # Write back assigned cluster labels to original DataFrame indices
        peak_df.loc[group.index, "dbscan_labels"] = labels

        # Compute median mass per assigned cluster within this NM group
        for label_id in np.unique(labels):
            if label_id == -1:
                continue
            cluster_masses = group["center_mass"].values[labels == label_id]
            consensus_peaks.append(float(np.median(cluster_masses)))

    # Return sorted consensus peaks and annotated DataFrame
    return sorted(consensus_peaks), peak_df

def plot_apd_peaks(
    df,
    mass_axis,
    spectrum,
    nominal_mass=None,
    search_range=0.5,
    y_margin=1.2,
    seed=42,
    vline_maximum=10000,
):
    """
    Plot reference spectrum and vertical markers for automated peak discovery (APD) clusters.

    Renders an intensity spectrum overlaid with vertical lines representing peak centers 
    discovered across spectra. Colors indicate DBSCAN cluster assignments, grey lines 
    indicate unclustered noise, and shaded background spans highlight high-variance clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Peak discovery results DataFrame returned by `Deployment.automated_peak_discovery`.
    mass_axis : array-like
        Calibrated mass-to-charge ($m/z$) axis array.
    spectrum : array-like
        1D reference intensity spectrum array matching `mass_axis`.
    nominal_mass : int, float, or None, default=None
        Target nominal mass integer ($m/z$) to filter and zoom into. If ``None``, plots the full spectrum.
    search_range : float, default=0.5
        Half-width $m/z$ window range around `nominal_mass` ($NM - Δ, NM + Δ]$).
    y_margin : float, default=1.2
        Multiplier scaling the maximum visible intensity for dynamic $y$-axis headroom (e.g., $1.2 = 120 percent$).
    seed : int, default=42
        Random seed ensuring reproducible high-contrast HSV color mapping across cluster labels.
    vline_maximum : float, default=10000
        Maximum upper height limit ($y_{max}$) for vertical cluster indicator lines.
    """
    # Ensure input spectral arrays are 1D NumPy arrays
    mass_axis = np.asarray(mass_axis)
    spectrum = np.asarray(spectrum)

    plt.figure(figsize=(12, 6))

    # --- Step 1: Render Base Reference Spectrum ---
    plt.plot(
        mass_axis,
        spectrum,
        color="black",
        label="reference spectrum",
        zorder=1,
    )

    # --- Step 2: Filter Peak DataFrame to Search Range ---
    if nominal_mass is not None:
        plot_df = df[
            (df["center_mass"] >= nominal_mass - search_range)
            & (df["center_mass"] <= nominal_mass + search_range)
        ].copy()
    else:
        plot_df = df.copy()

    # --- Step 3: Generate High-Contrast Cluster Color Lookup Table ---
    # Extract unique (Nominal Mass, Cluster Label) pairs, excluding noise (-1)
    unique_clusters = plot_df[plot_df["dbscan_labels"] != -1][
        ["NM", "dbscan_labels"]
    ].drop_duplicates()

    np.random.seed(seed)
    n_clusters = len(unique_clusters)

    # Assign shuffled HSV colormap hues for distinct visual separation
    if n_clusters > 0:
        hues = np.linspace(0, 1, n_clusters, endpoint=False)
        np.random.shuffle(hues)
        colors = plt.cm.hsv(hues)
        color_lookup = {
            (row.NM, row.dbscan_labels): colors[i]
            for i, row in enumerate(unique_clusters.itertuples(index=False))
        }
    else:
        color_lookup = {}

    # --- Step 4: Render Vertical Cluster Markers and High-Variance Spans ---
    added_labels = set()
    
    # Iterate through each cluster group
    for (nm, label), group in plot_df.groupby(["NM", "dbscan_labels"]):
        if label == -1:
            # Noise points get transparent grey lines
            color = "grey"
            alpha = 0.15 if nominal_mass is None else 0.35
            legend_lbl = (
                "Noise (-1)" if "Noise (-1)" not in added_labels else None
            )
        else:
            # Retrieve cluster color from lookup table
            color = color_lookup.get((nm, label), "black")
            alpha = 0.5 if nominal_mass is None else 0.85

            # Check if cluster was flagged as high variance in feature extraction
            is_hv = (
                group["is_high_variance"].any()
                if "is_high_variance" in group.columns
                else False
            )

            hv_suffix = " [HIGH VAR]" if is_hv else ""
            lbl_key = f"NM {nm} - Cluster {label}{hv_suffix}"
            legend_lbl = lbl_key if lbl_key not in added_labels else None

            # Shade mass span for high-variance clusters with a transparent background fill
            if is_hv and len(group) > 1:
                min_m = group["center_mass"].min()
                max_m = group["center_mass"].max()
                plt.axvspan(
                    xmin=min_m,
                    xmax=max_m,
                    color=color,
                    alpha=0.2,
                    edgecolor=color,
                    linestyle=":",
                    zorder=0,
                )

        # Plot vertical dashed lines at each discovered peak center mass
        plt.vlines(
            x=group["center_mass"].values,
            ymin=-1,
            ymax=vline_maximum,
            color=color,
            linestyle="--",
            alpha=alpha,
            label=legend_lbl,
            zorder=2,
        )

        if legend_lbl:
            added_labels.add(legend_lbl)

    # --- Step 5: Formatting, Legend Deduplication, and Dynamic Axis Limits ---
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.title(
        "Automated Peak Discovery using averaged dataset and DBSCAN clustering"
    )
    plt.xlabel("m/z")
    plt.ylabel("ions/s")

    if nominal_mass is not None:
        # Scale view tightly to requested nominal mass window
        min_x = nominal_mass - search_range
        max_x = nominal_mass + search_range

        mask = (mass_axis >= min_x) & (mass_axis <= max_x)
        window_signal = spectrum[mask]

        plt.xlim(min_x, max_x)

        # Dynamically set y-axis headroom based on local window intensity max
        if len(window_signal) > 0 and np.max(window_signal) > 0:
            max_y = np.max(window_signal) * y_margin
            plt.ylim(0, max_y)

        plt.legend(loc="upper right")
    else:
        # Cap full-spectrum legend entries to avoid cluttering the plot canvas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            list(by_label.values())[:10],
            list(by_label.keys())[:10],
            loc="upper right",
        )

    plt.tight_layout()
    plt.show()

def calculate_mass_deviation(experimental_mass: float, formula: str):
    """
    Compute the theoretical exact mass of a chemical formula and its PPM deviation.

    Calculates the exact theoretical mass for a given chemical formula using 
    :func:`return_mass` and evaluates the relative positional mass error in parts 
    per million (PPM) against an experimental mass reading:

    .. math::

        \\text{PPM} = \\frac{m_{\\text{exp}} - m_{\\text{theo}}}{m_{\\text{theo}}} \\times 10^6

    Parameters
    ----------
    experimental_mass : float
        Observed experimental mass-to-charge ($m/z$) reading.
    formula : str
        Chemical formula string or ion identifier (e.g., ``"C6H6+"`` or ``"IH2O-"``).

    Returns
    -------
    theoretical_mass : float or None
        Calculated theoretical exact mass in $m/z$ units, or ``None`` if evaluation fails.
    ppm_deviation : float or None
        Positional mass deviation in PPM, or ``None`` if evaluation fails or theoretical mass is 0.
    """
    try:
        # Calculate theoretical exact mass from chemical formula string
        theoretical_mass = return_mass(formula)
        
        # Guard clause: Return None if theoretical mass could not be evaluated or equals zero
        if theoretical_mass is None or theoretical_mass == 0:
            return None, None
            
        # Calculate relative positional mass deviation in parts per million (PPM)
        ppm_deviation = ((experimental_mass - theoretical_mass) / theoretical_mass) * 1e6
        
        return theoretical_mass, ppm_deviation
        
    except Exception:
        # Fail gracefully on unparseable formula strings or calculation exceptions
        return None, None


def find_possible_compositions(peak_mass, tolerance=0.01, formula_db=None, mass_array=None, verbose=True):
    """
    Search a formula database for candidate chemical compositions matching a peak mass.

    Executes a fast $O(\\log N)$ binary search across a pre-sorted exact mass array 
    to isolate candidate chemical formulas within a specified $m/z$ tolerance window. 
    Calculates PPM error for each candidate and returns a DataFrame sorted by absolute PPM error.

    Parameters
    ----------
    peak_mass : float
        Target experimental peak mass ($m/z$) to search against the database.
    tolerance : float, default=0.01
        Mass tolerance window in $m/z$ units ($[\\text{peak\\_mass} - \\text{tolerance}, \\text{peak\\_mass} + \\text{tolerance}]$).
    formula_db : pandas.DataFrame or None, default=None
        Database DataFrame containing an `'ExactMass'` column. Defaults to `Deployment.peak_assignment_db`.
    mass_array : numpy.ndarray or None, default=None
        Pre-sorted 1D array of exact masses corresponding to `formula_db["ExactMass"]`.
    verbose : bool, default=True
        If ``True``, prints a diagnostic warning message if no formula candidates match.

    Returns
    -------
    matches : pandas.DataFrame
        DataFrame of candidate chemical compositions within the tolerance window, 
        annotated with a `'ppm_error'` column and sorted by absolute PPM deviation.

    Raises
    ------
    ValueError
        If `formula_db` or `mass_array` is not supplied.
    """
    # Guard clause: Enforce explicit database initialization
    if formula_db is None or mass_array is None:
        raise ValueError("formula_db or mass_array not initialized. Please pass these manually!")

    # Calculate lower and upper m/z search boundaries
    lower = peak_mass - tolerance
    upper = peak_mass + tolerance
    
    # Binary search (searchsorted) to find bounding array indices in O(log N) time
    start_idx = np.searchsorted(mass_array, lower, side='left')
    end_idx = np.searchsorted(mass_array, upper, side='right')
    
    # Slice candidate subset from pre-sorted formula database
    matches = formula_db.iloc[start_idx:end_idx].copy()
    
    # Handle empty match case
    if matches.empty:
        if verbose:
            print("No compositions found that match entries in formula_db (which is usually Deployment.peak_assignment_db). "
                  "Adjust 'tolerance' for how far out to search? Or ensure entry in formula_db?")
        return pd.DataFrame()  # Return empty DataFrame for clean downstream handling

    # Calculate PPM mass error for each matching candidate formula
    matches["ppm_error"] = ((matches["ExactMass"] - peak_mass) / matches["ExactMass"]) * 1e6
    
    # Sort matching candidate DataFrame by absolute PPM deviation (closest mass match first)
    matches = matches.sort_values("ppm_error", key=abs)
    
    return matches

def aggregate_tw_expicit_time_export(dir_path):
    """
    Aggregate Tofware explicit time export CSV files into a unified pandas DataFrame.

    Parses a target directory containing exported Tofware time-series CSV files, 
    extracts timestamps, applies clean compound header labels to full data arrays, 
    appends max error columns with `_error` suffixes, combines user data columns, 
    and concatenates the datasets horizontally into a single cohesive DataFrame.

    Parameters
    ----------
    dir_path : str or pathlib.Path
        Path to the directory containing Tofware explicit time export CSV files.

    Returns
    -------
    aggregated_df : pandas.DataFrame
        Horizontally aggregated DataFrame containing aligned timestamp, full data, 
        error metrics, and user-defined metadata columns.
    """
    # Glob and sort all exported CSV file paths within target directory
    file_list = sorted(glob.glob(f"{dir_path}/*"))

    # Load timestamps file (typically last file in sorted list) and parse as datetime64[ns]
    df_final = pd.read_csv(file_list[-1], header=None, names=['timestamp'])
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp']).astype('datetime64[ns]')
    
    # Load clean compound column labels (penultimate file in sorted list)
    clean_labels = pd.read_csv(file_list[-2], header=None)[0].astype(str).values
    
    # Load full dataset CSV (file index 1) and apply clean compound column headers
    full_data = pd.read_csv(file_list[1])
    if len(clean_labels) == full_data.shape[1]:
        full_data.columns = clean_labels
        
    # Load maximum error dataset CSV (file index 2)
    max_err = pd.read_csv(file_list[2], header=None)
    
    # Drop first header row (NaNs) and reset row index to align with full data rows
    max_err = max_err.iloc[1:].reset_index(drop=True)
    
    # Generate matching error column labels suffixed with "_error"
    error_labels = [f"{label}_error" for label in clean_labels]
    
    # Apply error column labels if dimension lengths match
    if len(error_labels) == max_err.shape[1]:
        max_err.columns = error_labels
        
    # Load auxiliary user metadata dataset CSV (file index 3)
    user_data = pd.read_csv(file_list[3])
    
    # Concatenate all parsed sub-DataFrames horizontally along axis 1
    aggregated_df = pd.concat([df_final, full_data, max_err, user_data], axis=1)
    
    return aggregated_df

def regr_model_metrics(actual, pred):
    """
    Calculate comprehensive statistical evaluation metrics for regression models.

    Computes Root Mean Squared Error (RMSE), Relative RMSE (RRMSE in %), Mean Absolute 
    Error (MAE), Coefficient of Determination ($R^2$), $D^2$ absolute error score, 
    Maximum Absolute Error (MAXErr), and Explained Variance Score (EVS).

    Parameters
    ----------
    actual : array-like
        1D array of observed ground truth target values.
    pred : array-like
        1D array of model predicted values.

    Returns
    -------
    RMSE : float
        Root Mean Squared Error.
    RRMSE : float
        Relative Root Mean Squared Error expressed as a percentage ($100 \\times \\text{RMSE} / \\bar{y}$).
    MAE : float
        Mean Absolute Error.
    R2 : float
        $R^2$ regression score coefficient of determination.
    D2 : float
        $D^2$ absolute error regression score.
    MAXErr : float
        Maximum residual error value.
    EVS : float
        Explained Variance Score.
    """
    # Calculate Root Mean Squared Error (RMSE)
    RMSE = root_mean_squared_error(actual, pred)
    
    # Calculate actual target mean to evaluate Relative RMSE (RRMSE %)
    actual_mean = np.mean(actual)
    RRMSE = 100.0 * RMSE / actual_mean
    
    # Calculate Mean Absolute Error (MAE)
    MAE = mean_absolute_error(actual, pred)
    
    # Calculate R^2 Score (Coefficient of Determination)
    R2 = r2_score(actual, pred)
    
    # Calculate D^2 Absolute Error Score
    D2 = d2_absolute_error_score(actual, pred)
    
    # Calculate Maximum Absolute Error
    MAXErr = max_error(actual, pred)
    
    # Calculate Explained Variance Score (EVS)
    EVS = explained_variance_score(actual, pred)
    
    return RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS

def regr_scatter_plot(actual, pred, color='blue', title='scatter plot',  
                      x_label='x_label', y_label='y_label',
                      cbar_label='cbar_label', cm='Blues',
                      output_dir=None, plot_filename=None, log_flag=False,
                      show_plot_flag=True):
    """
    Generate a regression evaluation scatter plot with a 1:1 identity line and metrics annotation.

    Plots observed versus predicted values, overlays a dashed red $1:1$ identity reference 
    line, calculates statistical performance metrics via :func:`regr_model_metrics`, 
    annotates $R^2$, $D^2$, MAE, and RMSE on the figure canvas, and supports optional log-log 
    scaling and colormap encoding.

    Parameters
    ----------
    actual : numpy.ndarray or pandas.Series
        1D array of actual observed ground truth values.
    pred : numpy.ndarray or pandas.Series
        1D array of model predicted values.
    color : str or numpy.ndarray, default='blue'
        Single Matplotlib color string or 1D array of numerical values mapped to colormap `cm`.
    title : str, default='scatter plot'
        Figure title string.
    x_label : str, default='x_label'
        Label for the x-axis (Actual).
    y_label : str, default='y_label'
        Label for the y-axis (Predicted).
    cbar_label : str, default='cbar_label'
        Colorbar label string used when `color` is a numerical array.
    cm : str, default='Blues'
        Matplotlib colormap name used when `color` is a numerical array.
    output_dir : str or None, default=None
        Directory path where the output image file will be saved. Defaults to OpenTof default plot dir if ``None``.
    plot_filename : str or None, default=None
        Filename for the saved plot image.
    log_flag : bool, default=False
        If ``True``, renders both axes in logarithmic scale ($log_{10}$) with log major/minor locators.
    show_plot_flag : bool, default=True
        If ``True``, displays the figure interactively.
    """
    # Calculate regression model evaluation metrics
    RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS = regr_model_metrics(actual, pred)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot scatter points: single color vs. continuous colormap mapping
    if isinstance(color, str):
        ax.scatter(actual, pred, edgecolors=(0, 0, 0), c=color, alpha=0.7)
    else:
        # If 'color' is a numerical array, map values to colormap 'cm' and attach a colorbar
        sc = ax.scatter(actual, pred, edgecolors=(0, 0, 0), c=color, cmap=cm, alpha=0.7)
        plt.colorbar(sc, label=f'{cbar_label}')

    # Draw 1:1 identity reference line spanning min to max range
    limits = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(limits, limits, 'r--', lw=2, label='Identity Line')
    
    # Format statistical metric annotation text block
    text = (f"R² = {R2:.2f}\n"
            f"D² = {D2:.2f}\n"
            f"MAE = {MAE:.2f}\n"
            f"RMSE = {RMSE:.2f}")
    
    # Annotate summary metrics box in upper left quadrant
    ax.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Assign dynamic axis labels
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')

    # Apply logarithmic scaling or linear grid formatting
    if log_flag:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Configure major and minor log tick locators
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto"))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto"))

        # Render log gridlines
        ax.grid(True, which="major", linestyle="--", linewidth=0.5)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3)
    else:
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    
    # Resolve default output directory if unassigned
    if output_dir is None:
        output_dir = get_default_plot_dir()
        
    # Save figure image if filename is provided
    if plot_filename:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')

    # Render plot interactively if requested
    if show_plot_flag:
        plt.show()
    plt.close()

def ensure_dir(path: str):
    """
    Create a directory on disk if it does not already exist.

    Uses :func:`os.makedirs` with `exist_ok=True` to safely handle directory 
    creation without raising race-condition errors if the path exists.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path string to create.
    """
    # Create target directory and intermediate parent folders safely
    os.makedirs(path, exist_ok=True)

# ------------------------------------------------------------
#  Element and Periodic Table classes 
# 
# The "return_mass function is located below these classes"
# ------------------------------------------------------------
class Element:
    """Represents a single chemical element with isotopes."""
    def __init__(self, symbol, isotopes):
        """
        isotopes: list of dicts like:
        [{"mass": 12.00000, "relative_intensity": 0.9893}, ...]
        """
        self.symbol = symbol
        self.isotopes = isotopes

        # monoisotopic = the isotope with the highest abundance
        self.monoisotopic_mass = max(isotopes, key=lambda x: x["relative_intensity"])["mass"]

    @property
    def isotopes_df(self):
        return pd.DataFrame(self.isotopes)
    
    @property
    def print(self):
        print(f"Symbol: {self.symbol}")
        print(f"Monoisotopic mass: {self.monoisotopic_mass}")
        print(f"Isotopes: {self.isotopes}")
        print(f"Isotopes (pd.Dataframe):\n{self.isotopes_df}")
        return None

# probs = [a / sum(all_abundances) for a in abundances]
class PeriodicTable:
    """
    Accessible periodic table instance used as ot.ptoe.

    Currently this mimic's the information in Tofware. Abundances are 
    relative to the monoisotopic mass.
    """
    def __init__(self):
        self._elements = {}

        self._register("H", [
            {"mass": 1.00782503223, "relative_intensity": 100},
            {"mass": 2.01410177812, "relative_intensity": 0.01150132265210499},
        ])

        # # Mass lost from an electron
        # self._register("+", [
        #     {"mass": -0.0005486, "relative_intensity": 100}
        # ])

        # # Mass gained from an electron
        # self._register("-", [
        #     {"mass": 0.0005486, "relative_intensity": 100}
        # ])

        self._register("He", [
            {"mass": 4.00260325413, "relative_intensity": 100},
            {"mass": 3.0160293201, "relative_intensity": 0.0001340001795602406},
        ])

        self._register("Li", [
            {"mass": 7.0160034366, "relative_intensity": 100},
            {"mass": 6.0151228874, "relative_intensity": 8.213396818526133},
        ])

        self._register("Be", [
            {"mass": 9.012183065, "relative_intensity": 100},
        ])

        self._register("B", [
            {"mass": 11.00930536, "relative_intensity": 100},
            {"mass": 10.01293695, "relative_intensity": 24.84394506866417},
        ])

        self._register("C", [
            {"mass": 12.0, "relative_intensity": 100},
            {"mass": 13.0033548351, "relative_intensity": 1.081572829273223},
        ])

        self._register("N", [
            {"mass": 14.0030740044, "relative_intensity": 100},
            {"mass": 15.0001088989, "relative_intensity": 0.3653298004737243},
        ])

        self._register("O", [
            {"mass": 15.9949146196, "relative_intensity": 100},
            {"mass": 17.9991596129,  "relative_intensity": 0.2054993634531912},
            {"mass": 16.9991317565,  "relative_intensity": 0.03809256493278667},
        ])

        self._register("F", [
            {"mass": 18.9984031627, "relative_intensity": 100},
        ])

        self._register("Ne", [
            {"mass": 19.9924401762, "relative_intensity": 100},
            {"mass": 21.991385114,  "relative_intensity": 10.22325375773652},
            {"mass": 20.993846685,  "relative_intensity": 0.2984084880636605},
        ])

        self._register("Na", [
            {"mass": 22.989769282, "relative_intensity": 100},
        ])

        self._register("Mg", [
            {"mass": 23.985041697, "relative_intensity": 100},
            {"mass": 25.982592968,  "relative_intensity": 13.93847322445879},
            {"mass": 24.985836976,  "relative_intensity": 12.6598303582732},
        ])

        self._register("Al", [
            {"mass": 26.98153853, "relative_intensity": 100},
        ])

        self._register("Si", [
            {"mass": 27.9769265347, "relative_intensity": 100},
            {"mass": 28.9764946649,  "relative_intensity": 5.080077637899439},
            {"mass": 29.973770136,  "relative_intensity": 3.352742808193184},
        ])

        self._register("P", [
            {"mass": 30.9737619984, "relative_intensity": 100},
        ])

        self._register("S", [
            {"mass": 31.9720711744, "relative_intensity": 100},
            {"mass": 33.967867004,  "relative_intensity": 4.474155174228867},
            {"mass": 32.9714589098,  "relative_intensity": 0.7895567954521528},
            {"mass": 35.96708071,  "relative_intensity": 0.01052742393936204},
        ])

        self._register("Cl", [
            {"mass": 34.968852682, "relative_intensity": 100},
            {"mass": 36.965902602,  "relative_intensity": 31.99577613516367},
        ])

        self._register("Ar", [
            {"mass": 39.9623831237, "relative_intensity": 100},
            {"mass": 35.967545105,  "relative_intensity": 0.3349279894782814},
            {"mass": 37.96273211,  "relative_intensity": 0.06315039130151048},
        ])

        self._register("K", [
            {"mass": 38.9637064864, "relative_intensity": 100},
            {"mass": 40.9618252579,  "relative_intensity": 7.216745784012327},
            {"mass": 39.963998166,  "relative_intensity": 0.01254582711850231},
        ])

        self._register("Ca", [
            {"mass": 39.962590863, "relative_intensity": 100},
            {"mass": 43.95548156,  "relative_intensity": 2.151824305505411},
            {"mass": 41.95861783,  "relative_intensity": 0.6674162635004797},
            {"mass": 47.95252276,  "relative_intensity": 0.1929008365913288},
            {"mass": 42.95876644,  "relative_intensity": 0.1392599622450769},
            {"mass": 45.953689,  "relative_intensity": 0.004126221103557834},
        ])

        self._register("Sc", [
            {"mass": 44.95590828, "relative_intensity": 100},
        ])

        self._register("Ti", [
            {"mass": 47.94794198, "relative_intensity": 100},
            {"mass": 45.95262772,  "relative_intensity": 11.19099294628323},
            {"mass": 46.95175879,  "relative_intensity": 10.09224091155725},
            {"mass": 48.94786568,  "relative_intensity": 7.338578404774824},
            {"mass": 49.94478689,  "relative_intensity": 7.026587086272382},
        ])

        self._register("V", [
            {"mass": 50.94395704, "relative_intensity": 100},
            {"mass": 49.94715601,  "relative_intensity": 0.2506265664160401},
        ])

        self._register("Cr", [
            {"mass": 51.94050623, "relative_intensity": 100},
            {"mass": 52.94064815,  "relative_intensity": 11.3391972693313},
            {"mass": 49.94604183,  "relative_intensity": 5.185644893721132},
            {"mass": 53.93887916,  "relative_intensity": 2.822566207974794},
        ])

        self._register("Mn", [
            {"mass": 54.93804391, "relative_intensity": 100},
        ])

        self._register("Fe", [
            {"mass": 55.93493633, "relative_intensity": 100},
            {"mass": 53.93960899, "relative_intensity": 6.370294483074307},
            {"mass": 56.93539284, "relative_intensity": 2.309436100878436},
            {"mass": 57.93327443, "relative_intensity": 0.307343549055082},
        ])

        self._register("Co", [
            {"mass": 58.93319429, "relative_intensity": 100},
        ])

        self._register("Ni", [
            {"mass": 57.93534241, "relative_intensity": 100},
            {"mass": 59.93078588, "relative_intensity": 38.51961749195764},
            {"mass": 61.92834537, "relative_intensity": 5.338954419260542},
            {"mass": 60.93105557, "relative_intensity": 1.674427486522614},
            {"mass": 63.92796682, "relative_intensity": 1.359489989276848},
        ])

        self._register("Cu", [
            {"mass": 62.92959772, "relative_intensity": 100},
            {"mass": 64.92778970000001,  "relative_intensity": 44.61315979754157},
        ])

        self._register("Zn", [
            {"mass": 63.92914201, "relative_intensity": 100},
            {"mass": 65.92603381000001, "relative_intensity": 56.39617653040471},
            {"mass": 67.92484455, "relative_intensity": 37.52287980475899},
            {"mass": 66.92712775, "relative_intensity": 8.216392109009558},
            {"mass": 69.9253192, "relative_intensity": 1.240593858043523},
        ])

        self._register("Ga", [
            {"mass": 68.9255735, "relative_intensity": 100},
            {"mass": 70.92470258,  "relative_intensity": 66.36720569641314},
        ])

        self._register("Ge", [
            {"mass": 73.921177761, "relative_intensity": 100},
            {"mass": 71.922075826, "relative_intensity": 75.2054794520548},
            {"mass": 69.92424875, "relative_intensity": 56.35616438356164},
            {"mass": 72.923458956, "relative_intensity": 21.23287671232877},
            {"mass": 75.921402726, "relative_intensity": 21.17808219178082},
        ])

        self._register("As", [
            {"mass": 74.92159457, "relative_intensity": 100},
        ])

        self._register("Se", [
            {"mass": 79.9165218, "relative_intensity": 100},
            {"mass": 77.91730928, "relative_intensity": 47.91372707115501},
            {"mass": 75.919213704, "relative_intensity": 18.887321104616},
            {"mass": 81.91669949999999, "relative_intensity": 17.59725861721427},
            {"mass": 76.919914154, "relative_intensity": 15.37996371699254},
            {"mass": 73.922475934, "relative_intensity": 1.793993146543036},
        ])

        self._register("Br", [
            {"mass": 78.9183376, "relative_intensity": 100},
            {"mass": 80.91628969999999,  "relative_intensity": 97.27756954034326},
        ])

        self._register("Kr", [
            {"mass": 83.9114977282, "relative_intensity": 100},
            {"mass": 85.9106106269, "relative_intensity": 30.32095039219471},
            {"mass": 81.91348273, "relative_intensity": 20.34323617667188},
            {"mass": 82.91412716000001, "relative_intensity": 20.1800410619966},
            {"mass": 79.91637808, "relative_intensity": 4.011441205889063},
            {"mass": 77.92036494, "relative_intensity": 0.6229490936529384},
        ])

        self._register("Rb", [
            {"mass": 84.91178973789999, "relative_intensity": 100},
            {"mass": 86.909180531,  "relative_intensity": 38.56172925038104},
        ])
        
        self._register("Sr", [
            {"mass": 87.9056125, "relative_intensity": 100},
            {"mass": 85.9092606,  "relative_intensity": 11.93993703075805},
            {"mass": 86.9088775,  "relative_intensity": 8.476628723661905},
            {"mass": 83.9134191,  "relative_intensity": 0.6781302978929524},
        ])

        self._register("Y", [
            {"mass": 88.90584029999999, "relative_intensity": 100},
        ])
              
        self._register("Zr", [
            {"mass": 89.9046977, "relative_intensity": 100},
            {"mass": 93.9063108,  "relative_intensity": 33.78036929057336},
            {"mass": 91.9050347,  "relative_intensity": 33.33333333333332},
            {"mass": 90.9056396,  "relative_intensity": 21.80758017492711},
            {"mass": 95.9082714,  "relative_intensity": 5.442176870748297},
        ])

        self._register("Nb", [
            {"mass": 92.906373, "relative_intensity": 100},
        ])
              
        self._register("Mo", [
            {"mass": 97.90540482, "relative_intensity": 100},
            {"mass": 95.90467612,  "relative_intensity": 68.34768347683477},
            {"mass": 94.90583877,  "relative_intensity": 64.94464944649447},
            {"mass": 91.90680795999999,  "relative_intensity": 59.57359573595735},
            {"mass": 99.9074718,  "relative_intensity": 40.26240262402624},
            {"mass": 96.90601812,  "relative_intensity": 39.36039360393604},
            {"mass": 93.90508490000001,  "relative_intensity": 37.51537515375154},
        ])           
              
        self._register("Tc", [
            {"mass": 96.90636670000001, "relative_intensity": 100},
            {"mass": 97.90721240000001,  "relative_intensity": 100},
            {"mass": 98.9062508,  "relative_intensity": 100},
        ])  
              
        self._register("Ru", [
            {"mass": 101.9043441, "relative_intensity": 100},
            {"mass": 103.9054275,  "relative_intensity": 59.01743264659271},
            {"mass": 100.9055769,  "relative_intensity": 54.0729001584786},
            {"mass": 98.9059341,  "relative_intensity": 40.44374009508716},
            {"mass": 99.90421430000001,  "relative_intensity": 39.93660855784469},
            {"mass": 95.90759025,  "relative_intensity": 17.5594294770206},
            {"mass": 97.9052868,  "relative_intensity": 5.927099841521395},
        ])           
              
        self._register("Rh", [
            {"mass": 102.905498, "relative_intensity": 100},
        ])              
              
        self._register("Pd", [
            {"mass": 105.9034804, "relative_intensity": 100},
            {"mass": 107.9038916,  "relative_intensity": 96.81668496158069},
            {"mass": 104.9050796,  "relative_intensity": 81.70508598609587},
            {"mass": 109.9051722,  "relative_intensity": 42.8832784485913},
            {"mass": 103.9040305,  "relative_intensity": 40.76106842297842},
            {"mass": 101.9056022,  "relative_intensity": 3.732162458836444},
        ])           

        self._register("Ag", [
            {"mass": 106.9050916, "relative_intensity": 100},
            {"mass": 108.9047553, "relative_intensity": 92.90495572831266},
        ])   
              
        self._register("Cd", [
            {"mass": 113.90336509, "relative_intensity": 100},
            {"mass": 111.90276287,  "relative_intensity": 83.98886181691611},
            {"mass": 110.90418287,  "relative_intensity": 44.55273233553777},
            {"mass": 109.90300661,  "relative_intensity": 43.47372084928647},
            {"mass": 112.90440813,  "relative_intensity": 42.53393665158372},
            {"mass": 115.90476315,  "relative_intensity": 26.07030978071703},
            {"mass": 105.9064599,  "relative_intensity": 4.35085276714236},
            {"mass": 107.9041834,  "relative_intensity": 3.09780717020536},
        ])           

        self._register("In", [
            {"mass": 114.903878776, "relative_intensity": 100},
            {"mass": 112.90406184, "relative_intensity": 4.48229025180232},
        ])   
              
        self._register("Sn", [
            {"mass": 119.90220163, "relative_intensity": 100},
            {"mass": 117.90160657,  "relative_intensity": 74.34008594229589},
            {"mass": 115.9017428,  "relative_intensity": 44.62860650705954},
            {"mass": 118.90331117,  "relative_intensity": 26.365868631062},
            {"mass": 116.90295398,  "relative_intensity": 23.57274401473296},
            {"mass": 123.9052766,  "relative_intensity": 17.77163904235728},
            {"mass": 121.9034438,  "relative_intensity": 14.21117249846532},
            {"mass": 111.90482387,  "relative_intensity": 2.977286678944138},
            {"mass": 113.9027827,  "relative_intensity": 2.025782688766114},
            {"mass": 114.903344699,  "relative_intensity": 1.043585021485574},
        ])           

        self._register("Sb", [
            {"mass": 120.903812, "relative_intensity": 100},
            {"mass": 122.9042132, "relative_intensity": 74.79461632581716},
        ])   
              
        self._register("Te", [
            {"mass": 129.906222748, "relative_intensity": 100},
            {"mass": 127.90446128,  "relative_intensity": 93.13380281690141},
            {"mass": 125.9033109,  "relative_intensity": 55.28169014084507},
            {"mass": 124.9044299,  "relative_intensity": 20.74530516431925},
            {"mass": 123.9028171,  "relative_intensity": 13.90845070422535},
            {"mass": 121.9030435,  "relative_intensity": 7.482394366197182},
            {"mass": 122.9042698,  "relative_intensity": 2.61150234741784},
            {"mass": 119.9040593,  "relative_intensity": 0.2640845070422535},
        ])           

        self._register("I", [
            {"mass": 126.9044719, "relative_intensity": 100},
        ])   
              
        self._register("Xe", [
            {"mass": 131.904155086, "relative_intensity": 100},
            {"mass": 128.904780861,  "relative_intensity": 98.11212772124898},
            {"mass": 130.90508406,  "relative_intensity": 78.90562868376654},
            {"mass": 133.90539466,  "relative_intensity": 38.78202507748453},
            {"mass": 135.907214484,  "relative_intensity": 32.91624239090849},
            {"mass": 129.903509349,  "relative_intensity": 15.12899221810128},
            {"mass": 127.903531,  "relative_intensity": 7.098845722185473},
            {"mass": 123.905892,  "relative_intensity": 0.3537902380651539},
            {"mass": 125.9042983,  "relative_intensity": 0.3307492771827594},
        ])           

        self._register("Cs", [
            {"mass": 132.905451961, "relative_intensity": 100},
        ])   
              
        self._register("Ba", [
            {"mass": 137.905247, "relative_intensity": 100},
            {"mass": 136.90582714,  "relative_intensity": 15.66570894585623},
            {"mass": 135.90457573,  "relative_intensity": 10.95428045412703},
            {"mass": 134.90568838,  "relative_intensity": 9.194119780189125},
            {"mass": 133.90450818,  "relative_intensity": 3.371084270133057},
            {"mass": 129.9063207,  "relative_intensity": 0.1478423386984295},
            {"mass": 131.9050611,  "relative_intensity": 0.1408686434768055},
        ])           

        self._register("La", [
            {"mass": 138.9063563, "relative_intensity": 100},
            {"mass": 137.9071149, "relative_intensity": 0.08888894226962965},
        ])   

        self._register("Ce", [
            {"mass": 139.9054431, "relative_intensity": 100},
            {"mass": 141.9092504, "relative_intensity": 12.56529112492934},
            {"mass": 137.905991, "relative_intensity": 0.2837761447145279},
            {"mass": 135.90712921, "relative_intensity": 0.2091577162238553},
        ])                                   

        self._register("Pr", [
            {"mass": 140.9076576, "relative_intensity": 100},
        ])   

        self._register("Nd", [
            {"mass": 141.907729, "relative_intensity": 100},
            {"mass": 143.910093,  "relative_intensity": 87.64731879787858},
            {"mass": 145.9131226,  "relative_intensity": 63.30657041838538},
            {"mass": 142.90982,  "relative_intensity": 44.83647613435473},
            {"mass": 144.9125793,  "relative_intensity": 30.54286977018267},
            {"mass": 147.9168993,  "relative_intensity": 21.19917501473187},
            {"mass": 149.9209022,  "relative_intensity": 20.76458456098998},
        ])           

        self._register("Pm", [
            {"mass": 144.9127559, "relative_intensity": 100},
            {"mass": 146.915145, "relative_intensity": 100},
        ])   

        self._register("Sm", [
            {"mass": 151.9197397, "relative_intensity": 100},
            {"mass": 153.9222169,  "relative_intensity": 85.04672897196261},
            {"mass": 146.9149044,  "relative_intensity": 56.03738317757009},
            {"mass": 148.9171921,  "relative_intensity": 51.66355140186916},
            {"mass": 147.9148292,  "relative_intensity": 42.01869158878505},
            {"mass": 149.9172829,  "relative_intensity": 27.58878504672897},
            {"mass": 143.9120065,  "relative_intensity": 11.47663551401869},
        ])           

        self._register("Eu", [
            {"mass": 152.921238, "relative_intensity": 100},
            {"mass": 150.9198578, "relative_intensity": 91.60758766047135},
        ])   

        self._register("Gd", [
            {"mass": 157.9241123, "relative_intensity": 100},
            {"mass": 159.9270624,  "relative_intensity": 88.00322061191626},
            {"mass": 155.9221312,  "relative_intensity": 82.4074074074074},
            {"mass": 156.9239686,  "relative_intensity": 63.00322061191627},
            {"mass": 154.9226305,  "relative_intensity": 59.58132045088568},
            {"mass": 153.9208741,  "relative_intensity": 8.776167471819647},
            {"mass": 151.9197995,  "relative_intensity": 0.8051529790660226},
        ])           

        self._register("Tb", [
            {"mass": 158.9253547, "relative_intensity": 100},
        ])   

        self._register("Dy", [
            {"mass": 163.9291819, "relative_intensity": 100},
            {"mass": 161.9268056,  "relative_intensity": 90.14508138711962},
            {"mass": 162.9287383,  "relative_intensity": 88.09624911535739},
            {"mass": 160.9269405,  "relative_intensity": 66.84005661712668},
            {"mass": 159.9252046,  "relative_intensity": 8.241330502477},
            {"mass": 157.9244159,  "relative_intensity": 0.3361641896673744},
            {"mass": 155.9242847,  "relative_intensity": 0.1981599433828733},
        ])           

        self._register("Ho", [
            {"mass": 164.9303288, "relative_intensity": 100},
        ])   

        self._register("Er", [
            {"mass": 165.9302995, "relative_intensity": 100},
            {"mass": 167.9323767,  "relative_intensity": 80.5241321672686},
            {"mass": 166.9320546,  "relative_intensity": 68.25955884547653},
            {"mass": 169.9354702,  "relative_intensity": 44.50347730054025},
            {"mass": 163.9292088,  "relative_intensity": 4.778676536429574},
            {"mass": 161.9287884,  "relative_intensity": 0.4148882189654658},
        ])           

        self._register("Tm", [
            {"mass": 168.9342179, "relative_intensity": 100},
        ])   

        self._register("Yb", [
            {"mass": 173.9388664, "relative_intensity": 100},
            {"mass": 171.9363859,  "relative_intensity": 67.69499781427589},
            {"mass": 172.9382151,  "relative_intensity": 50.28102166989321},
            {"mass": 170.9363302,  "relative_intensity": 43.9955036532817},
            {"mass": 175.9425764,  "relative_intensity": 40.57952913257978},
            {"mass": 169.9347664,  "relative_intensity": 9.31118466246175},
            {"mass": 167.9338896,  "relative_intensity": 0.384062948854056},
        ])           

        self._register("Lu", [
            {"mass": 174.9407752, "relative_intensity": 100},
            {"mass": 175.9426897, "relative_intensity": 2.66835042761368},
        ])   

        self._register("Hf", [
            {"mass": 179.946557, "relative_intensity": 100},
            {"mass": 177.9437058,  "relative_intensity": 77.76510832383126},
            {"mass": 176.9432277,  "relative_intensity": 53.02166476624858},
            {"mass": 178.9458232,  "relative_intensity": 38.82554161915621},
            {"mass": 175.9414076,  "relative_intensity": 14.99429874572406},
            {"mass": 173.9400461,  "relative_intensity": 0.4561003420752566},
        ])           

        self._register("Ta", [
            {"mass": 180.9479958, "relative_intensity": 100},
            {"mass": 179.9474648, "relative_intensity": 0.01201144257425317},
        ])   

        self._register("W", [
            {"mass": 183.95093092, "relative_intensity": 100},
            {"mass": 185.9543628,  "relative_intensity": 92.78720626631853},
            {"mass": 181.94820394,  "relative_intensity": 86.48825065274151},
            {"mass": 182.95022275,  "relative_intensity": 46.70365535248042},
            {"mass": 179.9467108,  "relative_intensity": 0.3916449086161879},
        ])           

        self._register("Re", [
            {"mass": 186.9557501, "relative_intensity": 100},
            {"mass": 184.9529545, "relative_intensity": 59.7444089456869},
        ])   

        self._register("Os", [
            {"mass": 191.961477, "relative_intensity": 100},
            {"mass": 189.9584437,  "relative_intensity": 64.39431093673369},
            {"mass": 188.9581442,  "relative_intensity": 39.60274644433545},
            {"mass": 187.9558352,  "relative_intensity": 32.46689553702796},
            {"mass": 186.9557474,  "relative_intensity": 4.806277587052477},
            {"mass": 185.953835,  "relative_intensity": 3.898970083374203},
            {"mass": 183.9524885,  "relative_intensity": 0.04904364884747425},
        ])           

        self._register("Ir", [
            {"mass": 192.9629216, "relative_intensity": 100},
            {"mass": 190.9605893,  "relative_intensity": 59.48963317384369},
        ])           

        self._register("Pt", [
            {"mass": 194.9647917, "relative_intensity": 100},
            {"mass": 193.9626809,  "relative_intensity": 97.27649496743634},
            {"mass": 195.96495209,  "relative_intensity": 74.62995855535819},
            {"mass": 197.9678949,  "relative_intensity": 21.77619893428064},
            {"mass": 191.9610387,  "relative_intensity": 2.3149792776791},
            {"mass": 189.9599297,  "relative_intensity": 0.03552397868561278},
        ])           

        self._register("Au", [
            {"mass": 196.96656879, "relative_intensity": 100},
        ])           

        self._register("Hg", [
            {"mass": 201.9706434, "relative_intensity": 100},
            {"mass": 199.96832659,  "relative_intensity": 77.36101808439385},
            {"mass": 198.96828064,  "relative_intensity": 56.49698593436036},
            {"mass": 200.97030284,  "relative_intensity": 44.13931681178835},
            {"mass": 197.9667686,  "relative_intensity": 33.38914936369726},
            {"mass": 203.97349398,  "relative_intensity": 23.00736771600804},
            {"mass": 195.9658326,  "relative_intensity": 0.5023442732752847},
        ])           

        self._register("Tl", [
            {"mass": 204.9744278, "relative_intensity": 100},
            {"mass": 202.9723446, "relative_intensity": 41.88422247446083},
        ])           

        self._register("Pb", [
            {"mass": 207.9766525, "relative_intensity": 100},
            {"mass": 205.9744657, "relative_intensity": 45.99236641221374},
            {"mass": 206.9758973, "relative_intensity": 42.17557251908397},
            {"mass": 203.973044, "relative_intensity": 2.671755725190839},
        ])           

        self._register("Bi", [
            {"mass": 208.9803991, "relative_intensity": 100},
        ])           

        self._register("Po", [
            {"mass": 208.9824308, "relative_intensity": 100},
            {"mass": 209.9828741, "relative_intensity": 100},
        ])           

        self._register("At", [
            {"mass": 209.9871479, "relative_intensity": 100},
            {"mass": 210.9874966, "relative_intensity": 100},
        ])           

        self._register("Rn", [
            {"mass": 210.9906011, "relative_intensity": 100},
            {"mass": 220.0113941, "relative_intensity": 100},
            {"mass": 222.0175782, "relative_intensity": 100},
        ])      

        self._register("Fr", [
            {"mass": 223.019736, "relative_intensity": 100},
        ])           

        self._register("Ra", [
            {"mass": 224.020212, "relative_intensity": 100},
            {"mass": 228.0310707, "relative_intensity": 100},
            {"mass": 223.0185023, "relative_intensity": 100},
            {"mass": 226.0254103, "relative_intensity": 100},
        ])   

        self._register("Ac", [
            {"mass": 227.0277523, "relative_intensity": 100},
        ])           

        self._register("Th", [
            {"mass": 232.0380558, "relative_intensity": 100},
        ])    

        self._register("Pa", [
            {"mass": 231.0358842, "relative_intensity": 100},
        ])    

        self._register("U", [
            {"mass": 238.0507884, "relative_intensity": 100},
            {"mass": 235.0439301, "relative_intensity": 0.7256668902897229},
            {"mass": 234.0409523, "relative_intensity": 0.005439479743981821},
        ])


    def _register(self, symbol, isotopes):
        self._elements[symbol] = Element(symbol, isotopes)

    def __getattr__(self, item):
        if item in self._elements:
            return self._elements[item]
        raise AttributeError(f"Element '{item}' not found in periodic table.")

    def __getitem__(self, item):
        return self._elements[item]


# Single shared table instance
ptoe = PeriodicTable()

def truncate(num, digits):
    """
    Truncate a floating-point number to a specified number of decimal places without rounding.

    Parameters
    ----------
    num : float or int
        Numerical value to truncate.
    digits : int
        Number of decimal places to retain.

    Returns
    -------
    float
        Truncated floating-point value.
    """
    # Calculate scale factor as a power of 10
    factor = 10 ** digits
    
    # Cast to integer after multiplying by the factor to drop extra decimals, then scale back down
    return int(num * factor) / factor

def return_mass(str_formula):
    """
    Calculate the exact theoretical mass for a chemical formula or ion string.

    Natively parses neutral molecules, charged ions, and specific bracketed isotopes 
    (e.g., ``'[81]Br-'``, ``'H2O[35Cl]+'``, or ``'[35Cl]2+'``). Accounts for electron 
    mass adjustments based on net positive or negative charges.

    Parameters
    ----------
    str_formula : str, int, or float
        Chemical formula string (e.g., ``"C6H6+"``, ``"[81]Br-"``), unknown placeholder 
        string (e.g., ``"<unknown0001>"``), or pre-calculated numeric mass float/int.

    Returns
    -------
    float or None
        Calculated exact mass truncated to 6 decimal places, or ``None`` if the string 
        represents an unparseable placeholder (e.g., ``"<unknown0001>"``).
    """
    # Fast path: Return numeric inputs directly as floats
    if isinstance(str_formula, (int, float)):
        return float(str_formula)
        
    # Handle ambiguous Tofware placeholders (e.g., '<unknown0001>') by returning None
    if str_formula.startswith('<') and str_formula.endswith('>'):
        return None

    # Count positive (+) and negative (-) charges to adjust for electron mass later
    pos_charges = len(re.findall(r'\+', str_formula))
    neg_charges = len(re.findall(r'-', str_formula))

    # Extract target isotopes and optional multipliers via regex
    # Format 1 matches old format: '[81]Br2' -> ('81', 'Br', '2')
    format1 = re.findall(r'\[(\d+)\]([A-Z][a-z]?)(\d*)', str_formula)
    # Format 2 matches new format: '[35Cl]2' -> ('35', 'Cl', '2')
    format2 = re.findall(r'\[(\d+)([A-Z][a-z]?)\](\d*)', str_formula)
    
    # Combine extracted isotope tuples
    isotope_matches = format1 + format2
    
    # Strip isotope brackets so 'chemparse' can parse standard neutral chemical element counts
    almost_clean = re.sub(r'\[\d+\]([A-Z][a-z]?)', r'\1', str_formula)
    almost_clean = re.sub(r'\[\d+([A-Z][a-z]?)\]', r'\1', almost_clean)
    
    # Strip positive and negative charge symbols to isolate neutral stoichiometry
    clean_formula = re.sub(r'[\+-]', '', almost_clean) 
    
    # Parse formula into dictionary mapping element symbols to counts
    parsed = parse_formula(clean_formula)

    total_mass = 0.0

    # Calculate base neutral mass using standard monoisotopic masses from periodic table
    for element, count in parsed.items():
        total_mass += ptoe[element].monoisotopic_mass * count

    # Adjust mass by replacing monoisotopic mass with specific requested isotope masses
    for iso_num_str, element, count_str in isotope_matches:
        target_nominal_mass = int(iso_num_str)
        # Default multiplier to 1 if no explicit count is appended
        iso_count = int(count_str) if count_str else 1
        
        available_isotopes = ptoe[element].isotopes
        
        # Match registered isotope closest to the requested bracketed nominal integer mass
        matched_iso = min(available_isotopes, key=lambda x: abs(x["mass"] - target_nominal_mass))
        
        # Deduct standard monoisotopic mass and add the exact mass of the specific isotope
        total_mass -= (ptoe[element].monoisotopic_mass * iso_count)
        total_mass += (matched_iso["mass"] * iso_count)

    # --- Adjust for Electron Mass ---
    # Physical constant: Electron mass in Unified Atomic Mass Units (u)
    ELECTRON_MASS = 0.00054858
    total_mass += (neg_charges * ELECTRON_MASS) - (pos_charges * ELECTRON_MASS)

    # Truncate final calculated mass value to 6 decimal places
    return truncate(total_mass, 8)

def get_default_plot_dir():
    """
    Return the standard default plotting directory path.

    Returns
    -------
    str
        System path string pointing to ``~/OpenTof``.
    """
    # Expand user home directory path and append default 'OpenTof' directory name
    return os.path.join(os.path.expanduser("~"), "OpenTof")