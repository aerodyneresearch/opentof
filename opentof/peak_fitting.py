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

# peak_fitting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.optimize import nnls
# from scipy.optimize import lsq_linear
import scipy.signal as signal
import inspect
import os
from tqdm.auto import tqdm
import dask
from dask.diagnostics import ProgressBar
# from concurrent.futures import ThreadPoolExecutor, as_completed

from opentof.utils import (
    get_nm_segment_data, GlobalMinMaxScaler, return_mass, truncate,
    ensure_dir, get_default_plot_dir, median_absolute_deviation,
)

# ----------------
# Peak Shape Models
# ----------------
def gaussian(x, A, x_c, FWHM):
    """
    Evaluate a 1D Gaussian peak function parameterized by Full Width at Half Maximum (FWHM).

    Parameters
    ----------
    x : numpy.ndarray or float
        Independent variable coordinates (e.g., mass-to-charge m/z or time-of-flight in ns).
    A : float
        Peak height amplitude at the center x_c.
    x_c : float
        Peak center position coordinate.
    FWHM : float
        Full Width at Half Maximum of the peak in coordinate units.

    Returns
    -------
    numpy.ndarray or float
        Evaluated Gaussian peak intensity values matching the shape of `x`.
    """
    # 2 * ln(4) approx 2.7725887 converts FWHM squared to standard Gaussian variance denominator
    return A * np.exp(-2 * np.log(4) * ((x - x_c) ** 2) / (FWHM ** 2))

def lorentzian(x, A, x_c, FWHM):
    """
    Evaluate a 1D Lorentzian peak function parameterized by Full Width at Half Maximum (FWHM).

    Parameters
    ----------
    x : numpy.ndarray or float
        Independent variable coordinates (e.g., mass-to-charge m/z or time-of-flight in ns).
    A : float
        Peak height amplitude at the center x_c.
    x_c : float
        Peak center position coordinate.
    FWHM : float
        Full Width at Half Maximum of the peak in coordinate units.

    Returns
    -------
    numpy.ndarray or float
        Evaluated Lorentzian peak intensity values matching the shape of `x`.
    """
    # Scale width parameter to ensure output evaluates to amplitude A exactly at x = x_c
    return A * (FWHM ** 2) / (4 * (x - x_c) ** 2 + FWHM ** 2)

def pseudo_voigt(x, A, x_c, FWHM, m_u):
    """
    Evaluate a 1D Pseudo-Voigt peak function as a linear combination of Gaussian and Lorentzian profiles.

    Parameters
    ----------
    x : numpy.ndarray or float
        Independent variable coordinates (e.g., mass-to-charge m/z or time-of-flight in ns).
    A : float
        Peak height amplitude at the center x_c.
    x_c : float
        Peak center position coordinate.
    FWHM : float
        Full Width at Half Maximum of the peak in coordinate units.
    m_u : float
        Profile mixing weight factor constrained to [0, 1] (m_u = 1 is pure Lorentzian, m_u = 0 is pure Gaussian).

    Returns
    -------
    numpy.ndarray or float
        Evaluated Pseudo-Voigt peak intensity values matching the shape of `x`.
    """
    # Compute Gaussian component
    g = gaussian(x, A, x_c, FWHM)
    
    # Compute Lorentzian component
    l = lorentzian(x, A, x_c, FWHM)
    
    # Return weighted linear combination of Lorentzian and Gaussian shapes
    return m_u * l + (1 - m_u) * g

def peak_function_selector(peak_type, custom_shape=None):
    """
    Dispatch factory function to select a callable peak shape model by identifier.

    Parameters
    ----------
    peak_type : {'gaussian', 'lorentzian', 'pseudo_voigt', 'custom'}
        String identifier specifying the desired peak function model.
    custom_shape : callable or None, default=None
        Custom callable function required if `peak_type='custom'`. Must accept 
        arguments matching `(x, A, x_c, FWHM)`.

    Returns
    -------
    callable
        The selected peak function object.

    Raises
    ------
    ValueError
        If `peak_type='custom'` but `custom_shape` is ``None``, or if an unrecognized 
        `peak_type` string is provided.
    """
    # Dispatch built-in models by name
    if peak_type == 'gaussian':
        return gaussian
    elif peak_type == 'lorentzian':
        return lorentzian
    elif peak_type == 'pseudo_voigt':
        return pseudo_voigt
    elif peak_type == 'custom':
        # Enforce requirement for an explicit custom callable function
        if custom_shape is None:
            raise ValueError("custom_shape must be provided when using peak_type='custom'")
        return custom_shape
    else:
        raise ValueError(f"Unknown peak type: {peak_type}")

def levenberg_marquardt(
    model_func, x_data, y_data, initial_params,
    lower_bounds=None, upper_bounds=None,
    max_iter=1000, tol=1e-6, lambda_init=1e-3
):
    """
    Optimize model parameters using a custom Levenberg-Marquardt non-linear least squares algorithm.

    Fits `model_func` to target data by dynamically interpolating between Gauss-Newton and 
    gradient descent updates based on the damping factor λ. Computes numeric 
    Jacobian matrices via finite differences and enforces optional box parameter bounds.

    Victor's Note: This function is a legacy fitting function as it is much slower for 
    non-linear optimization than its 'C'-enabled SciPy cousins, so currently this is 
    only called during fitting for the peak-shape via 'fit_gaussian_for_peak_shape' 
    as speed isn't absolutely critical there and my attempts to update the function
    used in the peak shape procedure have yeilded incoherent results.

    Parameters
    ----------
    model_func : callable
        Target peak model function with signature ``model_func(x, *params)``.
    x_data : numpy.ndarray
        1D array of independent variable coordinate points.
    y_data : numpy.ndarray
        1D array of target observed intensity values.
    initial_params : sequence of float
        Initial parameter guesses for the optimization vector.
    lower_bounds : numpy.ndarray or None, default=None
        1D array of minimum allowed parameter values. Defaults to -infty if ``None``.
    upper_bounds : numpy.ndarray or None, default=None
        1D array of maximum allowed parameter values. Defaults to +infty if ``None``.
    max_iter : int, default=1000
        Maximum allowable solver iterations.
    tol : float, default=1e-6
        Convergence tolerance evaluated on cost function reduction Δ S.
    lambda_init : float, default=1e-3
        Initial Levenberg-Marquardt damping factor λ.

    Returns
    -------
    params : numpy.ndarray
        1D array containing optimized parameter values.
    iterations : int
        Number of iterations executed before convergence or termination.
    """
    # Cast initial guesses to a 1D double-precision float array
    params = np.array(initial_params, dtype=float)
    n_params = len(params)
    
    # Initialize damping factor lambda (controls interpolation between Gauss-Newton and Gradient Descent)
    lambda_ = lambda_init

    # Local helper function to evaluate residual vector r(p) = model(x; p) - y
    def residuals(p):
        return model_func(x_data, *p) - y_data

    # Local helper function to approximate the Jacobian matrix J (M x N) via forward finite differences
    def jacobian(p, eps=1e-8):
        J = np.zeros((len(y_data), n_params))
        for i in range(n_params):
            p_eps = p.copy()
            p_eps[i] += eps  # Perturb parameter i by step size eps
            
            # Compute partial derivative estimate wrt parameter i: d(residual)/d(p_i)
            J[:, i] = (residuals(p_eps) - residuals(p)) / eps
        return J

    # Set default unbounded bounds (-inf, +inf) if box bounds are omitted
    if lower_bounds is None:
        lower_bounds = -np.inf * np.ones_like(params)
    if upper_bounds is None:
        upper_bounds = np.inf * np.ones_like(params)

    # Begin main iterative optimization loop
    for iterations in range(max_iter):
        # Calculate current residuals and sum-of-squares cost S(p)
        res = residuals(params)
        cost = np.sum(res**2)

        # Estimate numerical Jacobian matrix J, approximate Hessian matrix H = J^T * J, and gradient g = J^T * r
        J = jacobian(params)
        H = J.T @ J
        g = J.T @ res

        # Construct damped Levenberg-Marquardt system matrix: A = H + lambda * I
        A = H + lambda_ * np.eye(n_params)
        
        # Solve linear system A * delta = -g to compute trial parameter step delta
        delta = np.linalg.solve(A, -g)

        # Calculate candidate parameters
        new_params = params + delta

        # Check for parameter box bound violations; if violated, reject step and increase damping
        if np.any(new_params < lower_bounds) or np.any(new_params > upper_bounds):
            lambda_ *= 10  # Increase lambda to take smaller, more gradient-descent-like steps
            continue

        # Evaluate cost with candidate parameters
        new_res = residuals(new_params)
        new_cost = np.sum(new_res**2)

        # Accept step if cost decreases; decrease damping factor lambda
        if new_cost < cost:
            params = new_params
            lambda_ *= 0.7  # Reduce lambda to shift toward faster Gauss-Newton convergence
            
            # Terminate optimization early if change in cost falls below tolerance threshold
            if np.abs(cost - new_cost) < tol:
                break
        else:
            # Reject step if cost increases; increase damping factor lambda
            lambda_ *= 2

    # Return optimized parameter vector along with total completed iterations
    return params, iterations

def generate_peak_mask(intensity_axis, noise_multiplier=3.0, smooth_fraction=0.10):
    """
    Generate a boolean mask classifying peak regions versus background noise.

    Employs an adaptive Savitzky-Golay filter to smooth noise spikes, establishes 
    a robust statistical noise baseline using Median Absolute Deviation (MAD), 
    and applies 1D binary spatial dilation to expand peak boundaries over peak tails.

    Parameters
    ----------
    intensity_axis : numpy.ndarray
        1D array of spectral intensity values.
    noise_multiplier : float, default=3.0
        Multiplier applied to MAD to establish the peak detection cutoff above the median.
    smooth_fraction : float, default=0.10
        Fraction of total array length used to set the adaptive smoothing window size.

    Returns
    -------
    extended_peak_mask : numpy.ndarray of bool
        1D boolean array where ``True`` marks peak regions and ``False`` marks background noise.
    """
    # Cast input array to double-precision float
    intensity_axis = np.asarray(intensity_axis, dtype=float)
    n_points = len(intensity_axis)
    
    # Return empty all-False boolean mask for short or invalid arrays
    if n_points < 4:
        return np.zeros_like(intensity_axis, dtype=bool)

    # Determine dynamic Savitzky-Golay window length proportional to total spectral points
    window_len = int(np.ceil(n_points * smooth_fraction))
    
    # Enforce odd integer window length requirement for Savitzky-Golay filter
    if window_len % 2 == 0:
        window_len += 1
        
    # Clamp window length between 3 and 15 sample bins for stability
    window_len = int(np.clip(window_len, 3, 15))

    # Apply 2nd-order Savitzky-Golay smoothing filter to suppress high-frequency noise spikes
    smoothed_intensity_axis = signal.savgol_filter(intensity_axis, window_length=window_len, polyorder=2)

    # Compute robust background statistics (Median and Median Absolute Deviation)
    local_median = np.median(smoothed_intensity_axis)
    mad = median_absolute_deviation(smoothed_intensity_axis)
    
    # Calculate statistical threshold floor; max(mad, 1e-6) prevents zero-division on flat baselines
    statistical_floor = local_median + (noise_multiplier * max(mad, 1e-6))

    # Identify core peak channels exceeding the statistical threshold
    core_peak_mask = smoothed_intensity_axis > statistical_floor
    
    # Set dilation padding width to half the filter window length
    padding_width = max(1, int(window_len // 2))
    extended_peak_mask = np.copy(core_peak_mask)

    # Perform 1D binary dilation via left/right bit-shifts to capture sloping peak tails
    for shift in range(1, padding_width + 1):
        extended_peak_mask_left = np.concatenate([core_peak_mask[shift:], np.zeros(shift, dtype=bool)])
        extended_peak_mask_right = np.concatenate([np.zeros(shift, dtype=bool), core_peak_mask[:-shift]])
        
        # Accumulate dilated regions using bitwise OR
        extended_peak_mask |= extended_peak_mask_left | extended_peak_mask_right

    return extended_peak_mask

def estimate_local_noise_level(intensity_axis, noise_multiplier=3.0, smooth_fraction=0.10):
    """
    Estimate the true background noise standard deviation from non-peak regions.

    Constructs a peak mask via :func:`generate_peak_mask`, isolates non-peak background 
    channels, and computes their sample standard deviation.

    Parameters
    ----------
    intensity_axis : numpy.ndarray
        1D array of spectral intensity values.
    noise_multiplier : float, default=3.5
        Multiplier applied to MAD in :func:`generate_peak_mask`.
    smooth_fraction : float, default=0.10
        Fraction of array length used to set adaptive smoothing in :func:`generate_peak_mask`.

    Returns
    -------
    float
        Calculated standard deviation of background noise channels.
    """
    intensity_axis = np.asarray(intensity_axis, dtype=float)
    
    # Generate boolean mask marking peak regions
    peak_mask = generate_peak_mask(intensity_axis, noise_multiplier=noise_multiplier, smooth_fraction=smooth_fraction)
    
    # Slice non-peak background channels using inverted mask
    noise_regions = intensity_axis[~peak_mask]
    
    # Fallback to MAD scaling factor (1.4826 * MAD approx std) if insufficient background points remain
    if noise_regions.size < 2:  
        mad = median_absolute_deviation(intensity_axis)
        return float(mad * 1.482602218505602)
        
    # Return sample standard deviation of isolated background channels
    return float(np.std(noise_regions))

def calculate_detection_threshold(intensity_axis, noise_level=None, noise_std_mult=10.0, noise_multiplier=3.0, signal_percentile=95):
    """
    Calculate local noise statistics, signal power, SNR, and detection/rejection thresholds for a spectral segment.

    Parameters
    ----------
    intensity_axis : numpy.ndarray
        1D array of spectral intensity values (ions/s).
    noise_level : float or None, default=None
        Background noise standard deviation (σ). Estimated automatically via `estimate_local_noise_level` if None.
    noise_std_mult : float, default=10.0
        Multiplier  (N * σ).
    signal_percentile : float, default=95
        Percentile rank evaluated to estimate total local signal power.

    Returns
    -------
    dict
        Dictionary containing calculated noise metrics:
            * ``'noise_level'`` : Baseline noise standard deviation (σ).
            * ``'detection_threshold'`` : Absolute intensity detection threshold (N * σ).
            * ``'signal_power'`` : Intensity value at the specified percentile rank.
            * ``'snr'`` : Estimated local Signal-to-Noise Ratio (signal_power / noise_level).
    """
    # Estimate noise level if omitted
    if noise_level is None:
        noise_level = estimate_local_noise_level(intensity_axis=intensity_axis, noise_multiplier=noise_multiplier)

    # Evaluate signal power percentile and find a typical threshold above the noise
    signal_power = float(np.percentile(intensity_axis, signal_percentile))
    detection_threshold = float(noise_std_mult * noise_level)
    snr = float(signal_power / (noise_level + 1e-8))

    return {
        'detection_threshold': detection_threshold,
        'noise_level': noise_level,
        'signal_power': signal_power,
        'snr': snr
    }

# Identify the largest peaks
def find_peaks_in_spectrum_ransac(spectrum, min_prominence=0.9):
    """
    Locate peak index positions in a 1D intensity spectrum above a prominence threshold.

    Uses :func:`scipy.signal.find_peaks` to identify local maxima exceeding a relative 
    prominence threshold, typically providing candidate peaks for RANSAC peak width regression.

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of spectral intensity values.
    min_prominence : float, default=0.9
        Minimum required topological peak prominence.

    Returns
    -------
    peak_locations : numpy.ndarray
        1D integer array containing array index coordinates of detected peaks.
    """
    # Identify local maxima matching or exceeding the specified prominence threshold
    peak_locations, _ = signal.find_peaks(spectrum, prominence=min_prominence)
    
    return peak_locations

def find_peaks_in_spectrum(spectrum, top_n=150, min_prominence=0.01):
    """
    Find and select the top N most prominent peak indices in a 1D spectrum.

    Identifies all local maxima above a baseline prominence floor, ranks candidates 
    by prominence (serving as a proxy for signal-to-noise ratio), selects the top 
    `top_n` candidates, and returns their indices sorted in ascending array order.

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of spectral intensity values.
    top_n : int, default=150
        Maximum number of most prominent peaks to retain.
    min_prominence : float, default=0.01
        Minimum prominence baseline threshold to qualify candidate peaks.

    Returns
    -------
    numpy.ndarray
        1D integer array of indices for the top `top_n` peaks, sorted in original 
        coordinate/mass order.
    """
    # Find all peak candidates meeting or exceeding the minimum prominence baseline
    peak_locations, properties = signal.find_peaks(spectrum, prominence=min_prominence)
    
    # If total detected candidates are within the requested limit, return all immediately
    if len(peak_locations) <= top_n:
        return peak_locations
        
    # Extract prominence values and identify indices of the top N highest prominence peaks
    prominences = properties['prominences']
    largest_peak_indices = np.argsort(prominences)[::-1][:top_n]
    
    # Return candidate array indices sorted back into chronological/mass coordinate order
    return np.sort(peak_locations[largest_peak_indices])


def fit_gaussian_for_peak_shape(x, y, peak_idx, search_range=20):
    """
    Fit a 1D Gaussian curve to a local window around a specific peak index.

    Slices a localized index segment centered at `peak_idx`, formulates heuristic 
    initial parameter guesses, and optimizes the Gaussian peak parameters using 
    a custom Levenberg-Marquardt optimizer.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of independent variable coordinate points (e.g., sample indices or m/z).
    y : numpy.ndarray
        1D array of spectral intensity values.
    peak_idx : int
        0-based index of the target peak center within `x` and `y`.
    search_range : int, default=20
        Half-width of the index slice window centered on `peak_idx`.

    Returns
    -------
    A_fit : float
        Fitted peak height amplitude.
    x_c_fit : float
        Fitted peak center coordinate.
    fwhm_fit : float
        Fitted Full Width at Half Maximum (FWHM) in coordinate units.
    x_slice : numpy.ndarray
        1D array of x-coordinates used in the local fit.
    y_slice : numpy.ndarray
        1D array of y-intensity values used in the local fit.
    
    Notes
    -----
    Returns ``None`` if the non-linear optimization fails to converge.
    """
    # Bound slicing indices to prevent IndexError at spectrum edges
    start = max(0, peak_idx - search_range)
    end = min(len(x), peak_idx + search_range)

    # Extract local coordinate and intensity window for peak fitting
    x_slice = x[start:end]
    y_slice = y[start:end]

    # Formulate heuristic initial parameter guesses
    A_guess = y[peak_idx]                               # Amplitude guess from peak apex intensity
    x_c_guess = x[peak_idx]                             # Center guess from peak apex coordinate
    fwhm_guess = (x[end - 1] - x[start]) / 5.0          # Crude initial FWHM guess (~1/5th window width)

    # Aggregate initial guesses into a vector
    initial_params = [A_guess, x_c_guess, fwhm_guess]

    # Define physical parameter box bounds for Levenberg-Marquardt optimization
    lower_bounds = [0.0, x_slice[0], 0.0]               # Amplitude >= 0, center >= window start, FWHM >= 0
    upper_bounds = [np.inf, x_slice[-1], np.inf]        # Center <= window end

    # Execute optimization inside exception guard to handle non-convergence gracefully
    try:
        fitted_params, iterations = levenberg_marquardt(
            gaussian,
            x_slice,
            y_slice,
            initial_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
        )
        
        # Unpack optimized Gaussian parameters: Amplitude, Center, and FWHM
        A_fit, x_c_fit, fwhm_fit = fitted_params
        return A_fit, x_c_fit, fwhm_fit, x_slice, y_slice

    except Exception as e:
        # Log failure and return None to prevent pipeline execution halts
        print(f"Fit failed at peak {peak_idx}: {e}")
        return None

def normalize_peak_shape(x_fit, y_fit, x_c, fwhm):
    """
    Normalize a peak shape onto a dimensionless standard deviation (`σ`) x-axis.

    Centers the x-axis coordinate grid at zero (x_c = 0) and scales the coordinate units 
    by the Gaussian standard deviation (1 σ = FWHM / 2.35482).

    Parameters
    ----------
    x_fit : numpy.ndarray
        1D array of original x-axis coordinate points.
    y_fit : numpy.ndarray
        1D array of corresponding intensity values.
    x_c : float
        Fitted peak center coordinate.
    fwhm : float
        Fitted Full Width at Half Maximum (FWHM).

    Returns
    -------
    x_normalized : numpy.ndarray
        1D array of centered and scaled x-coordinates in units of standard deviations (σ).
    y_fit : numpy.ndarray
        1D array of unmodified intensity values.
    """
    # Convert FWHM to Gaussian standard deviation (sigma): FWHM = 2 * sqrt(2 * ln(2)) * sigma approx 2.35482 * sigma
    sigma = fwhm / 2.3548200450309493
    
    # Center the coordinate axis at zero and scale by sigma
    x_normalized = (x_fit - x_c) / sigma
    
    return x_normalized, y_fit


def fit_and_compute_fwhm(spectrum, 
                         si_axis, 
                         peak_locations, 
                         pchip_interp, 
                         search_range=20, 
                         peak_type='gaussian',
                         custom_shape=None):
    """
    Fit local peak candidates and convert center and FWHM parameters into mass space.

    Iterates over candidate peak indices, slices a local window around each peak, 
    fits the requested model shape in sample index space via :func:`fit_and_extract_pos`, 
    and maps both the center position and Full Width at Half Maximum (FWHM) to 
    calibrated m/z units using PCHIP interpolation.

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of spectral intensity values.
    si_axis : numpy.ndarray
        1D array of sample index channel positions corresponding to `spectrum`.
    peak_locations : numpy.ndarray
        1D array of integer candidate peak indices (e.g., from :func:`find_peaks_in_spectrum`).
    pchip_interp : scipy.interpolate.PchipInterpolator
        PCHIP interpolator mapping sample index positions (SIP) to calibrated mass (m/z).
    search_range : int, default=20
        Width of the local fitting window (in sample index bins) centered on each peak.
    peak_type : {'gaussian', 'lorentzian', 'pseudo_voigt', 'custom'}, default='gaussian'
        Peak shape model identifier passed to :func:`fit_and_extract_pos`.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if `peak_type='custom'`.

    Returns
    -------
    numpy.ndarray
        2D array of shape ``(N_valid_peaks, 2)`` where column 0 contains calibrated peak 
        centers (m/z) and column 1 contains mass-based FWHMs (Δ m/z).
    """
    peak_data_mass = []

    # Iterate over each candidate peak location
    for peak in peak_locations:
        # Encapsulate local fit in try/except block to bypass convergence failures gracefully
        try:
            # Calculate window slicing boundaries centered on candidate index
            lower_i_slice = max(0, peak - int(search_range / 2))
            upper_i_slice = min(len(spectrum), peak + int(search_range / 2))

            # Fit local peak segment and extract fitted sample index center (x_c) and FWHM
            x_c, FWHM, _, _ = fit_and_extract_pos(
                spectrum[lower_i_slice:upper_i_slice], 
                si_axis[lower_i_slice:upper_i_slice], 
                peak_type=peak_type,
                custom_shape=custom_shape
            )

            # Translate fitted center position to calibrated m/z space
            x_c_mass = pchip_interp(x_c)
            
            # Interpolate upper and lower FWHM bounds (x_c +/- 0.5 * FWHM) to mass space
            lower_mass = pchip_interp(x_c - 0.5 * FWHM)
            upper_mass = pchip_interp(x_c + 0.5 * FWHM)
            
            # Calculate mass-based FWHM (Delta m/z)
            mass_based_fwhm = upper_mass - lower_mass

            # Append valid (m/z center, m/z FWHM) tuple
            peak_data_mass.append((x_c_mass, mass_based_fwhm))

        except RuntimeError:
            # Bypass failed fits without halting processing loop
            continue

    # Return structured 2D NumPy array of fitted mass centers and FWHMs
    return np.array(peak_data_mass)

def fit_and_extract_pos(intensity_axis, sample_idx_axis,
                        peak_type='pseudo_voigt',
                        plotting_flag=False, 
                        compound="", interval="", 
                        return_shape=False,
                        custom_shape=None,
                        output_dir=None,
                        plot_filename=None):
    """
    Fit a single peak within a sliced sample index window and extract optimized parameters.

    Normalizes the input segment intensity, formulates heuristic initial parameter guesses, 
    and optimizes peak parameters via the Levenberg-Marquardt algorithm. Optionally renders 
    a diagnostic plot and calculates normalized peak shape coordinates in units of standard deviations.

    Parameters
    ----------
    intensity_axis : numpy.ndarray
        1D array of intensity values for the local peak segment.
    sample_idx_axis : numpy.ndarray
        1D array of sample index channel positions corresponding to `intensity_axis`.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
        Peak shape model identifier.
    plotting_flag : bool, default=False
        If ``True``, generates and saves a diagnostic fit plot.
    compound : str, default=""
        Name or formula string of the target compound (used in plot titles/filenames).
    interval : str or int, default=""
        Time or calibration interval identifier (used in default plot filenames).
    return_shape : bool, default=False
        If ``True``, computes and appends normalized coordinates (x_{scaled}, y_{scaled})
        to the returned tuple.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    output_dir : str or None, default=None
        Directory path to save diagnostic plots. Defaults to OpenTof default plot dir if ``None``.
    plot_filename : str or None, default=None
        Custom filename for the saved plot. Defaults to ``mc{interval}_{compound}.png`` if ``None``.

    Returns
    -------
    x_c_opt : float
        Optimized peak center in sample index units.
    FWHM_opt : float
        Optimized Full Width at Half Maximum (FWHM) in sample index units.
    params : numpy.ndarray
        1D array of optimized parameters [A, x_c, FWHM, (m_u)].
    cov_placeholder : None
        Placeholder for covariance matrix compatibility.
    x_scaled : numpy.ndarray, optional
        1D array of centered x-coordinates in standard deviation units (returned if ``return_shape=True``).
    y_scaled : numpy.ndarray, optional
        1D array of peak-normalized intensity values (returned if ``return_shape=True``).
    """
    # Select peak function corresponding to requested model string or custom callable
    peak_function = peak_function_selector(peak_type, custom_shape)

    # Normalize intensity segment by max height to stabilize numerical optimization
    intensity_max = np.max(intensity_axis)
    normalized_intensityaxis = intensity_axis / intensity_max

    # Formulate initial parameter guesses for non-linear optimization
    A_guess = np.max(normalized_intensityaxis) # Amplitude guess (normalized)
    argmax_idx = np.argmax(normalized_intensityaxis)
    x_c_guess = sample_idx_axis[argmax_idx] # Center position guess
    
    # Estimate FWHM: fallback to 1 bin if peak is at slice edge, else double 1-bin neighborhood spacing
    FWHM_guess = (
        1 if argmax_idx in [0, len(sample_idx_axis) - 1] 
        else 2 * (sample_idx_axis[argmax_idx + 1] - sample_idx_axis[argmax_idx - 1])
    )
    m_u_guess = 0.5 # 50/50 Gaussian-Lorentzian mixing guess

    # Assemble initial guess vector based on peak model type
    if peak_type == 'pseudo_voigt':
        initial_guesses = [A_guess, x_c_guess, FWHM_guess, m_u_guess]
    else:
        initial_guesses = [A_guess, x_c_guess, FWHM_guess]

    # Execute Levenberg-Marquardt solver to optimize parameters
    params, iterations = levenberg_marquardt(
        peak_function, 
        sample_idx_axis, 
        normalized_intensityaxis, 
        initial_guesses
    )

    # Unpack optimized peak position and width parameters
    x_c_opt = params[1]   # Fitted peak center (SIP)
    FWHM_opt = params[2]  # Fitted FWHM (sample index channels)
    
    # Evaluate fitted model curve across local sample index segment
    fitted_curve = peak_function(sample_idx_axis, *params) 

    # Render diagnostic plot if requested
    if plotting_flag:
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        if plot_filename is None:
            plot_filename = f"mc{interval}_{compound}.png"
       
        plt.figure(figsize=(8, 5))
        plt.plot(sample_idx_axis, intensity_axis, label='Intensity Data', marker='o', linestyle='-', markersize=3)
        plt.plot(sample_idx_axis, fitted_curve * intensity_max, label=f'Fitted {peak_type} Curve', color='red', linewidth=2)
        plt.axvline(x=x_c_opt, color='green', linestyle='--', label=f'Fitted SIP: {x_c_opt:.2f}')
        plt.title(f'Peak Fitting for {compound}')
        plt.xlabel('Sample Index Axis')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    # Calculate centered and scaled shape vectors if requested
    if return_shape:
        sigma = FWHM_opt / 2.3548200450309493                     # Convert FWHM to Gaussian sigma
        x_centered = sample_idx_axis - x_c_opt                    # Center coordinate origin at zero
        x_scaled = x_centered / sigma                             # Scale x-axis into standard deviation units
        y_scaled = fitted_curve / np.max(fitted_curve)            # Scale peak amplitude height to 1.0
        return x_c_opt, FWHM_opt, params, None, x_scaled, y_scaled

    return x_c_opt, FWHM_opt, params, None

def fit_single_peak(peak_mass, intensity_axis_data, mass_axis_data, search_range=20, 
                    peak_type='pseudo_voigt', plotting_flag=False,
                    custom_shape=None,
                    output_dir=None,
                    plot_filename=None):
    """
    Locate and fit a single target mass peak within full mass spectrum arrays.

    Identifies the channel closest to `peak_mass`, slices a local search window, 
    fits the peak in sample index space via :func:`fit_and_extract_pos`, and interpolates 
    the fitted sample index center back to a calibrated m/z coordinate.

    Parameters
    ----------
    peak_mass : float
        Target mass-to-charge ratio (m/z) to locate and fit.
    intensity_axis_data : numpy.ndarray
        1D array containing full mass spectrum intensity values.
    mass_axis_data : numpy.ndarray
        1D array containing calibrated mass-to-charge (m/z) coordinates.
    search_range : int, default=20
        Total width of the sample index search window centered on `peak_mass`.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
        Peak shape model identifier.
    plotting_flag : bool, default=False
        If ``True``, generates and exports a diagnostic plot of the fit.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    output_dir : str or None, default=None
        Directory path to save diagnostic plots. Defaults to OpenTof default plot dir if ``None``.
    plot_filename : str or None, default=None
        Filename for the saved plot image. Defaults to ``single_peak_fit.png`` if ``None``.

    Returns
    -------
    sample_idx_position_for_peak : float
        Fitted peak center position in sample index units.
    fwhm_for_peak : float
        Fitted Full Width at Half Maximum (FWHM) in sample index units.
    params : numpy.ndarray
        1D array of optimized model parameters [A, x_c, FWHM, (m_u)].
    mass_pos : float
        Calibrated m/z coordinate corresponding to `sample_idx_position_for_peak`.
    
    Notes
    -----
    Returns ``None`` if the underlying non-linear optimization fails to converge.
    """
    # Locate sample index closest to requested target peak mass
    peak_index = np.argmin(np.abs(mass_axis_data[:] - peak_mass))
    
    # Calculate local slicing boundaries centered on peak_index
    start_idx = int(peak_index - (search_range / 2))
    end_idx = int(peak_index + (search_range / 2))

    # Slice intensity segment and create matching sample index and mass coordinate slices
    intensityaxis = intensity_axis_data[start_idx:end_idx]
    sample_idx_for_peak = np.arange(start_idx, end_idx)
    small_mass_axis = mass_axis_data[sample_idx_for_peak.astype(int)]

    # Execute fit inside exception block to catch non-convergence gracefully
    try:
        sample_idx_position_for_peak, fwhm_for_peak, params, _ = fit_and_extract_pos(
            intensityaxis, sample_idx_for_peak, 
            peak_type=peak_type, 
            compound=str(peak_mass), 
            plotting_flag=False,
            custom_shape=custom_shape
        )
    except Exception as e:
        print(f"fit_single_peak fit failed: ({e})")
        print("Returning: None to protect against crashes")
        return None

    # Retrieve callable peak function object and evaluate fitted curve across slice
    peak_function = peak_function_selector(peak_type, custom_shape)
    fitted_curve = peak_function(sample_idx_for_peak, *params)

    # Render diagnostic plot if requested
    if plotting_flag:
        if output_dir is None:
            output_dir = get_default_plot_dir()
        ensure_dir(output_dir)
        
        if plot_filename is None:
            plot_filename = "single_peak_fit.png"

        plt.figure(figsize=(8, 5))
        plt.plot(sample_idx_for_peak, intensityaxis, label='Intensity Data', marker='o', linestyle='-', markersize=3)
        plt.plot(sample_idx_for_peak, fitted_curve * np.max(intensityaxis), label=f'Fitted {peak_type} Curve', color='red', linewidth=2)
        plt.axvline(x=sample_idx_position_for_peak, color='green', linestyle='--', label=f'Fitted TOF: {np.round(sample_idx_position_for_peak, 3)}')
        plt.title(f'Peak Fitting at Mass {truncate(peak_mass, 6)}')
        plt.xlabel('Sample Index Axis')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.clf()
        plt.close()

    # Interpolate fitted sample index position back into calibrated m/z space
    mass_pos = np.interp(sample_idx_position_for_peak, sample_idx_for_peak, small_mass_axis)

    return sample_idx_position_for_peak, fwhm_for_peak, params, mass_pos

def fit_unconstrained_peaks(x_axis, signal, 
                            centers_guess, fwhms_guess, amplitudes_guess,
                            peak_type='pseudo_voigt', custom_shape=None,
                            center_wiggle=0.5, fwhm_range=(0.2, 5.0),
                            max_iter=1000, tol=1e-8):
    """
    Fit multiple overlapping peaks simultaneously by optimizing amplitude, center, and width.

    Formulates parameter box bounds for each candidate peak and solves the non-linear 
    least-squares problem using SciPy's Trust Region Reflective (TRF) algorithm 
    (:func:`scipy.optimize.least_squares`). Supports per-peak center drift bounds 
    (x_c +-FWHM * wiggle) and FWHM range multipliers.

    Parameters
    ----------
    x_axis : numpy.ndarray
        1D array of independent coordinate points (e.g., time-of-flight in ns or m/z).
    signal : numpy.ndarray
        1D array of observed intensity values (y-axis).
    centers_guess : sequence of float
        Initial position guesses x_c for each candidate peak.
    fwhms_guess : sequence of float
        Initial Full Width at Half Maximum (FWHM) guesses for each candidate peak.
    amplitudes_guess : sequence of float
        Initial amplitude height guesses A for each candidate peak.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
        Peak shape model identifier.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    center_wiggle : float or sequence of float, default=0.5
        Fraction of initial FWHM that peak centers are allowed to drift during optimization. 
        Can be specified as a scalar (applied globally) or a sequence of floats per peak.
    fwhm_range : tuple of (float, float), default=(0.2, 5.0)
        Min and max multipliers scaling the FWHM bounds relative to `fwhms_guess` 
        ([FWHM_{guess} * f_0, FWHM_{guess} * f_1]).
    max_iter : int, default=1000
        Maximum allowable solver iterations (`max_nfev = max_iter * 10`).
    tol : float, default=1e-4
        Convergence tolerance passed to TRF optimizer (`ftol`, `xtol`, `gtol`).

    Returns
    -------
    popt : numpy.ndarray
        1D array of optimized flat parameters organized as [A_1, x_{c1}, FWHM_1, (m_{u1}), A_2, x_{c2}, FWHM_2, (m_{u2}), ...].
        If optimization fails, returns the unoptimized initial guess vector as a fallback.
    """
    # Retrieve callable model function object corresponding to peak_type
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)
    
    initial_params = []
    lower_bounds = []
    upper_bounds = []
    
    n_peaks = len(centers_guess)
    is_pv = (peak_type == 'pseudo_voigt')
    n_params_per_peak = 4 if is_pv else 3

    # Support either a single global float scalar or per-peak list/array of wiggles
    if not isinstance(center_wiggle, (list, tuple, np.ndarray)):
        center_wiggle = [center_wiggle] * n_peaks

    # Construct initial parameter vector and corresponding lower/upper box bounds
    for i in range(n_peaks):
        # Enforce minimum positive amplitude floor (1e-6) to prevent negative noise bound violations
        A = max(amplitudes_guess[i], 1e-6)
        xc = centers_guess[i]
        fwhm = fwhms_guess[i]
        wig = center_wiggle[i]  # Local center drift tolerance multiplier for peak i
        
        initial_params.extend([A, xc, fwhm])
        # Parameter bounds: [A_min, xc_min, fwhm_min] -> [A_max, xc_max, fwhm_max]
        lower_bounds.extend([0.0, xc - (fwhm * wig), fwhm * fwhm_range[0]])
        upper_bounds.extend([np.inf, xc + (fwhm * wig), fwhm * fwhm_range[1]])
        
        # Append Pseudo-Voigt mixing factor parameter (m_u) and bounds [0.0, 1.0] if active
        if is_pv:
            initial_params.append(0.5)  # Initial 50/50 Gaussian-Lorentzian mixing guess
            lower_bounds.append(0.0)    # Minimum mixing weight (pure Gaussian)
            upper_bounds.append(1.0)    # Maximum mixing weight (pure Lorentzian)

    # Local residual objective evaluator: r(p) = model_composite(x; p) - observed_signal
    def objective_residuals(params):
        result = np.zeros_like(x_axis, dtype=float)
        for i in range(n_peaks):
            p = params[i * n_params_per_peak : (i + 1) * n_params_per_peak]
            if is_pv:
                result += peak_func(x_axis, p[0], p[1], p[2], p[3])
            else:
                result += peak_func(x_axis, p[0], p[1], p[2])
        return result - signal  # Residual vector

    # Execute non-linear bounded least-squares optimization using Trust Region Reflective (TRF) algorithm
    try:
        res = least_squares(
            objective_residuals,
            x0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            method='trf', 
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_iter * 10
        )
        popt = res.x
    except Exception as e:
        # Fallback to initial unoptimized parameter guesses if optimization fails or raises an error
        print(f"Fit failed: {e}")
        popt = np.array(initial_params, dtype=float)
        
    return popt

def multi_overlap_peak_fit(mass_axis,
                           intensity_axis,
                           tof_axis,
                           peak_type='gaussian',
                           custom_shape=None,
                           min_separation_fwhm=None,
                           max_peaks_per_mq=None,
                           noise_level=None,
                           noise_std_mult=10,
                           rel_filter=None,
                           abs_filter=None,
                           smooth_factor=-1,
                           smooth_function='inverse linear',
                           tol=1e-8,
                           max_iter=1000,
                           plot_flag=False,
                           plot_type="mass",
                           show_plot_flag=True,
                           save_plot_flag=True,
                           output_dir=None,
                           verbose=False,
                           plot_filename='mopf.png',
                           # --- ADVANCED PARAMETERS ---
                           deriv_threshold=0.0,        # % threshold of max 2nd derivative prominence
                           peaks_to_add_post_fit=0,    # Grand Loop iteration limit
                           abs_residual_threshold=0.0, # Target floor to terminate Grand Loop
                           peak_width_constraint_factor=5.0, # The 'X' box constraint factor
                           ):
    """
    Deconvolute and fit multiple overlapping peaks within a spectral region.

    Identifies candidate peak centers via Savitzky-Golay smoothed 2nd derivative 
    extrema, applies threshold filters (detection threshold, relative/absolute intensity, 
    prominence), enforces spatial separation constraints, and performs non-linear 
    least-squares fitting. Supports an iterative "Grand Loop" to add forced peak 
    candidates at regions of large residual discrepancy.

    Parameters
    ----------
    mass_axis : numpy.ndarray
        1D array of mass-to-charge (m/z) coordinates for the spectral region.
    intensity_axis : numpy.ndarray
        1D array of spectral intensity values (ions/s).
    tof_axis : numpy.ndarray
        1D array of physical time-of-flight coordinates in nanoseconds.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='gaussian'
        Peak shape model identifier.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    min_separation_fwhm : float or None, default=None
        Minimum required spatial separation between adjacent peak centers in FWHM units.
        If ``None``, defaults to ``max(0.005, nominal_mass / 2000.0)``.
    max_peaks_per_mq : int or None, default=None
        Maximum allowable number of peak candidates retained per nominal mass window.
    noise_level : float or None, default=None
        Background noise standard deviation (σ). Estimated automatically if ``None``.
    noise_std_mult : float, default=10.0
        Signal-to-noise detection threshold multiplier (N * σ).
    rel_filter : float or None, default=None
        Minimum normalized intensity threshold in [0.0, 1.0].
    abs_filter : float or None, default=None
        Minimum absolute intensity threshold (ions/s).
    smooth_factor : int, default=-1
        Savitzky-Golay smoothing window length in sample bins. If ``-1``, calculated 
        dynamically from the estimated signal-to-noise ratio.
    smooth_function : {'inverse linear', 'logistic', 'hyperbolic decay', 'double logarithmic', 'logarithmic'}, default='inverse linear'
        Model equation used to compute adaptive Savitzky-Golay window size from SNR.
    tol : float, default=1e-8
        Convergence tolerance passed to the non-linear least-squares optimizer.
    max_iter : int, default=1000
        Maximum iterations allowed per optimization call.
    plot_flag : bool, default=False
        If ``True``, generates diagnostic 2-panel fit and residual figures.
    plot_type : {'mass', 'tof'}, default='mass'
        Coordinate unit displayed on the plot x-axis.
    show_plot_flag : bool, default=True
        If ``True``, renders generated plots interactively.
    save_plot_flag : bool, default=True
        If ``True``, exports generated diagnostic plots to disk.
    output_dir : str or None, default=None
        Export directory path. Defaults to OpenTof default plot directory if ``None``.
    verbose : bool, default=False
        If ``True``, prints detailed diagnostic and optimization messages.
    plot_filename : str, default='mopf.png'
        Filename for exported plot images.
    deriv_threshold : float, default=0.0
        Minimum 2nd derivative prominence threshold expressed as a percentage (0-100%) 
        of the maximum observed derivative curvature.
    peaks_to_add_post_fit : int, default=0
        Maximum number of forced residual additions ("Grand Loop" iterations).
    abs_residual_threshold : float, default=0.0
        Absolute residual intensity floor to terminate "Grand Loop" forced peak additions.
    peak_width_constraint_factor : float, default=5.0
        FWHM box constraint multiplier X establishing parameter bounds 
        [1/X * FWHM, X * FWHM].

    Returns
    -------
    dict or None
        A dictionary containing deconvoluted fit results:
            * ``"popt"`` : 1D numpy.ndarray of flat optimized parameters.
            * ``"pcov"`` : Covariance matrix (set to ``None``).
            * ``"peaks"`` : List of dictionaries detailing parameters for each fitted peak.
            * ``"area_original"`` : Integrated area of the original input spectrum.
            * ``"area_fitted"`` : Integrated area of the fitted composite model.
            * ``"area_ratio_percent"`` : Ratio of fitted area to original area (100 * A_{fit} / A_{orig}).
            * ``"snr"`` : Estimated signal-to-noise ratio.
            * ``"smoothing_factor"`` : Savitzky-Golay window length used.
        Returns ``None`` if no peaks pass thresholding or fitting constraints.
    """
    # Initialize output directory defaults
    if output_dir is None:
        output_dir = get_default_plot_dir()
        
    ensure_dir(output_dir)
        
    # Determine central nominal mass label for diagnostic reporting
    nominal_mass_label = int(np.round(np.median(mass_axis)))

    # Establish default minimum separation threshold if unassigned
    if min_separation_fwhm is None:
        print("'min_separation_fwhm' is not specified. Ensure an appropriate peak width guess is provided!")
        min_separation_fwhm = max(0.005, nominal_mass_label / 2000.0)

    # Lambda mapping helper to convert TOF (ns) coordinates back to m/z values
    tof_to_mass = lambda t: np.interp(t, tof_axis, mass_axis)
    
    # Retrieve peak function object and determine parameter count per peak (4 for Pseudo-Voigt, 3 otherwise)
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)
    n_p = 4 if peak_type == 'pseudo_voigt' else 3

    # Estimate a reasonable guess at the "noise" within the intensity_axis
    noise_dict = calculate_detection_threshold(intensity_axis=intensity_axis, noise_level=noise_level, noise_std_mult=noise_std_mult)

    # Normalize intensity to [0, 1] range to ensure stable Jacobian scaling
    scaler = GlobalMinMaxScaler().fit(intensity_axis)
    norm_intensity = scaler.transform(intensity_axis) 
    
    # --- INLINE DIAGNOSTIC FALLBACK HANDLER ---
    # Renders 2-panel diagnostic figures when no peak candidates pass filtering
    def handle_no_peaks_found(message, candidates=None, verbose=verbose):
        if verbose:
            print(message)
        if plot_flag:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, 
                figsize=(10, 6), 
                sharex=True, 
                gridspec_kw={'height_ratios': [3, 1]},
                dpi=150
            )
            
            plot_x = mass_axis if plot_type.lower() == "mass" else tof_axis
            x_label = "Mass (m/z)" if plot_type.lower() == "mass" else "ToF (ns)"
            
            # --- TOP PANEL: Spectrum and detection thresholds ---
            ax1.plot(plot_x, intensity_axis, label='Original Signal', color='black', alpha=0.7, zorder=2)
            
            # Draw detection threshold gatekeeper lines
            ax1.axhline(noise_dict['detection_threshold'], label=f'Detection Threshold ({noise_std_mult}σ)', linestyle="-.", color='red', alpha=0.3)
            ax1.axhline(noise_dict['signal_power'], label='Signal Power (p95)', linestyle="-.", color='purple', alpha=0.3)

            # Scatter rejected candidates
            if candidates is not None and len(candidates) > 0:
                ax1.scatter(plot_x[candidates], intensity_axis[candidates], 
                            color='darkorange', marker='v', s=25, label='Rejected Candidates', zorder=4)

            # Diagnostic summary text box
            diagnostic_text = "0.000% of Original\nPeaks: 0"
            ax1.text(
                0.98, 0.95, diagnostic_text,
                horizontalalignment='right', verticalalignment='top',
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="lightgray", facecolor="white", alpha=0.8)
            )

            ax1.set_title(f"Multi-Overlapping Peak Fit - No Peaks Retained (m/z ~ {nominal_mass_label})")
            ax1.set_ylabel("Intensity (ions/s)")
            ax1.legend(fontsize='small', loc='upper left')
            ax1.grid(True, linestyle=':', alpha=0.5)

            # --- BOTTOM PANEL: Unfitted Residuals ---
            ax2.plot(plot_x, intensity_axis, color='crimson', lw=1, label='Residuals (Unfitted Profile)')
            ax2.axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.7)
            
            # Shade 1-sigma noise floor band
            ax2.fill_between(plot_x, -noise_dict['noise_level'], noise_dict['noise_level'], color='gray', alpha=0.15, label='Noise Band (1σ)')

            ax2.set_ylabel("Residual Error")
            ax2.set_xlabel(x_label)
            ax2.grid(True, linestyle=':', alpha=0.5)
            ax2.legend(fontsize='x-small', loc='lower left')

            plt.tight_layout()
            if save_plot_flag:
                plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
            if show_plot_flag:
                plt.show()
            plt.close()
        return None

    # --- DYNAMIC SAVITZKY-GOLAY SMOOTHING WINDOW CALCULATION ---
    if smooth_factor == -1:
        snr = noise_dict['snr']
        
        smoothing_models = {
            'logistic': 3 + 8 / (1 + np.exp(snr - 5)),
            'hyperbolic decay': 3 + 20 / (snr + 1),
            'double logarithmic': 3 + np.log1p(np.log1p(20 / (snr + 1))),
            'logarithmic': 9 - 0.5 * np.log1p(snr),
            'inverse linear': 3 + 8 / (1 + snr)
        }

        # Select window size model and enforce an odd integer window length
        smf = int(np.clip(smoothing_models.get(smooth_function, 3 + 8 / (1 + snr)), 3, 15))
        smooth_factor = smf if smf % 2 == 1 else smf + 1

    # --- PEAK DISCOVERY VIA 2ND DERIVATIVE ---
    deriv = signal.savgol_filter(norm_intensity, window_length=smooth_factor, polyorder=2, deriv=2)
    neg_deriv = -deriv  # Negate so concave-down peaks become positive maxima
    candidate_idxs = np.array(signal.find_peaks(neg_deriv)[0])

    if len(candidate_idxs) == 0:
        return handle_no_peaks_found(f"--> No peak candidates found in the smoothed 2nd derivative for NM {nominal_mass_label}.")

    # --- CANDIDATE FILTERING ---
    mask = np.ones_like(candidate_idxs, dtype=bool)

    # Always perform filtering based on the rejection threshold
    mask &= intensity_axis[candidate_idxs] >= noise_dict['detection_threshold']

    if abs_filter is not None:
        mask &= intensity_axis[candidate_idxs] >= abs_filter
    if rel_filter is not None:
        mask &= norm_intensity[candidate_idxs] >= rel_filter
        
    # Filter candidates by 2nd derivative prominence percentage
    if deriv_threshold > 0.0 and len(candidate_idxs) > 0:
        max_deriv_val = np.max(neg_deriv[candidate_idxs])
        mask &= neg_deriv[candidate_idxs] >= (deriv_threshold / 100.0) * max_deriv_val
        
    valid_peaks = candidate_idxs[mask]

    if len(valid_peaks) == 0:
        return handle_no_peaks_found(f"--> No peaks survived threshold filtering for NM {nominal_mass_label}.", candidates=candidate_idxs)

    # --- SPATIAL SEPARATION CONSTRAINTS ---
    mz_per_index = np.mean(np.diff(mass_axis))
    min_index_separation = int(np.ceil(min_separation_fwhm / mz_per_index))
    sorted_peaks = valid_peaks[np.argsort(intensity_axis[valid_peaks])[::-1]]

    filtered_peaks = []
    for idx in sorted_peaks:
        if all(abs(idx - sel_idx) >= min_index_separation for sel_idx in filtered_peaks):
            filtered_peaks.append(idx)

    # Cap maximum candidates per region if requested
    if max_peaks_per_mq is not None:
        filtered_peaks = filtered_peaks[:max_peaks_per_mq]

    # Convert peak width constraint factor 'X' into parameter box bounds
    fwhm_bounds_tuple = (1.0 / peak_width_constraint_factor, peak_width_constraint_factor)
    dm_dtof = np.gradient(mass_axis, tof_axis) 

    # =========================================================================
    # CORE ENGINE: UNCONSTRAINED FITTING WITH UNDERCONSTRAINED PRUNING LOOP
    # =========================================================================
    popt = None
    active_peak_indices = list(filtered_peaks)

    # Iteratively fit active candidates; if optimization fails/underconstrained, prune weakest peak
    while len(active_peak_indices) > 0:
        centers_guess, fwhms_guess, amplitudes_guess = [], [], []
        for i in active_peak_indices:
            amplitudes_guess.append(intensity_axis[i])
            centers_guess.append(tof_axis[i])
            fwhms_guess.append((min_separation_fwhm / dm_dtof[i]) * 0.8)

        try:
            popt = fit_unconstrained_peaks(
                x_axis=tof_axis, signal=intensity_axis,
                centers_guess=centers_guess, fwhms_guess=fwhms_guess, amplitudes_guess=amplitudes_guess,
                peak_type=peak_type, custom_shape=custom_shape, center_wiggle=0.2, 
                fwhm_range=fwhm_bounds_tuple, max_iter=max_iter, tol=tol
            )
            # Verify stable parameter vector returned
            if len(popt) == len(active_peak_indices) * n_p:
                break
        except Exception:
            pass

        # Underconstrained fit fallback: prune candidate with lowest intensity
        if verbose:
            print(f"--> Fit underconstrained for NM {nominal_mass_label}. Dropping weakest peak candidate.")
        weakest_local_idx = np.argmin([intensity_axis[i] for i in active_peak_indices])
        active_peak_indices.pop(weakest_local_idx)
    else:
        return handle_no_peaks_found(f"--> All peak candidates for NM {nominal_mass_label} were eliminated as underconstrained.", candidates=filtered_peaks)

    # Helper lambda to evaluate composite peak model
    def evaluate_profile(p_vector):
        curve = np.zeros_like(tof_axis, dtype=float)
        for i_p in range(len(p_vector) // n_p):
            p = p_vector[i_p * n_p : (i_p + 1) * n_p]
            curve += peak_func(tof_axis, p[0], p[1], p[2], p[3]) if peak_type == 'pseudo_voigt' else peak_func(tof_axis, p[0], p[1], p[2])
        return curve

    fitted_curve = evaluate_profile(popt)

    # =========================================================================
    # ADVANCED FEATURE: THE GRAND LOOP (FORCED RESIDUAL ADDITIONS)
    # =========================================================================
    for grand_iter in range(peaks_to_add_post_fit):
        residuals = intensity_axis - fitted_curve
        max_abs_residual = np.max(np.abs(residuals))
        
        # Terminate if maximum residual drops below threshold floor
        if max_abs_residual <= abs_residual_threshold:
            if verbose:
                print(f"--> Grand Loop terminated: Residual ({max_abs_residual:.4f}) dropped below floor.")
            break

        # Locate spatial coordinate where fit residual discrepancy is highest
        forced_coord_idx = np.argmax(np.abs(residuals))
        forced_tof_center = tof_axis[forced_coord_idx]
        forced_amp_guess = np.abs(residuals[forced_coord_idx])
        forced_fwhm_guess = (min_separation_fwhm / dm_dtof[forced_coord_idx]) * 0.8

        # Append new candidate parameters to active parameter vector
        new_peak_guesses = [forced_amp_guess, forced_tof_center, forced_fwhm_guess]
        if peak_type == 'pseudo_voigt':
            new_peak_guesses.append(0.5)
            
        extended_initial_guesses = list(popt) + new_peak_guesses
        
        # Unpack initial guesses for refitting driver
        next_amps, next_centers, next_fwhms = [], [], []
        for i_p in range(len(extended_initial_guesses) // n_p):
            next_amps.append(extended_initial_guesses[i_p * n_p])
            next_centers.append(extended_initial_guesses[i_p * n_p + 1])
            next_fwhms.append(extended_initial_guesses[i_p * n_p + 2])

        try:
            # Re-optimize expanded peak system
            popt_extended = fit_unconstrained_peaks(
                x_axis=tof_axis, signal=intensity_axis,
                centers_guess=next_centers, fwhms_guess=next_fwhms, amplitudes_guess=next_amps,
                peak_type=peak_type, custom_shape=custom_shape, center_wiggle=0.2,
                fwhm_range=fwhm_bounds_tuple, max_iter=max_iter, tol=tol
            )
            if len(popt_extended) == len(extended_initial_guesses):
                popt = popt_extended
                fitted_curve = evaluate_profile(popt)
                if verbose:
                    print(f"--> Grand Loop successfully added forced peak {grand_iter + 1} at {forced_tof_center:.2f} TOF.")
        except Exception as e:
            if verbose:
                print(f"--> Grand Loop failed to append peak layout configuration: {e}")
            break

    # =========================================================================
    # RECONSTRUCT RESULTS DICTIONARY & ANALYTICAL AREA INTEGRATION
    # =========================================================================
    area_original = simpson(intensity_axis, tof_axis)
    area_fitted = simpson(fitted_curve, tof_axis)
    area_ratio = 100 * (area_fitted / area_original) if area_original > 0 else 0.0

    results = []
    gauss_const = np.sqrt(np.pi / (4 * np.log(2)))
    total_fitted_peaks = len(popt) // n_p
    
    for i in range(total_fitted_peaks):
        params = popt[i * n_p : (i + 1) * n_p]
        amp, center, fwhm = params[0], params[1], params[2]

        # Analytical area integration by shape model
        if peak_type == "gaussian":
            peak_area = amp * fwhm * gauss_const
        elif peak_type == "lorentzian":
            peak_area = amp * (np.pi * fwhm / 2)
        elif peak_type == "pseudo_voigt":
            peak_area = params[3] * (amp * np.pi * fwhm / 2) + (1 - params[3]) * (amp * fwhm * gauss_const)
        else:
            peak_area = simpson(peak_func(tof_axis, *params), tof_axis)

        results.append({
            "center_tof": center,
            "center_mass": float(tof_to_mass(center)),
            "fwhm_tof": fwhm,
            "fwhm_mass": abs(float(tof_to_mass(center + fwhm/2)) - float(tof_to_mass(center - fwhm/2))),
            "amplitude": amp,
            "area": peak_area,
            **({"mixing": params[3]} if peak_type == "pseudo_voigt" else {})
        })

    # --- DIAGNOSTIC PLOTTING ---
    if plot_flag:
        # Create a 2-panel layout: Main spectrum on top, Residuals on the bottom
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=(10, 6), 
            sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]},
            dpi=150
        )
        
        plot_x = mass_axis if plot_type.lower() == "mass" else tof_axis
        x_label = "Mass (m/z)" if plot_type.lower() == "mass" else "ToF (ns)"
        
        # --- TOP PANEL: Raw data and deconvoluted peak curves ---
        ax1.plot(plot_x, intensity_axis, label='Original Signal', color='black', alpha=0.7, zorder=2)
        ax1.plot(plot_x, fitted_curve, label='Total Composite Fit', color='green', lw=2, zorder=3)
        
        for i, raw_p in enumerate(results):
            peak_vals = popt[i * n_p : (i + 1) * n_p]
            ax1.plot(plot_x, peak_func(tof_axis, *peak_vals), '--', alpha=0.6, 
                     label=f"Peak {i+1} ({raw_p['center_mass']:.3f} m/z)")
            
            center_coord = raw_p['center_mass'] if plot_type.lower() == "mass" else raw_p['center_tof']
            ax1.axvline(center_coord, color='gray', linestyle=':', alpha=0.5)

        # Plot the detection threshold ("where the noise ends and signal starts") cutoff line
        ax1.axhline(noise_dict['detection_threshold'], label=f'Detection Threshold ({noise_std_mult}σ)', linestyle="-.", color='red', alpha=0.3)
        ax1.axhline(noise_dict['signal_power'], label='Signal Power (p95)', linestyle="-.", color='purple', alpha=0.3)

        # Draw visual marks for raw 2nd derivative candidates before filtering/pruning
        if len(filtered_peaks) > 0:
            ax1.scatter(plot_x[filtered_peaks], intensity_axis[filtered_peaks], 
                        color='darkorange', marker='v', s=25, label='Initial Candidates', zorder=4)

        diagnostic_text = f"{area_ratio:.3f}% of Original\nPeaks: {len(results)}"
        ax1.text(
            0.98, 0.95, diagnostic_text,
            horizontalalignment='right', verticalalignment='top',
            transform=ax1.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="lightgray", facecolor="white", alpha=0.8)
        )

        ax1.set_title(f"Multi-Overlapping Peak Fit (m/z ~ {nominal_mass_label})")
        ax1.set_ylabel("Intensity (ions/s)")
        ax1.legend(fontsize='small', loc='upper left', ncol=2 if len(results) > 4 else 1)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- BOTTOM PANEL: Residual error profile ---
        raw_residuals = intensity_axis - fitted_curve
        ax2.plot(plot_x, raw_residuals, color='crimson', lw=1, label='Residuals')
        ax2.axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.7)

        # Shade the noise band based on the calculated noise level
        ax2.fill_between(plot_x, -noise_dict['noise_level'], noise_dict['noise_level'], color='gray', alpha=0.15, label='Noise Band (1σ)')

        ax2.set_ylabel("Residual Error")
        ax2.set_xlabel(x_label)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(fontsize='x-small', loc='lower left')

        plt.tight_layout()
        if save_plot_flag:
            plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
        if show_plot_flag:
            plt.show()
        plt.close()

    return {
        "popt": popt, 
        "pcov": None, 
        "peaks": results,
        "area_original": area_original, 
        "area_fitted": area_fitted, 
        "area_ratio_percent": area_ratio,
        "noise_dict" : noise_dict,
        "smoothing_factor": smooth_factor
    }

def fit_mopf_nm(spectra, n_mass, mass_axis, 
                peak_width_function, 
                peak_type='gaussian', 
                custom_shape=None, 
                nm_search_range=0.5, 
                tol=1e-8, lamda_init=1e-2, max_iter=100,
                smooth_function='inverse linear',
                noise_level=None,
                noise_std_mult=10.0,
                rel_filter=None,
                abs_filter=None,
                smooth_factor=-1,
                plot_flag=False, 
                plot_label='',
                verbose=False):
    """
    Slices a specific nominal mass window from a spectrum and fits overlapping peaks.

    Acts as a wrapper around :func:`multi_overlap_peak_fit`. Extracts local 
    intensity, m/z, and time-of-flight segments centered on `n_mass`, evaluates expected 
    peak width constraints, estimates local baseline noise if omitted, and runs multi-peak 
    deconvolution.

    Parameters
    ----------
    spectra : numpy.ndarray
        1D array of spectral intensity values (ions/s).
    n_mass : int or float
        Target nominal mass integer (m/z) to slice and deconvolute.
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge (m/z) coordinates matching `spectra`.
    peak_width_function : callable
        Callable function that accepts an m/z value and returns expected FWHM peak width.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'} or None, default=None
        Peak shape model identifier passed to :func:`multi_overlap_peak_fit`.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if `peak_type='custom'`.
    plot_flag : bool, default=False
        If ``True``, generates diagnostic 2-panel fit and residual figures for the window.
    nm_search_range : float, default=0.5
        Half-width of the nominal mass search window in m/z units ([n_mass - Δ, n_mass + Δ]).
    tol : float, default=1e-8
        Convergence tolerance passed to the non-linear solver.
    lamda_init : float, default=1e-2
        Initial Levenberg-Marquardt solver damping factor λ.
    max_iter : int, default=100
        Maximum allowable iterations per optimization call.
    smooth_function : {'inverse linear', 'logistic', 'hyperbolic decay', 'double logarithmic', 'logarithmic'}, default='inverse linear'
        Model equation used to compute adaptive Savitzky-Golay window size from SNR.
    rel_filter : float or None, default=None
        Minimum normalized intensity threshold in [0.0, 1.0].
    abs_filter : float or None, default=None
        Minimum absolute intensity threshold (ions/s).
    smooth_factor : int, default=-1
        Savitzky-Golay smoothing window length in sample bins. Calculated dynamically if ``-1``.
    noise_level : float or None, default=None
        Background noise standard deviation (σ). Estimated automatically if ``None``.
    noise_std_mult : float, default=10.0
        Signal-to-noise detection threshold multiplier (N * σ).
    plot_label : str, default=''
        Custom title or identifier string passed for plot labeling.
    verbose : bool, default=False
        If ``True``, prints detailed diagnostic and optimization messages.

    Returns
    -------
    fit_result : dict or None
        Deconvolution result dictionary returned by :func:`multi_overlap_peak_fit`.
    """
    # Slice localized intensity, m/z, and TOF segments centered on the target nominal mass
    int_segment, mz_segment, tof_segment = get_nm_segment_data(
        n_mass, 
        spectra, 
        mass_axis, 
        nm_search_range
    )

    # Evaluate expected peak FWHM resolution constraint for the target nominal mass
    min_separation = peak_width_function(n_mass)

    # Execute multi-peak deconvolution optimization on the sliced local window
    fit_result = multi_overlap_peak_fit(
        mz_segment,
        int_segment,
        tof_axis=tof_segment,
        peak_type=peak_type,
        custom_shape=custom_shape,
        min_separation_fwhm=min_separation,
        max_peaks_per_mq=None,
        rel_filter=rel_filter, 
        abs_filter=abs_filter,
        smooth_factor=smooth_factor,
        smooth_function=smooth_function,
        noise_level=noise_level,
        noise_std_mult=noise_std_mult,
        plot_flag=plot_flag,
        verbose=verbose,
        tol=tol,
        lambda_init=lamda_init,
        max_iter=max_iter,
        plot_label=plot_label
    )

    return fit_result

def fit_partially_constrained_nm(
    mass_axis,
    intensity_axis,
    tof_axis,
    known_peaks,
    n_unknowns=1,
    peak_width_function=None,
    peak_type="gaussian",
    custom_shape=None,
    known_wiggle=0.05,
    unknown_wiggle=2.0,
    noise_level=None,
    noise_std_mult=10.0,
):
    """
    Deconvolute a nominal mass region using hybrid positional constraints.

    Anchors expected chemical target peaks tightly around their theoretical mass coordinates 
    while allowing `n_unknowns` free-floating peaks to discover unassigned signal maxima in 
    the fitting residual.

    Parameters
    ----------
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge (m/z) coordinates for the nominal mass segment.
    intensity_axis : numpy.ndarray
        1D array of spectral intensity values (ions/s) for the local segment.
    tof_axis : numpy.ndarray
        1D array of physical time-of-flight coordinates in nanoseconds.
    known_peaks : list of (str or float)
        Sequence of known target peak chemical formulas or exact m/z floats to anchor tightly.
    n_unknowns : int, default=1
        Maximum number of free-floating unknown peaks to discover in the residual trace.
    peak_width_function : callable or None, default=None
        Callable function mapping m/z to expected FWHM resolution.
    peak_type : {'gaussian', 'lorentzian', 'pseudo_voigt', 'custom'}, default='gaussian'
        Peak shape model identifier.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    known_wiggle : float, default=0.05
        Tight positional wiggle constraint (as a fraction of FWHM) for known target peaks.
    unknown_wiggle : float, default=2.0
        Flexible positional wiggle constraint (as a fraction of FWHM) for discovered unknown peaks.
    noise_level : float or None, default=None
        Background noise standard deviation (σ). Estimated automatically if ``None``.
    noise_std_mult : float, default=10
        Signal-to-noise detection threshold multiplier (N * σ).

    Returns
    -------
    results : list of dict
        A list of dictionaries containing deconvoluted fit results for each peak:
            * ``"label"`` : Peak formula string, float mass label, or ``"Unknown_m.z"``.
            * ``"is_known"`` : Boolean flag indicating if the peak was from `known_peaks`.
            * ``"center_mass"`` : Fitted peak center in m/z space.
            * ``"center_tof"`` : Fitted peak center in time-of-flight space (ns).
            * ``"amplitude"`` : Fitted peak height amplitude (ions/s).
            * ``"fwhm_tof"`` : Fitted Full Width at Half Maximum in time-of-flight units (ns).
    """
    # Retrieve model shape function object
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)

    noise_dict = calculate_detection_threshold(intensity_axis=intensity_axis, 
                                               noise_level=noise_level,
                                               noise_std_mult=noise_std_mult)


    # --- STEP 1: Parse Known Peaks & Formulate Initial Guesses ---
    # Resolve exact theoretical mass for each known compound or float target
    known_masses = [
        return_mass(p) if isinstance(p, str) else float(p) for p in known_peaks
    ]
    
    # Calculate expected FWHM resolution bounds in mass space
    known_fwhms = [peak_width_function(m) for m in known_masses]

    # Map theoretical mass positions to time-of-flight (TOF) coordinates
    known_tof_centers = [
        np.interp(m, mass_axis, tof_axis) for m in known_masses
    ]
    
    # Convert mass-space FWHM bounds to physical TOF FWHM spans (ns)
    known_tof_fwhms = [
        abs(
            np.interp(m + f / 2, mass_axis, tof_axis)
            - np.interp(m - f / 2, mass_axis, tof_axis)
        )
        for m, f in zip(known_masses, known_fwhms)
    ]

    # Sample initial amplitude guesses directly from observed signal at mapped TOF centers
    known_amps = [
        intensity_axis[np.argmin(np.abs(tof_axis - t))]
        for t in known_tof_centers
    ]

    # --- STEP 2: Residual Search for Unassigned (Unknown) Peaks ---
    # Construct synthetic model trace based strictly on known peak initial guesses
    init_trace = np.zeros_like(tof_axis)
    for a, t, w in zip(known_amps, known_tof_centers, known_tof_fwhms):
        init_trace += peak_func(tof_axis, a, t, w)

    # Calculate non-negative residual signal (unassigned spectral intensity)
    residual = np.maximum(intensity_axis - init_trace, 0)

    unknown_tof_centers = []
    unknown_tof_fwhms = []
    unknown_amps = []

    # Use average known peak TOF width as default starting FWHM guess for unknowns
    avg_fwhm_tof = np.mean(known_tof_fwhms) if known_tof_fwhms else 10.0

    # Iteratively locate local maxima in the residual trace
    for _ in range(n_unknowns):
        max_idx = np.argmax(residual)
        u_tof = tof_axis[max_idx]
        u_amp = residual[max_idx]

        # NOISE GATE: Terminate unknown discovery if peak residual height is below detection threshold
        if u_amp < noise_dict['detection_threshold']:
            break

        unknown_tof_centers.append(u_tof)
        unknown_amps.append(u_amp)
        unknown_tof_fwhms.append(avg_fwhm_tof)

        # Mask out surrounding residual region (1.5x average FWHM) to find next distinct peak
        mask_range = int(avg_fwhm_tof * 1.5)
        r_start = max(0, max_idx - mask_range)
        r_end = min(len(residual), max_idx + mask_range)
        residual[r_start:r_end] = 0.0

    # If no candidate peaks passed noise floor and no known targets were supplied, return empty
    if not known_peaks and not unknown_tof_centers:
        return []

    # --- STEP 3: Assemble Hybrid Optimization Inputs ---
    # Concatenate known and discovered unknown vectors
    all_centers = known_tof_centers + unknown_tof_centers
    all_fwhms = known_tof_fwhms + unknown_tof_fwhms
    all_amps = known_amps + unknown_amps

    # Assign tight wiggle bounds for known targets and flexible wiggle bounds for unknown discovery
    all_wiggles = [known_wiggle] * len(known_peaks) + [
        unknown_wiggle
    ] * len(unknown_tof_centers)

    # Normalize amplitude guesses to scale TRF Jacobian conditioning
    max_sig = np.max(intensity_axis)
    norm_sig = intensity_axis / (max_sig if max_sig > 0 else 1.0)
    norm_amps = [a / (max_sig if max_sig > 0 else 1.0) for a in all_amps]

    # --- STEP 4: Non-Linear Optimization ---
    # Execute simultaneous non-linear least squares fit with hybrid bounds
    popt = fit_unconstrained_peaks(
        x_axis=tof_axis,
        signal=norm_sig,
        centers_guess=all_centers,
        fwhms_guess=all_fwhms,
        amplitudes_guess=norm_amps,
        peak_type=peak_type,
        custom_shape=custom_shape,
        center_wiggle=all_wiggles,
        max_iter=1000,
    )

    # --- STEP 5: Extract and Package Results ---
    n_p = 4 if peak_type == "pseudo_voigt" else 3
    results = []

    # Construct 1D interpolator mapping fitted TOF centers back to m/z space
    tof_to_mass = interp1d(tof_axis, mass_axis, fill_value="extrapolate")

    for i in range(len(all_centers)):
        p = popt[i * n_p : (i + 1) * n_p]
        amp = p[0] * max_sig
        c_tof = p[1]
        f_tof = p[2]
        c_mass = float(tof_to_mass(c_tof))

        is_known = i < len(known_peaks)
        results.append({
            "label": known_peaks[i] if is_known else f"Unknown_{c_mass:.4f}",
            "is_known": is_known,
            "center_mass": c_mass,
            "center_tof": c_tof,
            "amplitude": amp,
            "fwhm_tof": f_tof,
        })

    return results


def unpack_fit_result(fit_result, list_container, ms_index, nominal_mass):
    """
    Unpack and reformat nominal mass fit results into standardized dictionary rows.

    Parses the deconvolution output dictionary from peak fitting routines (e.g., 
    :func:`multi_overlap_peak_fit`), extracts individual peak parameters and region 
    summary metrics, converts types to standard floats/ints/Nones, and appends the 
    resulting row dictionaries directly to `list_container`. Handles three fit outcomes:
    valid peaks resolved, zero peaks resolved (region metadata preserved), and complete 
    fit failure (filler row appended).

    Parameters
    ----------
    fit_result : dict or None
        Deconvolution result dictionary returned by peak fitting functions, or ``None`` 
        if fitting failed completely. Expected keys include ``"peaks"``, ``"area_original"``, 
        ``"area_fitted"``, ``"area_ratio_percent"``, ``"snr"``, ``"noise_power"``, 
        ``"signal_power"``, and ``"smoothing_factor"``.
    list_container : list
        Mutable Python list container to which generated dictionary rows are appended in-place.
    ms_index : int
        0-based index of the target mass spectrum / writebuf.
    nominal_mass : int or float
        Target nominal mass integer (m/z) corresponding to the fitted spectral region.

    Returns
    -------
    None
        Appends dictionary entries to `list_container` in-place.
    """
    # Case 1: Valid fit_result dictionary returned by peak fitting routine
    if fit_result is not None:
        # Sub-case 1a: At least one peak was successfully deconvoluted in this window
        if fit_result['peaks']:
            for peak in fit_result['peaks']:
                row = {
                    'MS_index': ms_index,
                    'nominal_mass': nominal_mass,
                    'peak_center_mass': float(peak['center_mass']),
                    'peak_center_tof': float(peak['center_tof']),
                    'amplitude': float(peak['amplitude']),
                    'fwhm_mass': float(peak['fwhm_mass']),
                    'fwhm_tof': float(peak['fwhm_tof']),
                    'peak_area': float(peak['area']),
                    'nm_area_original': float(fit_result['area_original']),
                    'nm_area_fitted': float(fit_result['area_fitted']),
                    'nm_area_ratio_percent': float(fit_result['area_ratio_percent']),
                    'n_peaks': len(fit_result['peaks']),
                    'snr': float(fit_result['snr']),
                    'noise_power': float(fit_result['noise_power']),
                    'signal_power': float(fit_result['signal_power']),
                    'smoothing_factor': float(fit_result['smoothing_factor']),
                }
                list_container.append(row)
        else:
            # Sub-case 1b: Region was evaluated but zero peaks passed thresholding (filler row preserving region metadata)
            row = {
                'MS_index': ms_index,
                'nominal_mass': nominal_mass,
                'peak_center_mass': None,
                'peak_center_tof': None,
                'amplitude': None,
                'fwhm_mass': None,
                'fwhm_tof': None,
                'peak_area': None,
                'nm_area_original': float(fit_result['area_original']),
                'nm_area_fitted': float(fit_result['area_fitted']),
                'nm_area_ratio_percent': float(fit_result['area_ratio_percent']),
                'n_peaks': 0,
                'snr': float(fit_result['snr']),
                'noise_power': float(fit_result['noise_power']),
                'signal_power': float(fit_result['signal_power']),
                'smoothing_factor': float(fit_result['smoothing_factor']),
            }
            list_container.append(row)
    else:
        # Case 2: Complete fit failure (fit_result is None); append filler row with None values
        row = {
            'MS_index': ms_index,
            'nominal_mass': None,
            'peak_center_mass': None,
            'peak_center_tof': None,
            'amplitude': None,
            'fwhm_mass': None,
            'fwhm_tof': None,
            'peak_area': None,
            'nm_area_original': None,
            'nm_area_fitted': None,
            'nm_area_ratio_percent': None,
            'n_peaks': 0,
            'snr': None,
            'noise_power': None,
            'signal_power': None,
            'smoothing_factor': None,
        }
        list_container.append(row)

def _unconstrained_map_worker(chunk, block_info=None, **kwargs):
    """
    Parallel worker function executed via Dask `map_blocks` for unconstrained peak fitting.

    Processes a 2D chunk block of spectral data, fitting unconstrained peaks (allowing 
    amplitude, center, and width to vary) for each nominal mass window using 
    Trust Region Reflective (TRF) non-linear least squares (:func:`scipy.optimize.least_squares`). 
    Supports exact analytical Jacobians for Gaussian and empirical custom shapes, fast noise-gating, 
    and sequential vectorized isotope subtraction to prevent lower-mass isotopic interference.

    Parameters
    ----------
    chunk : numpy.ndarray
        2D array slice of time-of-flight spectral intensity data of shape ``(chunk_spectra, num_samples)``.
    block_info : dict or None, default=None
        Dask metadata dictionary providing block array location indices.
    **kwargs : dict
        Keyword arguments passed from :func:`full_fitting_integration_unconstrained`:
            * ``calibration_results`` (dict): Mass calibration parameter dictionary.
            * ``unique_nms`` (list of int): Sorted list of unique nominal mass integers.
            * ``nm_to_peaks`` (dict): Mapping of nominal mass to target peak formulas/labels.
            * ``precomputed_isotopes`` (dict): Pre-calculated theoretical isotope distributions.
            * ``peak_list`` (list): Master ordered peak target list.
            * ``peak_type`` (str): Model peak shape identifier.
            * ``custom_shape`` (callable or None): Custom shape function if ``peak_type='custom'``.
            * ``sample_index_axis`` (numpy.ndarray): 1D array of sample index channels.
            * ``tof_axis`` (numpy.ndarray): 1D physical TOF axis in nanoseconds.
            * ``peak_width_function`` (callable): Function mapping m/z to expected FWHM.
            * ``nm_search_range`` (float): Half-width search window in m/z units.
            * ``max_iter`` (int): Maximum optimizer function evaluations per peak group.
            * ``tol`` (float): Convergence tolerance for least-squares solver.
            * ``noise_level`` (float): standard deviations of noise regions along intensity axis, if None estimated automatically
            * ``noise_std_mult`` (float): multiplier applied to the noise_level to determine detection threshold

    Returns
    -------
    results : numpy.ndarray
        2D array of shape ``(chunk_spectra, 1 + 4 * num_peaks)`` containing the global 
        `MS_index` followed by 4 output parameters per peak: ``[amplitude, center_mass, fwhm_mass, area]``.
    """
    from opentof.mass_calibration import apply_mass_calibration
    from opentof.isotopes import isotope_signal_on_axis

    # Extract starting spectrum index for this specific Dask chunk block
    start_idx = block_info[0]['array-location'][0][0]
    n_spectra = chunk.shape[0]

    # Unpack configuration parameters from kwargs
    cal_res = kwargs['calibration_results']
    unique_nms = kwargs['unique_nms']
    nm_to_peaks = kwargs['nm_to_peaks']
    precomputed_isotopes = kwargs['precomputed_isotopes']
    peak_list = list(kwargs['peak_list'])
    peak_type = kwargs['peak_type']
    custom_shape = kwargs.get('custom_shape', None)
    sample_index_axis = kwargs['sample_index_axis']
    tof_axis = kwargs['tof_axis']
    peak_width_function = kwargs['peak_width_function']
    nm_search_range = kwargs.get('nm_search_range', 0.5)
    max_iter = kwargs.get('max_iter', 25)
    tol = kwargs.get('tol', 1e-3)
    noise_level = kwargs.get('noise_level', None)
    noise_std_mult = kwargs.get('noise_std_mult', 10.0)

    # --- STEP 1: Custom Shape Grid & Derivative Extraction ---
    # Extract underlying 1D sampling grids (x, y) and calculate numeric 1st derivative (dy/dx) for analytical Jacobian
    x_grid, y_grid, dy_grid = None, None, None
    if custom_shape is not None:
        if hasattr(custom_shape, '__self__') and hasattr(custom_shape.__self__, 'x'):
            x_grid = custom_shape.__self__.x
            y_grid = custom_shape.__self__.y
        elif hasattr(custom_shape, 'x_grid'):
            x_grid = custom_shape.x_grid
            y_grid = custom_shape.y_grid
        if x_grid is not None and y_grid is not None:
            dy_grid = np.gradient(y_grid, x_grid)

    # Determine parameter stride count per peak (4 for Pseudo-Voigt, 3 for Gaussian/Custom)
    is_pv = (peak_type == 'pseudo_voigt')
    n_p = 4 if is_pv else 3
    
    # Retrieve peak function selector and check for empirical area constant ps_sigma
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)
    is_custom = (peak_type == 'custom' and hasattr(peak_func, 'ps_sigma'))
    ps_sigma = peak_func.ps_sigma if is_custom else 0.0
    gauss_const = np.sqrt(np.pi / (4 * np.log(2)))  # Constant factor for Gaussian analytical integration

    # Scaling constants for analytical derivatives
    c_gauss = 2.772588722239781  # 4 * ln(2)
    scale_ps = 2.3548200450309493 # Conversion factor between FWHM and standard deviation (sigma)

    # --- STEP 2: Analytical Model & Exact Jacobian Evaluators ---
    if is_custom and x_grid is not None and y_grid is not None and dy_grid is not None:
        # Exact Analytical Jacobian evaluator for empirical custom spline shapes via chain rule
        def eval_model_and_jac(p_params, x_data):
            n_peaks = len(p_params) // n_p
            mod = np.zeros_like(x_data, dtype=np.float64)
            J = np.zeros((len(x_data), len(p_params)), dtype=np.float64)
            for pk_i in range(n_peaks):
                A, xc, w = p_params[pk_i * 3 : pk_i * 3 + 3]
                u = (x_data - xc) * scale_ps / w
                
                # Interpolate shape intensity (y) and slope derivative (dy) on standardized domain
                y_val = np.interp(u, x_grid, y_grid, left=0.0, right=0.0)
                dy_val = np.interp(u, x_grid, dy_grid, left=0.0, right=0.0)
                
                mod += A * y_val
                
                # Partial derivatives wrt A, xc, and FWHM (w)
                col_base = pk_i * 3
                J[:, col_base] = y_val
                J[:, col_base + 1] = -A * dy_val * (scale_ps / w)
                J[:, col_base + 2] = -A * dy_val * (scale_ps * (x_data - xc) / (w * w))
            return mod, J

    elif peak_type == 'gaussian':
        # Exact Analytical Jacobian evaluator for standard Gaussian profiles
        def eval_model_and_jac(p_params, x_data):
            n_peaks = len(p_params) // n_p
            mod = np.zeros_like(x_data, dtype=np.float64)
            J = np.zeros((len(x_data), len(p_params)), dtype=np.float64)
            for pk_i in range(n_peaks):
                A, xc, w = p_params[pk_i * 3 : pk_i * 3 + 3]
                dx = x_data - xc
                u = dx / w
                E = np.exp(-c_gauss * u**2)
                f_pk = A * E
                mod += f_pk
                
                # Partial derivatives wrt A, xc, and FWHM (w)
                col_base = pk_i * 3
                J[:, col_base] = E
                J[:, col_base + 1] = f_pk * (2.0 * c_gauss * dx / (w**2))
                J[:, col_base + 2] = f_pk * (2.0 * c_gauss * dx**2 / (w**3))
            return mod, J

    else:
        # Fallback numeric finite-difference Jacobian evaluator (Pseudo-Voigt or arbitrary models)
        def eval_model_and_jac(p_params, x_data):
            n_peaks = len(p_params) // n_p
            mod = np.zeros_like(x_data, dtype=np.float64)
            for pk_i in range(n_peaks):
                p = p_params[pk_i * n_p : (pk_i + 1) * n_p]
                if is_pv:
                    mod += peak_func(x_data, p[0], p[1], p[2], p[3])
                else:
                    mod += peak_func(x_data, p[0], p[1], p[2])
            return mod, None

    # --- STEP 3: Setup Output Mapping Matrix ---
    # Map each target peak formula to its starting column in the output array
    peak_to_col = {p: 1 + 4 * i for i, p in enumerate(peak_list)}
    out_width = 1 + 4 * len(peak_list)
    results = np.zeros((n_spectra, out_width), dtype=np.float64)
    results[:, 0] = np.arange(start_idx, start_idx + n_spectra) # Column 0 = Global MS_index

    # Identify unique calibration intervals present within this spectrum chunk
    interval_indices = cal_res['interval_indices'][start_idx : start_idx + n_spectra]
    unique_intervals = np.unique(interval_indices)

    # --- STEP 4: Process Block by Calibration Interval ---
    for interval_id in unique_intervals:
        if np.isnan(interval_id):
            continue
            
        # Isolate spectra belonging to current interval
        local_mask = (interval_indices == interval_id)
        local_row_indices = np.where(local_mask)[0]
        spectra_block = chunk[local_mask, :].copy()

        # Reconstruct mass axis for active interval
        params = cal_res['batch_params'][int(interval_id)]
        mode = cal_res.get('mass_cal_mode', 2)
        mass_axis = apply_mass_calibration(sample_index_axis, params, mode=mode)

        # Iterate over nominal mass groups in ascending order
        for nm in unique_nms:
            mask = (mass_axis >= (nm - nm_search_range)) & (mass_axis <= (nm + nm_search_range))
            if not np.any(mask):
                continue

            mz_seg = mass_axis[mask]
            tof_seg = tof_axis[mask]
            nm_peaks = nm_to_peaks[nm]
            n_nm_peaks = len(nm_peaks)

            # Pre-calculate initial parameter guesses and box bounds once per NM window
            nm_peaks_data = []
            lower_bounds, upper_bounds = [], []

            for peak in nm_peaks:
                peak_pos = return_mass(peak)
                xc_guess = np.interp(peak_pos, mz_seg, tof_seg)
                
                f_mass = peak_width_function(peak_pos)
                if peak_type == 'custom' and hasattr(peak_func, 'gauss_to_ps'):
                    f_mass *= peak_func.gauss_to_ps

                f_tof_guess = abs(np.interp(peak_pos + f_mass / 2, mz_seg, tof_seg) - 
                                  np.interp(peak_pos - f_mass / 2, mz_seg, tof_seg))
                xc_idx = np.argmin(np.abs(tof_seg - xc_guess))

                nm_peaks_data.append({
                    'xc': xc_guess,
                    'f_tof': f_tof_guess,
                    'f_mass': f_mass,
                    'xc_idx': xc_idx,
                    'peak': peak,
                    'col': peak_to_col[peak]
                })

                # Parameter bounds: Amplitude >= 0, Center within +/- 2.0 FWHM, FWHM within [0.2x, 5.0x] guess
                lower_bounds.extend([0.0, xc_guess - 2.0 * f_tof_guess, 0.2 * f_tof_guess])
                upper_bounds.extend([np.inf, xc_guess + 2.0 * f_tof_guess, 5.0 * f_tof_guess])
                if is_pv:
                    lower_bounds.append(0.0)
                    upper_bounds.append(1.0)

            bounds_tuple = (lower_bounds, upper_bounds)
            block_amplitudes = np.zeros((spectra_block.shape[0], n_nm_peaks), dtype=np.float64)

            # Residual & Jacobian solver callbacks
            def segment_residuals(p_params, x_data, y_norm):
                mod, _ = eval_model_and_jac(p_params, x_data)
                return mod - y_norm

            def segment_jacobian(p_params, x_data, y_norm):
                _, J = eval_model_and_jac(p_params, x_data)
                return J

            # --- STEP 5: Spectrum Fit Loop ---
            for s_idx in range(spectra_block.shape[0]):
                int_seg = spectra_block[s_idx, mask]
                int_max = np.max(int_seg)

                noise_dict = calculate_detection_threshold(intensity_axis=int_seg, 
                                                           noise_level=noise_level, 
                                                           noise_std_mult=noise_std_mult)

                if int_max <= noise_dict['detection_threshold']:
                    continue
                if int_max <= 0:
                    continue

                int_norm = int_seg / int_max

                # Formulate starting parameter guesses for this spectrum
                x0 = []
                for p_info in nm_peaks_data:
                    a_g = max(int_norm[p_info['xc_idx']], 1e-4)
                    x0.extend([a_g, p_info['xc'], p_info['f_tof']])
                    if is_pv:
                        x0.append(0.5)

                # Solve non-linear least squares using TRF algorithm and exact Jacobian
                try:
                    _, J_test = eval_model_and_jac(x0, tof_seg)
                    jac_callback = segment_jacobian if J_test is not None else '2-point'

                    res = least_squares(
                        segment_residuals,
                        x0=x0,
                        args=(tof_seg, int_norm),
                        jac=jac_callback,
                        bounds=bounds_tuple,
                        method='trf',
                        ftol=tol,
                        xtol=tol,
                        gtol=tol,
                        max_nfev=max_iter
                    )
                    popt = res.x
                except Exception:
                    popt = np.array(x0, dtype=np.float64)

                # Unpack fitted parameters and calculate area integrals
                for i_p, p_info in enumerate(nm_peaks_data):
                    A_norm, x_c, FWHM = popt[i_p * n_p : i_p * n_p + 3]
                    A = max(A_norm * int_max, 0.0)
                    block_amplitudes[s_idx, i_p] = A

                    # Convert fitted TOF coordinates back to m/z space
                    c_mass = np.interp(x_c, tof_seg, mz_seg)
                    f_mass = abs(np.interp(x_c + FWHM / 2, tof_seg, mz_seg) - 
                                 np.interp(x_c - FWHM / 2, tof_seg, mz_seg))

                    # Calculate analytical integrated area by peak model type
                    if is_custom:
                        area = A * (FWHM / 2.3548) * ps_sigma
                    elif peak_type == 'gaussian':
                        area = A * FWHM * gauss_const
                    elif peak_type == 'lorentzian':
                        area = A * (np.pi * FWHM / 2)
                    else:
                        area = A * (FWHM / 2.3548) * 2.5066

                    # Save parameters [amplitude, center_mass, fwhm_mass, area] to results array
                    col_idx = p_info['col']
                    results[local_row_indices[s_idx], col_idx : col_idx + 4] = [A, c_mass, f_mass, area]

            # --- STEP 6: Vectorized Isotope Subtraction ---
            # Subtract calculated isotope profiles of fitted peaks from higher m/z regions
            for i_p, peak in enumerate(nm_peaks):
                A_array = block_amplitudes[:, i_p]
                if isinstance(peak, str) and np.any(A_array > 0):
                    p_m = return_mass(peak)
                    sub_mask = (mass_axis >= p_m) & (mass_axis <= p_m + 10.0)
                    if np.any(sub_mask):
                        unit_pattern = isotope_signal_on_axis(
                            formula=peak,
                            mass_axis=mass_axis[sub_mask],
                            parent_amplitude=1.0,
                            peak_width_function=peak_width_function,
                            peak_type=peak_type,
                            custom_shape=peak_func if peak_type == 'custom' else None,
                            precomputed_pattern=precomputed_isotopes.get(peak)
                        )
                        # Subtract isotope contribution across all chunk spectra simultaneously
                        spectra_block[:, sub_mask] -= (A_array[:, np.newaxis] * unit_pattern)
                        spectra_block[:, sub_mask] = np.maximum(spectra_block[:, sub_mask], 0)

    return results

def full_fitting_integration_unconstrained(
        peak_list, tofdata_subtracted, calibration_results, 
        sample_index_axis, tof_axis, peak_width_function,
        peak_type='pseudo_voigt', custom_shape=None,
        nm_search_range=0.5, verbose=True, max_iter=25, tol=1e-4,
        chunk_size=1000, noise_level=None, noise_std_mult=10, **kwargs
    ):
    """
    Perform parallel unconstrained peak fitting and integration over an entire dataset using Dask `map_blocks`.

    Orchestrates out-of-core parallel peak deconvolution across all spectra in `tofdata_subtracted`. 
    Clusters peak targets by nominal mass integer, pre-computes theoretical isotope patterns, 
    re-chunks Dask arrays for optimal RAM usage, dispatches parallel execution via 
    :func:`_unconstrained_map_worker`, and returns a structured pandas DataFrame.

    Parameters
    ----------
    peak_list : list of (str or float)
        List of target peak chemical formulas or m/z values to fit.
    tofdata_subtracted : dask.array.Array
        2D baseline-subtracted time-of-flight spectral data array of shape ``(num_spectra, num_samples)``.
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`.
    sample_index_axis : numpy.ndarray
        1D array of sample index channel positions.
    tof_axis : numpy.ndarray
        1D physical time-of-flight axis in nanoseconds.
    peak_width_function : callable
        Callable function mapping m/z to expected peak FWHM resolution.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
        Peak shape model identifier.
    custom_shape : callable or None, default=None
        Custom shape callable required if ``peak_type='custom'``.
    nm_search_range : float, default=0.5
        Half-width of nominal mass search windows in m/z units ([n_mass - Δ, n_mass + Δ]).
    verbose : bool, default=True
        If ``True``, displays progress bars and diagnostic console messages.
    max_iter : int, default=25
        Maximum optimization iterations per peak group per spectrum.
    tol : float, default=1e-3
        Convergence tolerance passed to the TRF solver.
    chunk_size : int, default=1000
        Number of spectra per chunk block for parallel computation.
    noise_level : float or None, default=None
        Background noise standard deviation (σ) for fast noise-gating.
    noise_std_mult : float, default=10.0
        Muliplier for noise_level that determines threshold for peak detection.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pandas.DataFrame
        Structured DataFrame containing a `MS_index` column followed by 4 output 
        columns per peak target: ``f"{peak}_amplitude"``, ``f"{peak}_center_mass"``, 
        ``f"{peak}_fwhm_mass"``, and ``f"{peak}_area"``.
    """
    from .isotopes import batch_isotope_masses
    # Group target peaks by nominal mass integer
    nm_to_peaks = {}
    for peak in peak_list:
        nm = int(np.round(return_mass(peak)))
        nm_to_peaks.setdefault(nm, []).append(peak)
    unique_nms = sorted(nm_to_peaks.keys())

    # Pre-calculate theoretical isotope distributions for all formula peaks
    precomputed_isotopes = batch_isotope_masses(peak_list, cutoff=1e-4)

    # Re-chunk Dask array to match requested processing chunk size
    if chunk_size:
        tofdata_subtracted = tofdata_subtracted.rechunk({0: chunk_size})

    # Define output DataFrame column headers
    out_cols = ["MS_index"]
    for p in peak_list:
        out_cols.extend([f"{p}_amplitude", f"{p}_center_mass", f"{p}_fwhm_mass", f"{p}_area"])

    if verbose:
        print(f"Parallelizing unconstrained fit over {len(peak_list)} peaks across {tofdata_subtracted.shape[0]} spectra via Dask...")

    # Execute worker function in parallel across Dask blocks
    with ProgressBar() if verbose else dask.config.set():
        final_array = tofdata_subtracted.map_blocks(
            _unconstrained_map_worker,
            calibration_results=calibration_results,
            unique_nms=unique_nms,
            nm_to_peaks=nm_to_peaks,
            precomputed_isotopes=precomputed_isotopes,
            peak_list=peak_list,
            peak_type=peak_type,
            custom_shape=custom_shape,
            sample_index_axis=sample_index_axis,
            tof_axis=tof_axis,
            peak_width_function=peak_width_function,
            nm_search_range=nm_search_range,
            max_iter=max_iter,
            tol=tol,
            noise_level=noise_level,
            noise_std_mult=noise_std_mult,
            dtype='float64',
            chunks=(tofdata_subtracted.chunks[0], len(out_cols))
        ).compute()

    # Package output matrix into structured pandas DataFrame
    return pd.DataFrame(final_array.reshape(-1, len(out_cols)), columns=out_cols)

def _get_basis_matrix(x_axis, peak_centers, fwhms, peak_func, peak_type):
    """
    Pre-calculate unit-amplitude basis matrix M for a constrained peak group.

    Constructs a 2D basis matrix where each column q represents the normalized 
    unit-height line shape of candidate peak q evaluated over coordinate vector `x_axis`.

    Parameters
    ----------
    x_axis : numpy.ndarray
        1D array of independent coordinate values (e.g., m/z or TOF in ns).
    peak_centers : sequence of float
        1D sequence of fixed target peak centers x_c.
    fwhms : sequence of float
        1D sequence of fixed target peak Full Width at Half Maximum (FWHM) values.
    peak_func : callable
        Callable model shape function (e.g., Gaussian, Pseudo-Voigt, or empirical custom shape).
    peak_type : str
        Peak shape identifier string ('pseudo_voigt', 'gaussian', 'lorentzian', or 'custom').

    Returns
    -------
    M : numpy.ndarray
        2D basis matrix of shape ``(len(x_axis), len(peak_centers))`` where column q 
        contains the unit-amplitude model shape for peak q.
    """
    n_points = len(x_axis)
    n_peaks = len(peak_centers)
    
    # Pre-allocate basis matrix M of shape (N_coordinate_points, N_peaks)
    M = np.zeros((n_points, n_peaks), dtype=np.float64)
    
    # Populate each column q with the unit-amplitude line shape evaluated at (center, FWHM)
    for q in range(n_peaks):
        if peak_type == 'pseudo_voigt':
            # Evaluate Pseudo-Voigt profile with unit amplitude (1.0) and default 50/50 mixing (0.5)
            M[:, q] = peak_func(x_axis, 1.0, peak_centers[q], fwhms[q], 0.5)
        else:
            # Evaluate Gaussian, Lorentzian, or empirical custom shape with unit amplitude (1.0)
            M[:, q] = peak_func(x_axis, 1.0, peak_centers[q], fwhms[q])
            
    return M

def _solve_nnls_block(M, signal_block):
    """
    Solve Non-Negative Least Squares (NNLS) for a 2D block of spectra against basis matrix M.

    Solves the constrained optimization problem minimizeing ||M * h - y_i||_2 for h >= 0
    for each spectrum vector y_i in `signal_block` using :func:`scipy.optimize.nnls`.

    Parameters
    ----------
    M : numpy.ndarray
        2D basis matrix of shape ``(num_bins, num_peaks)`` generated by :func:`_get_basis_matrix`.
    signal_block : numpy.ndarray
        2D array of intensity spectra of shape ``(num_spectra, num_bins)``.

    Returns
    -------
    heights : numpy.ndarray
        2D array of fitted peak heights/amplitudes of shape ``(num_spectra, num_peaks)``.
    """
    n_spectra = signal_block.shape[0]
    n_peaks = M.shape[1]
    
    # Pre-allocate output height matrix of shape (N_spectra, N_peaks)
    heights = np.zeros((n_spectra, n_peaks), dtype=np.float64)
    
    # Solve NNLS independently for each spectrum row in the block
    for i in range(n_spectra):
        # Solves h >= 0 such that ||M * h - y_i||_2 is minimized
        h, _ = nnls(M, signal_block[i])
        heights[i] = h
        
    return heights

def _constrained_map_worker(chunk, block_info=None, **kwargs):
    """
    Parallel worker function executed via Dask `map_blocks` for fully constrained peak fitting.

    Processes a 2D chunk block of spectral data by applying mass calibration interval mapping, 
    constructing unit-amplitude basis matrices (M), solving Non-Negative Least Squares (NNLS) 
    for peak amplitudes, computing analytical integrated peak areas, and performing 
    vectorized sequential isotope subtraction to remove downstream isotopic interferences.

    Parameters
    ----------
    chunk : numpy.ndarray
        2D array slice of time-of-flight spectral intensity data of shape ``(chunk_spectra, num_samples)``.
    block_info : dict or None, default=None
        Dask metadata dictionary providing block array location indices.
    **kwargs : dict
        Keyword arguments passed from :func:`full_fitting_integration_constrained`:
            * ``calibration_results`` (dict): Mass calibration parameter dictionary.
            * ``unique_nms`` (list of int): Sorted list of unique nominal mass integers.
            * ``nm_to_peaks`` (dict): Mapping of nominal mass to target peak formulas/labels.
            * ``precomputed_nm_data`` (dict): Pre-calculated peak centers and FWHMs per nominal mass.
            * ``precomputed_isotopes`` (dict): Pre-calculated theoretical isotope distributions.
            * ``peak_list`` (list): Master ordered peak target list.
            * ``peak_type`` (str): Model peak shape identifier.
            * ``custom_shape`` (callable or None): Custom shape function if ``peak_type='custom'``.
            * ``sample_index_axis`` (numpy.ndarray): 1D array of sample index channels.
            * ``tof_axis`` (numpy.ndarray): 1D physical TOF axis in nanoseconds.
            * ``nm_search_range`` (float): Half-width search window in m/z units.
            * ``peak_width_function`` (callable): Function mapping m/z to expected FWHM.

    Returns
    -------
    results : numpy.ndarray
        2D array of shape ``(chunk_spectra, 1 + 2 * num_peaks)`` containing the global 
        `MS_index` followed by 2 output parameters per peak: ``[amplitude, area]``.
    """
    from opentof.mass_calibration import apply_mass_calibration
    from opentof.isotopes import isotope_signal_on_axis

    # Extract starting spectrum index for this specific Dask chunk block
    start_idx = block_info[0]['array-location'][0][0]
    n_spectra = chunk.shape[0]

    # Unpack shared parameters passed from the orchestrator function
    cal_results = kwargs['calibration_results']
    unique_nms = kwargs['unique_nms']
    nm_to_peaks = kwargs['nm_to_peaks']
    precomputed_nm_data = kwargs['precomputed_nm_data']
    precomputed_isotopes = kwargs['precomputed_isotopes']
    peak_list = kwargs['peak_list']
    peak_type = kwargs['peak_type']
    custom_shape = kwargs.get('custom_shape', None)
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)
    sample_index_axis = kwargs['sample_index_axis']
    tof_axis = kwargs['tof_axis']
    nm_search_range = kwargs.get('nm_search_range', 0.5)
    peak_width_function = kwargs['peak_width_function']

    # Pre-allocate output matrix: Column 0 = MS_index, followed by 2 columns (Amplitude, Area) per peak
    out_width = 1 + 2 * len(peak_list)
    results = np.zeros((n_spectra, out_width), dtype=np.float64)
    results[:, 0] = np.arange(start_idx, start_idx + n_spectra)
    
    # Identify unique calibration intervals active within this specific spectrum block
    interval_indices = cal_results['interval_indices'][start_idx : start_idx + n_spectra]
    unique_intervals_in_chunk = np.unique(interval_indices)

    # --- STEP 1: Process Block by Calibration Interval ---
    for interval_id in unique_intervals_in_chunk:
        if np.isnan(interval_id): 
            continue
            
        # Create a mutable copy of spectra belonging to current interval
        local_mask = (interval_indices == interval_id)
        spectra_block = chunk[local_mask, :].copy()

        # Reconstruct calibrated mass axis for active interval
        params = cal_results['batch_params'][int(interval_id)]
        mode = cal_results.get('mass_cal_mode', 2)
        mass_axis = apply_mass_calibration(sample_index_axis, params, mode=mode)

        # --- STEP 2: Process Nominal Mass Windows ---
        for nm in unique_nms:
            peak_centers, peak_fwhms = precomputed_nm_data[nm]

            # Slice local m/z window mask around target nominal mass
            mask = (mass_axis >= (nm - nm_search_range)) & (mass_axis <= (nm + nm_search_range))
            if not np.any(mask): 
                continue

            mz_segment = mass_axis[mask]
            int_segment_block = spectra_block[:, mask]

            # Construct unit-amplitude basis matrix M of shape (num_bins, num_window_peaks)
            M = _get_basis_matrix(mz_segment, peak_centers, peak_fwhms, peak_func, peak_type)
            
            # Solve Non-Negative Least Squares (NNLS) to resolve peak amplitudes across block
            heights_block = _solve_nnls_block(M, int_segment_block)

            # --- STEP 3: Analytical Area Integration & Output Storage ---
            for i, peak in enumerate(nm_to_peaks[nm]):
                A_array = heights_block[:, i] 
                center_mass = peak_centers[i]
                fwhm_mass = peak_fwhms[i]
                
                # Convert mass-space FWHM to time-of-flight FWHM (in ns) via local axis interpolation
                fwhm_tof = abs(np.interp(center_mass + fwhm_mass / 2, mz_segment, tof_axis[mask]) - 
                               np.interp(center_mass - fwhm_mass / 2, mz_segment, tof_axis[mask]))
                t_sigma = fwhm_tof / 2.3548200450309493  # Convert FWHM to Gaussian sigma
                
                # Calculate analytical integrated peak areas based on model shape
                if peak_type == 'custom' and hasattr(peak_func, 'ps_sigma'):
                    areas = A_array * t_sigma * peak_func.ps_sigma
                elif peak_type == 'gaussian':
                    areas = A_array * fwhm_tof * np.sqrt(np.pi / (4 * np.log(2)))
                elif peak_type == 'lorentzian':
                    areas = A_array * (np.pi * fwhm_tof / 2)
                else:
                    # Fallback analytical Gaussian integral approximation
                    areas = A_array * t_sigma * 2.5066282746310002

                # Identify target output column indices for current peak
                peak_idx_in_list = list(peak_list).index(peak)
                col_idx = 1 + 2 * peak_idx_in_list
                
                # Store fitted amplitudes and integrated areas into results array
                results[local_mask, col_idx] = A_array
                results[local_mask, col_idx + 1] = areas

                # --- STEP 4: Vectorized Isotope Subtraction ---
                # Subtract calculated isotope profiles of fitted parents from higher m/z channels
                if isinstance(peak, str) and np.any(A_array > 0):
                    isotope_tail_buffer = 10.0  # m/z window buffer to evaluate downstream isotopes
                    sub_mask = (mass_axis >= center_mass) & (mass_axis <= center_mass + isotope_tail_buffer)
                    
                    if np.any(sub_mask):
                        # Generate normalized unit-amplitude isotope signal pattern
                        unit_pattern = isotope_signal_on_axis(
                            formula=peak,
                            mass_axis=mass_axis[sub_mask],
                            parent_amplitude=1.0,
                            peak_width_function=peak_width_function,
                            peak_type=peak_type,
                            custom_shape=peak_func if peak_type == 'custom' else None,
                            precomputed_pattern=precomputed_isotopes.get(peak)
                        )
                        # Broadcast subtraction across all block spectra: (N_spectra, 1) * (1, N_bins)
                        spectra_block[:, sub_mask] -= (A_array[:, np.newaxis] * unit_pattern)
                        
                        # Constrain baseline intensities to non-negative values
                        spectra_block[:, sub_mask] = np.maximum(spectra_block[:, sub_mask], 0)

    return results

def full_fitting_integration_constrained(
        peak_list, tofdata_subtracted, calibration_results, 
        sample_index_axis, tof_axis, peak_width_function,
        peak_type='pseudo_voigt', custom_shape=None, 
        nm_search_range=0.5, verbose=True,
        chunk_size=1000,
    ):
    """
    Perform parallel fully constrained peak fitting and integration over a dataset using Dask `map_blocks`.

    Orchestrates out-of-core parallel peak deconvolution where peak positions (m/z) 
    and widths (FWHM) are fixed, allowing only peak amplitudes to vary. Solves amplitudes 
    via Non-Negative Least Squares (NNLS) and sequentially subtracts calculated isotope 
    contributions in ascending m/z order to eliminate downstream isotopic interferences.

    Parameters
    ----------
    peak_list : list of (str or float)
        List of target peak chemical formulas or m/z values to fit.
    tofdata_subtracted : dask.array.Array
        2D baseline-subtracted time-of-flight spectral data array of shape ``(num_spectra, num_samples)``.
    calibration_results : dict
        Results dictionary returned by :func:`run_mass_calibration`.
    sample_index_axis : numpy.ndarray
        1D array of sample index channel positions.
    tof_axis : numpy.ndarray
        1D physical time-of-flight axis in nanoseconds.
    peak_width_function : callable
        Callable function mapping m/z to expected peak FWHM resolution.
    peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
        Peak shape model identifier.
    custom_shape : callable or None, default=None
        Custom shape callable required if ``peak_type='custom'``.
    nm_search_range : float, default=0.5
        Half-width of nominal mass search windows in m/z units ([n_mass - Δ, n_mass + Δ]).
    verbose : bool, default=False
        If ``True``, displays progress bars and diagnostic console messages.
    chunk_size : int, default=1000
        Number of spectra per chunk block for parallel computation.

    Returns
    -------
    pandas.DataFrame
        Structured DataFrame containing a `MS_index` column followed by 2 output 
        columns per peak target: ``f"{peak}_amplitude"`` and ``f"{peak}_area"``.
    """
    from .isotopes import batch_isotope_masses

    # Group target peaks by nominal mass integer
    peak_list_nm = []
    unique_nms = []
    nm_to_peaks = {}
    
    for peak in peak_list:
        nm = int(np.round(return_mass(peak)))
        peak_list_nm.append(nm)

        if nm not in unique_nms:
            unique_nms.append(nm)
        
        nm_to_peaks.setdefault(nm, []).append(peak)

    unique_nms = sorted(unique_nms)

    # Retrieve peak shape function selector
    peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)

    # Pre-compute fixed peak centers and FWHM constraints per nominal mass window
    precomputed_nm_data = {}
    for nm in unique_nms:
        centers = [return_mass(p) for p in nm_to_peaks[nm]]
        fwhms = [peak_width_function(c) for c in centers]
        
        # Scale theoretical FWHM to empirical custom shape boundaries if applicable
        if peak_type == 'custom' and hasattr(peak_func, 'gauss_to_ps'):
            fwhms = [f * peak_func.gauss_to_ps for f in fwhms]
            
        precomputed_nm_data[nm] = (centers, fwhms)

    # Pre-calculate theoretical isotope distributions for formula peaks
    precomputed_isotopes = batch_isotope_masses(peak_list, cutoff=1e-4)

    # Re-chunk Dask array to match requested processing chunk size
    if chunk_size:
        tofdata_subtracted = tofdata_subtracted.rechunk({0: chunk_size})

    # Formulate output DataFrame column headers
    out_columns = ["MS_index"]
    for p in peak_list:
        out_columns.extend([f"{p}_amplitude", f"{p}_area"])
    
    if verbose:
        print(f"Parallelizing fit over {len(peak_list)} peaks across Dask array...")

    # Execute worker function in parallel across Dask blocks
    with ProgressBar() if verbose else dask.config.set():
        final_array = tofdata_subtracted.map_blocks(
            _constrained_map_worker,
            calibration_results=calibration_results,
            unique_nms=unique_nms,
            nm_to_peaks=nm_to_peaks,
            precomputed_nm_data=precomputed_nm_data,
            precomputed_isotopes=precomputed_isotopes,
            peak_list=peak_list,
            peak_type=peak_type,
            custom_shape=custom_shape,
            sample_index_axis=sample_index_axis,
            tof_axis=tof_axis,
            nm_search_range=nm_search_range,
            peak_width_function=peak_width_function,
            dtype='float64',
            chunks=(tofdata_subtracted.chunks[0], len(out_columns))
        ).compute()

    # Package output matrix into structured pandas DataFrame
    return pd.DataFrame(final_array.reshape(-1, len(out_columns)), columns=out_columns)

