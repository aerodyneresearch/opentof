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

# peak_width_shape.py
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from sklearn.linear_model import RANSACRegressor, LinearRegression

# import from peak_fitting
from opentof.peak_fitting import (
    fit_gaussian_for_peak_shape, 
    fit_and_compute_fwhm, 
    find_peaks_in_spectrum,
    find_peaks_in_spectrum_ransac,
    normalize_peak_shape,
)
from opentof.utils import truncate, ensure_dir, get_default_plot_dir

def ransac_peak_width(reference_spectrum, rs_mass_axis, 
               min_prominence=0.9,
               minimum_mass_spacing=1.0, 
               residual_threshold_multiplier=0.10,
               search_range=None,
               peak_type='gaussian', 
               custom_shape=None,
               max_mz=None,
               force_zero_intercept=False, # probably uneeded and maybe even unphysical?
               output_dir=None,
               plot_flag=True,
               show_plot_flag=True,
               plot_name='peak_width.png',
               ransac_kwargs=None
               ):
    """
    Defines a peak width function that uses RANSAC to robustly fit a linear FWHM model 
    to mass spectral peaks, automatically ignoring outliers (e.g., overlapping isotopes).
    Computes an absolute minimum FWHM function with m/z.
    """
    if output_dir is None:
        output_dir = get_default_plot_dir()
    ensure_dir(output_dir)
    
    # Create the sample index axis
    si_axis = np.arange(len(reference_spectrum))
    # Sample index-mass space interpolator
    pchip_interp = PchipInterpolator(si_axis, rs_mass_axis)
    # Quickly find peaks in the spectrum based on prominence
    peak_index_locations = find_peaks_in_spectrum_ransac(reference_spectrum, min_prominence=min_prominence)

    if search_range is None:
        search_range = int(len(reference_spectrum) / 1000)

    # Initial Peak Fitting
    peak_data = fit_and_compute_fwhm(reference_spectrum, 
                                     si_axis, 
                                     peak_index_locations, 
                                     pchip_interp,
                                     search_range=search_range,
                                     peak_type=peak_type,
                                     custom_shape=custom_shape
                                     )

    # Initial Cleaning & Spacing Filter
    valid_mask = (peak_data[:, 1] > 0) & (peak_data[:, 1] <= 1.0)
    
    # Apply dynamic upper mass filter only if the user specifies one
    if max_mz is not None:
        valid_mask = valid_mask & (peak_data[:, 0] <= max_mz)
    
    interim_peak_data = peak_data[valid_mask]

    filtered_peak_data = filter_peaks_by_mass_spacing(interim_peak_data, min_mass_spacing=minimum_mass_spacing)

    # Extract Data
    masses = filtered_peak_data[:, 0]
    fwhms = filtered_peak_data[:, 1]
    dm = masses / fwhms  # Resolving Power

    # ==========================================
    # RANSAC ROBUST REGRESSION (FWHM vs. Mass)
    # ==========================================
    X_mass = masses.reshape(-1, 1)
    y_fwhm = fwhms

    # Dynamic threshold: e.g., 10% of the median FWHM
    dynamic_threshold = np.median(y_fwhm) * residual_threshold_multiplier 

    # Toggle the intercept fitting based on user input
    fit_intercept = not force_zero_intercept

    final_ransac_params = {
        'max_trials': 200,
        'random_state': 42,
        'residual_threshold': dynamic_threshold
    }

    if ransac_kwargs is not None:
        final_ransac_params.update(ransac_kwargs)

    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=fit_intercept),
        **final_ransac_params
    )
    
    # Fit the robust model
    ransac.fit(X_mass, y_fwhm)

    # Extract RANSAC inlier/outlier masks
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Extract Linear Fit Coefficients
    m_slope = ransac.estimator_.coef_[0]
    b_intercept = ransac.estimator_.intercept_ if fit_intercept else 0.0
    
    fwhm_fit_coeffs = [m_slope, b_intercept]
    peak_width_poly = np.poly1d(fwhm_fit_coeffs)

    # --- MINIMUM FWHM BOUND LOGIC ---
    m_inliers = masses[inlier_mask]
    fwhm_inliers = y_fwhm[inlier_mask]
    predicted_inliers = peak_width_poly(m_inliers)
    
    min_residual = np.min(fwhm_inliers - predicted_inliers)
    offset_b_intercept = b_intercept + min_residual
    
    # Create the offset polynomial function
    min_fwhm_fit_coeffs = [m_slope, offset_b_intercept]
    offset_peak_width_poly = np.poly1d(min_fwhm_fit_coeffs)

    print(f"Fit FWHM linear model (mass space): y = {truncate(fwhm_fit_coeffs[0], 5)} * x + {truncate(fwhm_fit_coeffs[1], 5)}")
    print(f"Minimum FWHM model: y = {truncate(m_slope, 5)} * x + {truncate(offset_b_intercept, 5)}")
    print(f"RANSAC identified {np.sum(inlier_mask)} inliers and {np.sum(outlier_mask)} outliers.")

    # ==========================================
    # RESOLVING POWER CURVE FIT
    # ==========================================
    def resolving_power_model(mass, a, b):
        return a * (mass ** b)

    dm_inliers = dm[inlier_mask]
    nonzero_mask = m_inliers > 0
    try:
        popt, _ = curve_fit(
            resolving_power_model,
            m_inliers[nonzero_mask],
            dm_inliers[nonzero_mask],
            p0=[1000, 0.5],
            bounds=([0, 0.1], [np.inf, 0.85]) 
        )
    except RuntimeError:
        print("Warning: Resolving power curve fit failed to converge. Defaulting to flat parameters.")
        popt = [np.median(dm_inliers), 0]

    # Reconstruct the fit curve
    m_fit_curve = np.linspace(0, masses.max(), 500)
    dm_fit_curve = resolving_power_model(m_fit_curve, *popt)
    
    # --- MAXIMUM RP BOUND LOGIC ---
    # Calculate the predicted RP for the inliers using our bounded model
    predicted_rp_inliers = resolving_power_model(m_inliers[nonzero_mask], *popt)
    
    # Find the maximum positive residual (the point highest above the average line)
    max_rp_residual = np.max(dm_inliers[nonzero_mask] - predicted_rp_inliers)
    
    # Create the theoretical maximum RP curve by shifting the average model UP
    max_dm_fit_curve = dm_fit_curve + max_rp_residual

    # Find the last whole-number nominal mass in the dataset
    last_nominal_mass = int(np.floor(masses.max()))
    
    # Calculate RP values
    rp_at_last_nominal = resolving_power_model(last_nominal_mass, *popt)
    max_rp_at_last_nominal = rp_at_last_nominal + max_rp_residual

    if plot_flag:
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        fig.suptitle(f"Peak Width & Resolving Power (RANSAC Fit)")

        for i, ax in enumerate(axs):
            y_data = dm if i == 1 else fwhms
                
            ax.scatter(masses[inlier_mask], y_data[inlier_mask], 
                       color='black', s=20, label='RANSAC Inliers')
            
            ax.scatter(masses[outlier_mask], y_data[outlier_mask], 
                       color='orange', marker='x', s=40, label='RANSAC Outliers')

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlabel("Mass-to-Charge (m/z)")

        # Plot FWHM Models
        axs[0].plot(masses, peak_width_poly(masses), color='red', linewidth=2, label='Linear FWHM Fit')
        axs[0].plot(masses, offset_peak_width_poly(masses), color='green', linewidth=2, linestyle='--', label='Minimum FWHM fit')
        axs[0].set_ylabel("FWHM (Mass Space)")
        
        # Add a light gray line exactly at y=0 so you can visually see where the limit of quantification occurs
        axs[0].axhline(0, color='gray', linestyle=':', alpha=0.6)
        axs[0].legend()

        # Plot the RP Models
        axs[1].plot(m_fit_curve, dm_fit_curve, color='red', linewidth=2, label='Fit RP Model')
        axs[1].plot(m_fit_curve, max_dm_fit_curve, color='green', linewidth=2, linestyle='--', label='Max Theoretical RP')
        axs[1].set_ylabel("Resolving Power (m/FWHM)")

        # Determine the highest relevant data point (max of either the inliers or the theoretical curve)
        max_relevant_rp = max(np.max(dm_inliers[nonzero_mask]), np.max(max_dm_fit_curve))
        y_upper_limit = max_relevant_rp * 1.15  # Add a 15% visual buffer
        
        # Clamp the y-axis
        axs[1].set_ylim(bottom=0, top=y_upper_limit)
        
        # Check for clipped outliers and alert the user visually
        clipped_outliers = np.sum(dm[outlier_mask] > y_upper_limit)
        if clipped_outliers > 0:
            axs[1].annotate(
                f"↑ {clipped_outliers} outliers above frame",
                xy=(0.5, 0.98),  # Positioned at the top center of the plot
                xycoords='axes fraction',
                ha='center', va='top',
                fontsize=10,
                color='darkorange',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.9)
            )

        # Annotation 
        if max_rp_at_last_nominal > 0:
            annotation_text = f"Fit RP: {int(round(rp_at_last_nominal))}"
            max_annotation_text = f"Max RP: {int(round(max_rp_at_last_nominal))}"
        else:
            annotation_text = f"Fit RP: {int(round(rp_at_last_nominal))}"
            max_annotation_text = f"Max RP: undefined"

        axs[1].annotate(
            annotation_text,
            xy=(last_nominal_mass, rp_at_last_nominal),      
            xytext=(0.75, 0.35),                             
            textcoords='axes fraction',                      
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray" , alpha=0.9),
            arrowprops=dict(arrowstyle="->", color='grey', lw=1.5)
        )
        axs[1].annotate(
            max_annotation_text,
            xy=(last_nominal_mass, max_rp_at_last_nominal),      
            xytext=(0.75, 0.4),                             
            textcoords='axes fraction',                      
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray" , alpha=0.9),
            arrowprops=dict(arrowstyle="->", color='grey', lw=1.5)
        )

        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_name)) # Always save to the directory
        if show_plot_flag:
            plt.show()
        plt.close()

    return fwhm_fit_coeffs, min_fwhm_fit_coeffs, popt

def peak_width(reference_spectrum, rs_mass_axis, 
               top_n=150, 
               min_prominence=0.01,
               minimum_mass_spacing=1.0, 
               max_num_outlires=10,
               n_manual_outliers=1, 
               search_range=None,
               peak_type='gaussian', 
               custom_shape=None,
               output_dir=None,
               plot_flag=True,
               show_plot_flag=True,
               plot_name='peak_width.png',
               ):
    """
    Defines a peak width function that excludes the top 'n' highest resolving power
    peaks as manual outliers before calculating final fit coefficients.
    """
    if output_dir is None:
        output_dir = get_default_plot_dir()
    ensure_dir(output_dir)
    
    # Create the sample index axis
    si_axis = np.arange(len(reference_spectrum))
    # Sample index-mass space interpolator
    pchip_interp = PchipInterpolator(si_axis, rs_mass_axis)
    # Quickly find peaks in the spectrum based on prominence
    peak_index_locations = find_peaks_in_spectrum(reference_spectrum, top_n=top_n, min_prominence=min_prominence)

    if search_range is None:
        search_range = int(len(reference_spectrum) / 1000)

    # Initial Peak Fitting
    peak_data = fit_and_compute_fwhm(reference_spectrum, 
                                     si_axis, 
                                     peak_index_locations, 
                                     pchip_interp,
                                     search_range=search_range,
                                     peak_type=peak_type,
                                     custom_shape=custom_shape
                                     )

    # Initial Cleaning & Spacing Filter
    valid_mask = (peak_data[:, 1] > 0) & (peak_data[:, 1] <= 1.0) & (peak_data[:, 0] <= 400)
    interim_peak_data = peak_data[valid_mask]
    filtered_peak_data = filter_peaks_by_mass_spacing(interim_peak_data, min_mass_spacing=minimum_mass_spacing)

    # Automatic Iterative Refinement
    # This gives us a baseline of "good" peaks according to the algorithm
    fwhm_fit_coeffs, fit_mask = iterative_fwhm_refinement(filtered_peak_data, max_outliers=max_num_outlires)

    # Extract Data for Manual Filtering
    masses = filtered_peak_data[:, 0]
    fwhms = filtered_peak_data[:, 1]
    dm = masses / fwhms  # Resolving Power

    # Apply Manual "Top N" Filter
    # We identify the N peaks with highest DM that are currently marked as 'fit'
    manual_exclusion_mask = np.zeros(len(masses), dtype=bool)
    
    # If the user wants to manually exclude peaks (similar to Tofware)
    if n_manual_outliers > 0:
        # Get indices of points that the auto-refiner liked
        currently_fit_indices = np.where(fit_mask)[0]
        
        # Of those, find the ones with the highest resolving power (dm)
        fit_dm_values = dm[fit_mask]
        # argsort gives indices of sorted values; [-n:] takes the largest
        top_n_relative_indices = np.argsort(fit_dm_values)[-n_manual_outliers:]
        top_n_absolute_indices = currently_fit_indices[top_n_relative_indices]
        
        # Mark these specifically for the Orange X plot
        manual_exclusion_mask[top_n_absolute_indices] = True
        
        # Update the fit_mask to REMOVE these before the final linear fit
        fit_mask = fit_mask & ~manual_exclusion_mask

    # FINAL LINEAR FIT (Peak Width Function)
    # Re-calculate coefficients excluding the manual outliers
    final_m_fit = masses[fit_mask]
    final_fwhm_fit = fwhms[fit_mask]
    fwhm_fit_coeffs = np.polyfit(final_m_fit, final_fwhm_fit, 1)
    peak_width_poly = np.poly1d(fwhm_fit_coeffs)

    print(f"Final FWHM linear model (mass space): y = {truncate(fwhm_fit_coeffs[0], 5)} * x + {truncate(fwhm_fit_coeffs[1], 5)}")

    # Resolving Power Curve Fit (Exponential Model)
    def resolving_power_model(mass, a, b):
        return a * (mass ** b)

    # Optimize the exponential model
    nonzero_mask = final_m_fit > 0
    popt, _ = curve_fit(
        resolving_power_model,
        final_m_fit[nonzero_mask],
        dm[fit_mask][nonzero_mask],
        p0=[1000, 0.5]
    )

    # Reconstruct the fit curve
    m_fit_curve = np.linspace(0, masses.max(), 500)
    dm_fit_curve = resolving_power_model(m_fit_curve, *popt)

    # Find the last whole-number nominal mass in the dataset
    last_nominal_mass = int(np.floor(masses.max()))
    # Calculate the RP value at exactly that mass using the fitted model
    rp_at_last_nominal = resolving_power_model(last_nominal_mass, *popt)

    if plot_flag:
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        fig.suptitle(f"Peak Width & Resolving Power (Manual Filter: n={n_manual_outliers})")

        for i, ax in enumerate(axs):
            y_data = dm if i == 1 else fwhms
            
            # Valid Points
            ax.scatter(masses[fit_mask], y_data[fit_mask], color='black', s=20, label='Fitted Peaks')
            
            # Automatically Excluded (Gray Circles)
            auto_mask = (~fit_mask & ~manual_exclusion_mask)
            ax.scatter(masses[auto_mask], y_data[auto_mask], s=20, marker='o', 
                    facecolor='none', edgecolors='darkgray', label='Auto-Excluded')
            
            # Manually Excluded (Orange X)
            ax.scatter(masses[manual_exclusion_mask], y_data[manual_exclusion_mask], 
                    color='orange', marker='x', s=40, label='Manual Outlier (High RP)')

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlabel("Mass-to-Charge (m/z)")

        # Plot Models
        axs[0].plot(masses, peak_width_poly(masses), color='red', linewidth=2, label='Linear FWHM Fit')
        axs[0].set_ylabel("FWHM (Mass Space)")
        axs[0].legend()

        # Plot the RP Model
        axs[1].plot(m_fit_curve, dm_fit_curve, color='red', linewidth=2, label='RP Model')
        axs[1].set_ylabel("Resolving Power (m/FWHM)")
        
        # --- Annotation with Arrow ---
        # annotation_text = f"RP @ m/z {last_nominal_mass}: {int(round(rp_at_last_nominal))}"
        annotation_text = f"{int(round(rp_at_last_nominal))}"
        
        axs[1].annotate(
            annotation_text,
            xy=(last_nominal_mass, rp_at_last_nominal),      # Arrow points here
            xytext=(0.8, 0.8),                               # Text location (relative to axes)
            textcoords='axes fraction',                      # Use 0-1 coordinate system for text
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=None , alpha=0.8),
            arrowprops=dict(
                arrowstyle="->",
                # connectionstyle="arc3,rad=-0.2",             # Curved arrow
                color='grey',
                lw=1.0
            )
        )
        # ----------------------------------

        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_name)) # Always save to the directory
        if show_plot_flag:
            plt.show()
        plt.close()

    # Return the linear fit coefficients
    return fwhm_fit_coeffs, popt

# Filter out any peaks if peak if peak FWHM is too small 
def filter_peaks_by_mass_spacing(peak_data, min_mass_spacing=1.0):
    """
    Function that filters out any peaks in peak_data that are closer together 
    in mass space than min_mass_spacing.
    """
    sorted_data = peak_data[np.argsort(peak_data[:, 0])]
    filtered = [sorted_data[0]]

    for i in range(1, len(sorted_data)):
        if sorted_data[i, 0] - filtered[-1][0] >= min_mass_spacing:
            filtered.append(sorted_data[i])

    return np.array(filtered)

# Iteratively fit a linear model to FWHM
def iterative_fwhm_refinement(peak_data, max_outliers=20):
    """
    Iteratively fit a linear model to peak FWHM values, removing peaks
    that are wider than the fitted line until only `max_outliers` remain.
    
    Parameters
    ----------
    peak_data : ndarray of shape (N, 2)
        Array where column 0 is x_c (center) and column 1 is FWHM values.
    max_outliers : int, optional
        Maximum number of outliers (wide peaks) allowed before stopping.
    
    Returns
    -------
    coeffs : ndarray
        Linear fit coefficients [slope, intercept].
    fit_mask : ndarray of bool
        Boolean mask of inliers used in the final fit.
    """
    fit_mask = np.ones(len(peak_data), dtype=bool)

    max_iterations = len(peak_data)
    iteration = 0

    while iteration < max_iterations:
        # Use only the inlier points for fitting
        x_c_values = peak_data[fit_mask, 0]
        FWHM_values = peak_data[fit_mask, 1]

        # Fit a linear model
        coeffs = np.polyfit(x_c_values, FWHM_values, 1)
        fit_line = np.poly1d(coeffs)(x_c_values)

        # Calculate residuals for only the inlier points
        residuals = FWHM_values - fit_line

        # Identify peaks with width greater than the fit
        outlier_indices = np.where(residuals > 0)[0]

        # Stop if we are within the tolerance
        if len(outlier_indices) <= max_outliers:
            break

        # Find the worst outlier index relative to fit_masked data
        worst_relative_idx = outlier_indices[np.argmax(residuals[outlier_indices])]

        # Map back to original index
        masked_indices = np.where(fit_mask)[0]
        worst_idx = masked_indices[worst_relative_idx]

        # Mark that index as not part of the fit
        fit_mask[worst_idx] = False
    else:
        raise RuntimeError("Iterative refinement did not converge/Removed all peaks")

    return coeffs, fit_mask

def peak_shape(reference_spectrum, 
               num_peak_threshold=70, 
               step=10,
               iqr_thresh_left=2.5, 
               iqr_thresh_right=2.5,
               peak_smoothing=0, # Apply smoothing to found raw peak shapes (no smoothing by default)
               mean_smoothing=1.0, # Apply minor smoothing to the mean shape (can help with center disconinutity)
               cursor_position=8,
               omega_r=0.70, # right side omega "apex tightness" parameter (higher = more aggressive)
               omega_l=0.80, # left side omega "apex tightness" parameter (higher = more aggressive)
               # Try to get rid of "flat" peaks
               flatness_region=0.5, # σ range from center to check
               flatness_thresh=0.95, # Higher = more tolerant to flatness
               tail_sigma_pos=9,
               tail_intensity_cutoff=0.03,
               output_dir=None,
               plot_flag=True,
               show_plot_flag=True,
               plot_filter_bounds=True,
               plot_flatness_filter=True,
               previous_peak_interp=None,
               plot_name='peak_shape.png',
               shape_filename='custom_shape',):
    """
    Extracts, normalizes, filters, and averages peak shapes from a reference spectrum to generate an empirical peak model.

    This function adaptively lowers an intensity threshold to find a target number of peaks. 
    Each peak is isolated, fitted to a Gaussian to establish its center and Full Width at 
    Half Maximum (FWHM), and normalized to a standard domain. Rigorous filtering is applied 
    independently to the left and right halves of the peaks to reject overlapping tails, 
    saturated (flat-topped) apexes, and shape outliers. The surviving halves are averaged, 
    saved to disk, and plotted.

    Args:
        reference_spectrum (np.ndarray): The 1D intensity spectrum array to extract peaks from.
        num_peak_threshold (int): The minimum number of peaks to collect before filtering.
        step (int): The intensity decrement step used during the adaptive peak finding loop.
        iqr_thresh_left (float): The Interquartile Range (IQR) multiplier for outlier rejection on the left half of the peaks.
        iqr_thresh_right (float): The IQR multiplier for outlier rejection on the right half of the peaks.
        peak_smoothing (float): The standard deviation (sigma) for Gaussian smoothing applied to individual peaks before filtering. Set to 0 to disable.
        mean_smoothing (float): The standard deviation (sigma) for Gaussian smoothing applied to the final averaged peak shape. Set to 0 to disable.
        cursor_position (float): The window width (±sigma) used to compute the fraction of the total peak area residing near the center.
        omega_r (float): Scaling parameter (0 to 1) for the right side IQR filter. Tapers the acceptance bounds near the peak apex.
        omega_l (float): Scaling parameter (0 to 1) for the left side IQR filter. Tapers the acceptance bounds near the peak apex.
        flatness_region (float): The standard deviation (sigma) distance from the center to evaluate for peak flatness (saturation).
        flatness_thresh (float): Threshold for flatness rejection. A value of 0.95 means if the minimum value in the `flatness_region` is > 95% of the maximum, the half is rejected.
        tail_sigma_pos (float): The x-axis position (in sigma) where tail intensity is evaluated to check for overlapping adjacent peaks.
        tail_intensity_cutoff (float): The maximum allowed normalized intensity at `tail_sigma_pos`. Peaks exceeding this are rejected.
        plot_filter_bounds (bool): If True, overlays the IQR filter acceptance bounds on the output plots.
        plot_flatness_filter (bool): If True, overlays the flatness evaluation boundaries on the output plots.
        previous_peak_interp (callable, optional): A previously generated cubic spline interpolator. If provided, it will be plotted for comparison.
        shape_filename (str): The base filename (without extension) used to save the resulting averaged x and y arrays as an `.npz` file.

    Returns:
        tuple: A tuple containing:
            - custom_interp (scipy.interpolate.interp1d): A cubic spline interpolator over the average normalized shape. Maps standardized x-values to normalized intensities.
            - custom_shape (callable): A function `f(x, A, x_c, FWHM)` that generates the empirical peak shape scaled to a specific amplitude, center, and width. Returns `(None, None)` if no valid peaks survive normalization.

    Notes:
        - The x-axis of the normalized domain is defined in standard deviations (sigma), spanning from -15 to +15.
        - Peak halves (left and right) are evaluated and retained/rejected completely independently to maximize the amount of usable empirical data.
    """
    if output_dir is None:
        output_dir = get_default_plot_dir()
    ensure_dir(output_dir)

    # Peak detection loop to collect at least `num_peak_threshold` peaks
    n_peaks = 0
    previous_n_peaks = 0  # Track how many we had last loop
    intensity_max = np.max(reference_spectrum)
    intensity_min = intensity_max - step

    loop_no = 1
    while n_peaks < num_peak_threshold:

        # If next step would go below zero, reduce step but DO NOT reset intensity_min
        if intensity_min - step < 0:
            # Save what the old step size was
            old_step = step
            # Then greatly reduce the step size
            step = step / 10
            print(f"WARNING: Intensity went negative trying to find enough peaks...\nReducing step from {old_step} to {step} starting at {intensity_min}.")
            # continue directly with the smaller step
            continue

        # Perform peak search
        positions, properties = signal.find_peaks(reference_spectrum, threshold=(intensity_min, intensity_max))
        n_peaks = len(positions)

        # ---------------------------------------------------------
        # Dynamic Step Reduction
        # ---------------------------------------------------------
        new_peaks_found = n_peaks - previous_n_peaks
        if new_peaks_found > 0:
            # Decrease step by 1 per new peak found, to a minimum of 1.
            # We only apply this if step is > 1 so we don't fight the /10 fallback above.
            if step > 1:
                step = max(1, step - new_peaks_found)
            
            # Update tracker for the next iteration
            previous_n_peaks = n_peaks

        # print(f"Found {n_peaks} after {loop_no} loops. Minimum peak intensity required: {intensity_min:.4f}")

        # Continue decreasing intensity threshold
        intensity_min -= step
        loop_no += 1

    # Normalize and collect peak shapes
    normalized_peaks = []
    for peak_idx in positions:
        result = fit_gaussian_for_peak_shape(np.arange(len(reference_spectrum)), reference_spectrum, peak_idx, search_range=20)
        if result is None:
            continue
        A, x_c, fwhm, x_fit, y_fit = result

        # Normalize peak to standard domain and scale height to 1
        x_norm, y_norm = normalize_peak_shape(x_fit, y_fit, x_c, fwhm)
        y_norm_scaled = y_norm / A
        normalized_peaks.append((x_norm, y_norm_scaled))

    if not normalized_peaks:
        print("No valid peaks after normalization and basic filtering.")
        return None, None

    # Interpolate each peak to shared x-axis and optionally smooth
    x_common = np.linspace(-20, 20, 400)  # Standardized grid (matched to Tofware domain)
    y_interp_list = []
    for x_p, y_p in normalized_peaks:
        y_interp = np.interp(x_common, x_p, y_p)

        if peak_smoothing > 0:
            y_interp = gaussian_filter1d(y_interp, sigma=peak_smoothing)

        y_interp_list.append(y_interp)

    y_interp_array = np.array(y_interp_list)  # Shape: (n_peaks, x_points)

    # Sanity check filter: Tail intensity at ±tail_sigma_pos must be low
    # Locate index in x_common nearest to ±tail_sigma_pos
    idx_left_check = np.argmin(np.abs(x_common + tail_sigma_pos))
    idx_right_check = np.argmin(np.abs(x_common - tail_sigma_pos))

    # Initialize half-keep flags (start True, then invalidate)
    keep_left = np.ones(len(y_interp_array), dtype=bool)
    keep_right = np.ones(len(y_interp_array), dtype=bool)

    # Create mask of booleans for tail intensity filter
    for i, y in enumerate(y_interp_array):
        # Left half check
        if y[idx_left_check] > tail_intensity_cutoff:
            keep_left[i] = False
        # Right half check
        if y[idx_right_check] > tail_intensity_cutoff:
            keep_right[i] = False

    # Split peaks into left/right halves and apply IQR filtering
    mid_idx = np.searchsorted(x_common, 0)
    y_left = y_interp_array[:, :mid_idx]
    y_right = y_interp_array[:, mid_idx:]

    # Apply tail sanity filter BEFORE computing percentiles
    y_left_w_tail_filter = y_left[keep_left]
    y_right_w_tail_filter = y_right[keep_right]

    # Safety fallback in case everything gets rejected
    if len(y_left_w_tail_filter) == 0:
        print("Warning: All left halves rejected by tail filter — using all for percentile filtering.")
        y_left_w_tail_filter = y_left

    if len(y_right_w_tail_filter) == 0:
        print("Warning: All right halves rejected by tail filter — using all for percentile filtering.")
        y_right_w_tail_filter = y_right

    # --- Left half filtering ---
    q1_left = np.percentile(y_left_w_tail_filter , 25, axis=0)
    q3_left = np.percentile(y_left_w_tail_filter , 75, axis=0)
    iqr_left = q3_left - q1_left

    # Scale factor for left side thresholds
    x_left_axis = x_common[:mid_idx]
    x_left_max = np.max(np.abs(x_left_axis))
    scale_left = 1 - omega_l * (1 - (np.abs(x_left_axis) / x_left_max)**2)

    lower_bound_left = q1_left - (iqr_thresh_left * scale_left) * iqr_left
    upper_bound_left = q3_left + (iqr_thresh_left * scale_left) * iqr_left
    mask_left = (y_left >= lower_bound_left) & (y_left <= upper_bound_left)

    # --- Right half filtering ---
    q1_right = np.percentile(y_right_w_tail_filter, 25, axis=0)
    q3_right = np.percentile(y_right_w_tail_filter, 75, axis=0)
    iqr_right = q3_right - q1_right

    x_right_axis = x_common[mid_idx:]
    x_right_max = np.max(np.abs(x_right_axis))
    scale_right = 1 - omega_r * (1 - (np.abs(x_right_axis) / x_right_max)**2)

    lower_bound_right = q1_right - (iqr_thresh_right * scale_right) * iqr_right
    upper_bound_right = q3_right + (iqr_thresh_right * scale_right) * iqr_right
    mask_right = (y_right >= lower_bound_right) & (y_right <= upper_bound_right)

    # Store the results of passing peaks
    keep_left = np.all(mask_left, axis=1)
    keep_right = np.all(mask_right, axis=1)

    # If the flatness region is non-zero (flatness filter is desired)
    if flatness_region != 0.0:
        # Flatness check on left halves
        apex_mask_left = (x_left_axis >= -flatness_region) & (x_left_axis <= 0)
        for i, y in enumerate(y_left):
            if keep_left[i]:
                region_vals = y[apex_mask_left]
                y_max = np.nanmax(region_vals)
                y_min = np.nanmin(region_vals)
                flatness = 1 - (y_max - y_min) / y_max
                if flatness > flatness_thresh:
                    keep_left[i] = False  # Reject flat-topped half
        apex_mask_right = (x_right_axis >= 0) & (x_right_axis <= flatness_region)
        for i, y in enumerate(y_right):
            if keep_right[i]:
                region_vals = y[apex_mask_right]
                y_max = np.nanmax(region_vals)
                y_min = np.nanmin(region_vals)
                flatness = 1 - (y_max - y_min) / y_max
                if flatness > flatness_thresh:
                    keep_right[i] = False

    # Combine filtered halves to form complete filtered peaks
    y_filtered = []
    for i in range(len(y_interp_array)):
        y = np.full_like(x_common, np.nan)
        if keep_left[i]:
            y[:mid_idx] = y_left[i]
        if keep_right[i]:
            y[mid_idx:] = y_right[i]
        y_filtered.append(y)
    y_filtered = np.array(y_filtered)

    n_total = len(y_interp_array)
    print(f"Left halves retained: {np.sum(keep_left)} / {n_total}")
    print(f"Right halves retained: {np.sum(keep_right)} / {n_total}")

    # === Compute average shape ===
    y_mean = np.nanmean(y_filtered, axis=0)
    peak_height = np.nanmax(y_mean)

    # === Optional smoothing ===
    if mean_smoothing > 0:
        y_mean = gaussian_filter1d(y_mean, sigma=mean_smoothing)
        y_mean *= peak_height / np.nanmax(y_mean)

    # Find the central peak in the averaged shape
    peaks, _ = signal.find_peaks(y_mean)
    main_peak_idx = peaks[np.argmax(y_mean[peaks])]
    
    # Measure its exact FWHM using SciPy
    widths, _, _, _ = signal.peak_widths(y_mean, [main_peak_idx], rel_height=0.5)
    dx = x_common[1] - x_common[0]
    measured_fwhm_sigma = widths[0] * dx
    
    # Calculate scaling factor (Gaussian theoretical FWHM is 2.3548 sigma)
    gauss_to_ps = measured_fwhm_sigma / 2.3548
    
    # Calculate the base integral of the normalized shape (PSsigma)
    ps_sigma = simpson(y_mean, x_common)
    # -----------------------------------------------------------------------------

    # === Gaussian reference ===
    sigma = 1
    gaussian_ref = np.exp(-0.5 * (x_common / sigma) ** 2)
    gaussian_ref /= np.max(gaussian_ref)

    # Area only relevant for panel 1, so Gaussian fill goes only there
    area_empirical = simpson(y_mean, x_common)
    area_gaussian = simpson(gaussian_ref, x_common)
    area_ratio = area_empirical / area_gaussian * 100

    # === Visualization multipanel ===
    if plot_flag:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

        # Define zoom presets we will apply to each subplot
        zoom_settings = [
            ("Full Range", 10, 0.0, 1.2),
            ("Peak Zoom", 1.3, 0.7, 1.2),
            ("Tail Zoom", 10, 0.0, 0.1),
        ]

        # --- PANEL 0: Unfiltered peaks (full view) ---
        for y in y_interp_array:
            axs[0].plot(x_common, y, linewidth=0.7, alpha=1)
        axs[0].set_title("Unfiltered Peaks (Full View)")
        axs[0].set_xlim(-zoom_settings[0][1], zoom_settings[0][1])
        axs[0].set_ylim(zoom_settings[0][2], zoom_settings[0][3])
        # axs[0].set_xlabel("Standard deviations (σ)")
        axs[0].set_ylabel("Normalized Intensity")

        # --- PANEL 1: Filtered peaks (full view) ---
        label_added = False
        for y in y_filtered:
            axs[1].plot(x_common, y, linewidth=0.7, alpha=1)
            if not label_added:
                axs[1].scatter(-tail_sigma_pos, tail_intensity_cutoff, color='red', marker='x', label='Tail Intensity Filter')
                axs[1].scatter( tail_sigma_pos, tail_intensity_cutoff, color='red', marker='x')
                label_added = True
            else:
                axs[1].scatter(-tail_sigma_pos, tail_intensity_cutoff, color='red', marker='x')
                axs[1].scatter( tail_sigma_pos, tail_intensity_cutoff, color='red', marker='x')
            
        axs[1].set_title("Filtered Peaks (Full View)")
        axs[1].set_xlim(-zoom_settings[0][1], zoom_settings[0][1])
        # axs[1].set_xlabel("Standard deviations (σ)")

        # --- PANEL 2: Peak Zoom ---
        for y in y_filtered:
            axs[2].plot(x_common, y, linewidth=0.7, alpha=1)
        axs[2].set_title("Filtered Peaks (Peak Zoom)")
        axs[2].set_xlim(-zoom_settings[1][1], zoom_settings[1][1])
        axs[2].set_ylim(zoom_settings[1][2], zoom_settings[1][3])
        axs[2].set_xlabel("Standard deviations (σ)")
        axs[2].set_ylabel("Normalized Intensity")

        # --- PANEL 3: Tail Zoom ---
        for y in y_filtered:
            axs[3].plot(x_common, y, linewidth=0.7, alpha=1)
        axs[3].set_title("Filtered Peaks (Tail Zoom)")
        axs[3].set_xlim(-zoom_settings[2][1], zoom_settings[2][1])
        axs[3].set_ylim(zoom_settings[2][2], zoom_settings[2][3])
        axs[3].set_xlabel("Standard deviations (σ)")

        if plot_filter_bounds:
            # Panels where filter bounds should appear
            bound_axes = [axs[1], axs[2], axs[3]]

            for ax in bound_axes:
                ax.plot(x_left_axis, lower_bound_left,
                        linestyle='dashdot', color='maroon')
                ax.plot(x_left_axis, upper_bound_left,
                        linestyle='dashdot', color='darkgreen')

                ax.plot(x_right_axis, lower_bound_right,
                        linestyle='dashdot', color='maroon', label=rf"\% Filter Lower Bound")
                ax.plot(x_right_axis, upper_bound_right,
                        linestyle='dashdot', color='darkgreen', label=rf"\% Filter Upper Bound")

            # === Flatness filter visualization ===
            if plot_flatness_filter and flatness_region > 0:
                y_max_ref = 1.0
                y_min_ref = flatness_thresh * y_max_ref

                for ax in bound_axes:
                    # Left slope
                    ax.plot([-flatness_region, 0],
                            [y_min_ref, y_max_ref],
                            linestyle="dashdot", color="blue")

                    # Right slope
                    ax.plot([0, flatness_region],
                            [y_max_ref, y_min_ref],
                            linestyle="dashdot", color="blue",
                            label="Flatness Filter")

        # --- Define which axes get these overlays (filtered + zooms)
        shape_axes = [axs[1], axs[2], axs[3]]

        # === Plot previous shape (if available) ===
        if previous_peak_interp is not None:
            y_previous = previous_peak_interp(x_common)
            for ax in shape_axes:
                ax.plot(
                    x_common,
                    y_previous,
                    color='darkcyan',
                    linewidth=2,
                    linestyle='-',
                    label='Previous Peak Shape'
                )

        # === Plot average peak shape ===
        for ax in shape_axes:
            ax.plot(
                x_common,
                y_mean,
                color='black',
                linewidth=2,
                label='Average Peak Shape'
            )

        for ax in axs:
            ax.fill_between(x_common, gaussian_ref, color='lightgray', alpha=0.5, label='Gaussian Area')
            ax.plot(x_common, gaussian_ref, 'r--', linewidth=1, label='Gaussian Trace')

        # Add legend to full plot
        axs[1].legend(loc='upper left')

        # Add zero line
        axs[1].axvline(0, color='gray', linestyle=':', linewidth=1)
        axs[2].axvline(0, color='gray', linestyle=':', linewidth=1)
        axs[3].axvline(0, color='gray', linestyle=':', linewidth=1)

        # --- Add area window annotations ---
        axs[1].axvline(-cursor_position, color='gray', linestyle='--', linewidth=1)
        axs[1].axvline(cursor_position, color='gray', linestyle='--', linewidth=1)

        # Compute area within ±cursor range
        mask = (x_common >= -cursor_position) & (x_common <= cursor_position)
        area_within_cursors = simpson(y_mean[mask], x_common[mask])
        area_total = simpson(y_mean, x_common)
        area_fraction = area_within_cursors / area_total * 100

        # --- Annotate statistics ---
        axs[1].annotate(
            f'Custom Area = {area_ratio:.2f}%\nof Gaussian',
            xy=(7.85, 0.85), ha='right', fontsize=10,
        )
        axs[1].annotate(
            f'Peaks Used: Left: {np.sum(keep_left)} / {n_total}\nRight: {np.sum(keep_right)} / {n_total}',
            xy=(7.85, 0.75), ha='right', fontsize=10,
        )
        axs[1].annotate(
            f'{area_fraction:.2f}% of total\narea within ±{cursor_position}σ',
            xy=(7.85, 0.65), ha='right', fontsize=10,
        )

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_name))
        if show_plot_flag:
            plt.show()
        plt.close()

    # === Return interpolator and shape function ===
    custom_interp = interp1d(x_common, y_mean, kind='cubic', fill_value=0, bounds_error=False)

    shape_name_npz = shape_filename + ".npz"
    save_path = os.path.join(output_dir, shape_name_npz)

    print(f"Saving Peak Shape to: {save_path}")
    np.savez(save_path, x_common=x_common, y_mean=y_mean)

    def custom_shape(x, A, x_c, FWHM):
        # Correct dimensionality scaling factor back to sigma
        x_normalized = (x - x_c) * 2.3548 / FWHM  
        return A * custom_interp(x_normalized)

    # Attach calculated constants to the function object for downstream use
    custom_shape.ps_sigma = ps_sigma
    custom_shape.gauss_to_ps = gauss_to_ps

    return custom_interp, custom_shape, area_ratio

