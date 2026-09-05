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

# opentof/__init__.py
from opentof.deployment import Deployment

from opentof.runner import run_do_it, main

from opentof.mass_calibration import (
    generate_averaged_dataset,
    run_mass_calibration, 
    define_reference_spectrum,
    combined_mc_aux_plots,
    mass_dependent_error_plot,
    estimate_global_noise_level,
    determine_baseline,
    apply_mass_calibration,
    fit_mass_calibration,
    MASS_CAL_MODES,
    get_mass_axis_for_ms_i,
    calibrate_mass,
    low_pass_filter,
)

from opentof.peak_width_shape import (
    peak_width, peak_shape, ransac_peak_width,
)

from opentof.peak_fitting import (
    gaussian, lorentzian, pseudo_voigt,
    peak_function_selector, 
    fit_and_extract_pos, 
    fit_single_peak, 
    fit_mopf_nm, 
    unpack_fit_result, 
    generate_peak_mask,
    estimate_local_noise_level,
    multi_overlap_peak_fit, 
    levenberg_marquardt,
    fit_unconstrained_peaks,
    full_fitting_integration_unconstrained,
    full_fitting_integration_constrained,
    fit_partially_constrained_nm,
)

from opentof.isotopes import (
    convolve_distributions,
    element_distribution, 
    calculate_isotope_masses,
    batch_isotope_masses,
    isotope_signal_on_axis,
    subtract_isotopes,
    reconstruct_total_isotope_signal,
)

from opentof.utils import (
    return_mass, get_raw_nm_data, get_nm_segment_data,
    std_list_array_lengths,
    GlobalMinMaxScaler, ptoe,
    plot_peak_fits,
    plot_apd_peaks,
    print_h5_structure,
    median_absolute_deviation,
    calculate_rolling_average,
    activity_classifier,
    truncate,
    spectra_nm_split,
    aggregate_tw_expicit_time_export,
    regr_model_metrics,
    regr_scatter_plot,
    load_peak_list,
    ensure_dir,
    find_possible_compositions,
    cluster_discovered_peaks,
    get_default_plot_dir,
)

from opentof.quantification import (
    load_TWEB_sensitivity_calibration_file, 
    quantify_signal,
)

# Experimental
from opentof.som_helpers import (
    RowMinMaxScaler, build_nm_som, organize_som_data,
    process_som_weight_peaks,
)

__all__ = [
    "Deployment",
    "run_do_it",
    "main",
    "return_mass",
    "get_raw_nm_data", 
    "get_nm_segment_data",
    "std_list_array_lengths",
    "GlobalMinMaxScaler",
    "ptoe",
    "plot_peak_fits",
    "plot_apd_peaks",
    "print_h5_structure",
    "median_absolute_deviation",
    "calculate_rolling_average",
    "activity_classifier",
    "truncate",
    "spectra_nm_split",
    "aggregate_tw_expicit_time_export",
    "regr_model_metrics",
    "regr_scatter_plot",
    "load_peak_list",
    "generate_averaged_dataset",
    "run_mass_calibration",
    "define_reference_spectrum",
    "combined_mc_aux_plots",
    "mass_dependent_error_plot",
    "estimate_global_noise_level",
    "determine_baseline",
    "apply_mass_calibration",
    "fit_mass_calibration",
    "MASS_CAL_MODES",
    "get_mass_axis_for_ms_i",
    "calibrate_mass",
    "low_pass_filter",
    "gaussian",
    "lorentzian",
    "pseudo_voigt",
    "peak_function_selector",
    "fit_and_extract_pos",
    "fit_single_peak",
    "fit_mopf_nm",
    "unpack_fit_result",
    "generate_peak_mask",
    "estimate_local_noise_level",
    "multi_overlap_peak_fit",
    "levenberg_marquardt",
    "full_fitting_integration_unconstrained",
    "fit_unconstrained_peaks",
    "full_fitting_integration_constrained",
    "fit_partially_constrained_nm",
    "peak_width",
    "ransac_peak_width",
    "peak_shape",
    "RowMinMaxScaler",
    "build_nm_som", 
    "organize_som_data",
    "process_som_weight_peaks",
    "load_TWEB_sensitivity_calibration_file", 
    "quantify_signal",
    "ensure_dir",
    "find_possible_compositions",
    "cluster_discovered_peaks",
    "get_default_plot_dir",
    "convolve_distributions",
    "element_distribution", 
    "calculate_isotope_masses",
    "batch_isotope_masses",
    "isotope_signal_on_axis",
    "subtract_isotopes",
    "reconstruct_total_isotope_signal",
]