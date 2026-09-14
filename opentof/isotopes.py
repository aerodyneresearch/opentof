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

# isotopes.py
from collections import defaultdict
from chemparse import parse_formula
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Internal utility to fetch atomic weights and natural abundances
from opentof.utils import PeriodicTable

# Single shared table instance
ptoe = PeriodicTable()

# Physical constant: Mass of an electron in Unified Atomic Mass Units (u)
ELECTRON_MASS = 0.000548579909065


def _normalize_isotope_relative_intensities(isotopes):
    """
    Convert monoisotopic-referenced relative intensities into a true probability distribution.

    Normalizes raw atomic relative abundance values (typically scaled to a monoisotopic base peak 
    height of 100) such that the sum of all isotopic probabilities equals 1.0 ($sum p_i = 1$).

    Parameters
    ----------
    isotopes : list of dict
        List of isotope dictionaries where each entry contains keys `'mass'` (float) and 
        `'relative_intensity'` (float).

    Returns
    -------
    list of tuple of (float, float)
        List of tuples containing ``(mass, probability)`` where probabilities sum to 1.0.
    """
    # Calculate the total sum of all relative intensities to use as the normalization denominator
    total = sum(iso["relative_intensity"] for iso in isotopes)

    # Divide each relative intensity by the total sum to yield true probabilities
    return [
        (iso["mass"], iso["relative_intensity"] / total)
        for iso in isotopes
    ]


def convolve_distributions(dist1, dist2, cutoff=1e-20, debug=False):
    """
    Convolve two discrete isotopic mass-probability distributions.

    Evaluates all pairwise combinations of isotope masses and probabilities between two 
    distributions. Sums probabilities for combinations resulting in identical total mass 
    and prunes low-probability noise below the specified cutoff threshold.

    Parameters
    ----------
    dist1 : list of tuple of (float, float)
        First distribution represented as a list of ``(mass, probability)`` tuples.
    dist2 : list of tuple of (float, float)
        Second distribution represented as a list of ``(mass, probability)`` tuples.
    cutoff : float, default=1e-20
        Minimum joint probability threshold ($p_1 * p_2$). Combinations below 
        this value are pruned to reduce memory overhead.
    debug : bool, default=False
        If ``True``, prints diagnostic convolution statistics to stdout.

    Returns
    -------
    result : list of tuple of (float, float)
        Convolved distribution represented as a list of ``(mass, probability)`` tuples.
    """
    # Use a defaultdict to aggregate combined probabilities for identical mass values
    out = defaultdict(float)

    # Evaluate all pairwise combinations between dist1 and dist2
    for m1, p1 in dist1:
        for m2, p2 in dist2:
            # Calculate joint probability of independent isotopic events
            p = p1 * p2
            
            # Prune joint probabilities below memory/precision cutoff threshold
            if p >= cutoff:
                # Accumulate joint probability at the combined total mass (m1 + m2)
                out[m1 + m2] += p

    # Convert aggregated dictionary back to a list of (mass, probability) tuples
    result = list(out.items())

    # Print diagnostic information if debug mode is active
    if debug:
        print(
            f"Convolution: {len(dist1)} x {len(dist2)} "
            f"→ {len(result)} peaks"
        )

    return result


def element_distribution(isotopes, count, debug=False):
    """
    Construct the isotopic probability distribution for a single element repeated `count` times.

    Normalizes atomic isotope abundances into probabilities and iteratively convolves the base 
    element distribution with itself `count` times using polynomial self-convolution.

    Parameters
    ----------
    isotopes : list of dict
        List of isotope dictionaries containing `'mass'` and `'relative_intensity'` keys.
    count : int or float
        Number of atoms of the element present in the sub-composition.
    debug : bool, default=False
        If ``True``, prints intermediate element distribution statistics to stdout.

    Returns
    -------
    dist : list of tuple of (float, float)
        Composite isotopic distribution for the repeated element as a list of 
        ``(mass, probability)`` tuples.
    """
    # Normalize raw atomic intensities into true probabilities summing to 1.0
    base = _normalize_isotope_relative_intensities(isotopes)

    # Initialize self-convolution with an identity distribution (mass 0.0, 100% probability)
    dist = [(0.0, 1.0)]
    
    # Iteratively convolve the running distribution with the single-atom base distribution
    for _ in range(int(count)):
        dist = convolve_distributions(dist, base)

    # Print detailed diagnostic log if debug flag is True
    if debug:
        print("\nElement isotopes (relative intensities):")
        for iso in isotopes:
            print(
                f"  mass={iso['mass']:.6f}, "
                f"relative_intensity={iso['relative_intensity']}"
            )

        print("Converted probabilities:")
        for m, p in base:
            print(f"  {m:.6f} -> {p:.6e}")

        print(
            f"Element distribution for count={count}: "
            f"{len(dist)} peaks, "
            f"total probability={sum(p for _, p in dist)}"
        )

    return dist

def get_parent_fraction(formula, debug=False):
    """
    Calculate the theoretical abundance fraction of the monoisotopic parent peak.

    Parses a chemical formula, convolves elemental isotopic distributions into a 
    full molecular distribution, and returns the absolute probability fraction ($0.0$ to $1.0$) 
    belonging to the lowest-mass monoisotopic peak relative to all isotopic variants.

    Parameters
    ----------
    formula : str
        Chemical formula string (e.g., ``"C6H6"`` or ``"IH2O-"``).
    debug : bool, default=False
        If ``True``, passes debug flags to underlying convolution methods.

    Returns
    -------
    parent_prob : float
        Absolute probability fraction ($0.0$ to $1.0$) of the monoisotopic parent peak.
    """
    # Parse formula into neutral elemental stoichiometry and net ionic charge
    neutral, charge = _parse_formula_with_charge(formula, debug=debug)
    
    # Build independent isotopic distributions for each element in the formula
    distributions = []
    for element, count in neutral.items():
        isotopes = ptoe[element].isotopes
        dist = element_distribution(isotopes, count, debug=debug)
        distributions.append(dist)

    # Return 1.0 (100%) if no valid elemental distributions were parsed
    if not distributions:
        return 1.0

    # Convolve elemental distributions sequentially to construct the molecular distribution
    iso_pattern = distributions[0]
    for dist in distributions[1:]:
        iso_pattern = convolve_distributions(iso_pattern, dist, cutoff=1e-20)
        
    # Sort isotopic peaks by mass ascending (lowest mass = monoisotopic parent peak)
    iso_pattern.sort(key=lambda x: x[0])
    
    # Extract probability fraction belonging to the monoisotopic parent peak
    parent_prob = iso_pattern[0][1]
    return parent_prob

def _parse_formula_with_charge(formula, debug=False):
    """
    Parse a chemical formula string into neutral elemental composition and net charge.

    Parameters
    ----------
    formula : str
        Chemical formula string containing element symbols and optional charge indicators 
        (e.g., ``"C2H5+"`` or ``"SO4-2"``).
    debug : bool, default=False
        If ``True``, prints formula parsing results to stdout.

    Returns
    -------
    neutral : dict
        Dictionary mapping element symbols (e.g., `'C'`, `'H'`) to atom counts.
    charge : int
        Net integer charge of the molecule (e.g., `+1`, `-1`, `0`).
    """
    # Parse chemical formula string using chemparse engine
    parsed = parse_formula(formula)

    neutral = {}
    charge = 0

    # Separate charge indicators ('+' and '-') from standard elemental stoichiometry
    for key, count in parsed.items():
        if key == "+":
            charge += count
        elif key == "-":
            charge -= count
        else:
            # Regular chemical element symbol (e.g., C, H, O, N)
            neutral[key] = count

    # Print parsed stoichiometry if debug flag is enabled
    if debug:
        print(f"\nFormula: {formula}")
        print("Parsed:", parsed)
        print("Neutral composition:", neutral)
        print("Net charge:", charge)

    return neutral, charge


def plot_isotope_stems(iso_pattern, title=None):
    """
    Plot discrete isotopic mass peaks as a stem plot ($m/z$ vs. relative intensity).

    Parameters
    ----------
    iso_pattern : list of tuple of (float, float)
        Discrete isotopic pattern represented as a list of ``(mass, intensity)`` tuples.
    title : str or None, default=None
        Optional title string displayed at the top of the plot canvas.
    """
    # Unpack mass (m/z) and intensity vectors from distribution tuples
    masses = [m for m, _ in iso_pattern]
    intensities = [i for _, i in iso_pattern]

    # Initialize Matplotlib figure and render stem plot
    plt.figure(figsize=(8, 3))
    plt.stem(masses, intensities)
    plt.xlabel("m/z")
    plt.ylabel("Relative intensity")
    
    if title:
        plt.title(title)
        
    plt.tight_layout()
    plt.show()


def calculate_isotope_masses(formula, cutoff=1e-3, debug=False):
    """
    Calculate normalized theoretical isotope masses and relative intensities for a chemical formula.

    Parses formula stoichiometry, convolves elemental isotopic distributions, applies electron 
    mass corrections for net charge ($m/z$), normalizes the highest intensity peak to 1.0 (base peak), 
    and filters out minor peaks below a relative intensity cutoff threshold.

    Parameters
    ----------
    formula : str
        Chemical formula string (e.g., ``"C6H6+"``, ``"C10H14N2O"``).
    cutoff : float, default=1e-3
        Minimum relative intensity threshold (normalized to max peak = 1.0) required 
        to retain a peak in the output dictionary.
    debug : bool, default=False
        If ``True``, prints diagnostic logs and displays an interactive stem plot.

    Returns
    -------
    dict
        Dictionary mapping exact isotopic masses ($m/z$) to normalized relative intensities ($0.0$ to $1.0$).
    """
    # Parse chemical formula into neutral stoichiometry dict and net integer charge
    neutral, charge = _parse_formula_with_charge(formula, debug=debug)

    # Compute independent isotopic probability distributions for each element
    distributions = []
    for element, count in neutral.items():
        isotopes = ptoe[element].isotopes
        dist = element_distribution(isotopes, count, debug=debug)
        distributions.append(dist)

    # Return empty dictionary if formula contains no valid elements
    if not distributions:
        return {}

    # Convolve elemental distributions sequentially to construct full molecular pattern
    iso_pattern = distributions[0]
    for dist in distributions[1:]:
        iso_pattern = convolve_distributions(
            iso_pattern, dist, cutoff=1e-20, debug=debug
        )

    # Apply electron mass correction for charged ions (m/z shift)
    if charge != 0:
        iso_pattern = [
            (m - charge * ELECTRON_MASS, p)
            for m, p in iso_pattern
        ]

    # Log un-normalized distribution details if debug mode is active
    if debug:
        print("\nFinal raw iso_pattern:")
        print("  Peaks:", len(iso_pattern))
        print("  Total probability:", sum(p for _, p in iso_pattern))
        print("  Max probability:", max(p for _, p in iso_pattern))

    # Identify maximum peak probability for base-peak normalization (setting base peak = 1.0)
    max_p = max(p for _, p in iso_pattern)
    
    # Normalize relative intensities and prune peaks below relative cutoff threshold
    iso_pattern = [
        (m, p / max_p)
        for m, p in iso_pattern
        if p / max_p >= cutoff
    ]

    # Sort isotopic peaks by mass ascending
    iso_pattern.sort(key=lambda x: x[0])

    # Display diagnostic text and stem plot if debug mode is active
    if debug:
        print("\nNormalized spectrum (relative intensities):")
        for m, i in iso_pattern[:10]:
            print(f"  {m:.6f} -> {i:.4f}")

        plot_isotope_stems(iso_pattern, title=formula)

    # Return results as a dictionary mapping mass -> relative intensity
    return dict(iso_pattern)


def batch_isotope_masses(peak_list, cutoff=1e-3, debug=False):
    """
    Calculate theoretical isotope distributions for a batch of chemical formulas or masses.

    Iterates over a peak list containing chemical formula strings or numeric $m/z$ values. 
    For chemical formulas, calls :func:`calculate_isotope_masses` to generate isotopic 
    distributions. For raw numeric masses (where stoichiometry is unknown), treats the entry 
    as a single monoisotopic peak with 100% relative intensity.

    Parameters
    ----------
    peak_list : sequence of (str, int, or float)
        Sequence of chemical formula strings (e.g., ``"C6H6+"``) or numeric $m/z$ values.
    cutoff : float, default=1e-3
        Minimum relative intensity threshold (normalized to base peak = 1.0) required 
        to retain an isotopic peak.
    debug : bool, default=False
        If ``True``, prints diagnostic logs and displays stem plots during formula parsing.

    Returns
    -------
    results : dict
        Dictionary mapping string identifiers to lists of ``(mass, relative_intensity)`` 
        tuples representing theoretical isotopic distributions.
    """
    results = {}

    for compound in peak_list:
        # Check if item is already a numeric mass (float/int) with unknown stoichiometry
        if isinstance(compound, (int, float)):
            # Treat numeric mass as a single monoisotopic peak with 1.0 (100%) relative intensity
            results[str(compound)] = [(float(compound), 1.0)]
            continue

        # Check if item is a chemical formula string
        if isinstance(compound, str):
            # Calculate theoretical isotopic distribution pattern for formula string
            spectrum = calculate_isotope_masses(
                compound, cutoff=cutoff, debug=debug
            )
            
            # Store non-empty isotopic distributions as a list of (mass, relative_intensity) tuples
            if spectrum:
                results[compound] = list(spectrum.items())

    return results


def isotope_signal_on_axis(
    formula,
    mass_axis,
    parent_amplitude,
    peak_width_function,
    peak_type='gaussian',
    custom_shape=None,
    parent_mu=0.5,
    cutoff=1e-4,
    precomputed_pattern=None,
):
    """
    Generate instrument-space continuous isotopic signal profiles across a mass axis.

    Calculates theoretical isotopic variants for a target compound formula, filters out 
    the monoisotopic parent peak (leaving only secondary isotope peaks), scales secondary 
    peak amplitudes relative to `parent_amplitude`, and renders continuous instrument line 
    shapes across `mass_axis` using the specified peak model and resolution function.

    Parameters
    ----------
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge ($m/z$) coordinates.
    parent_amplitude : float
        Fitted peak height amplitude of the monoisotopic parent peak.
    peak_width_function : callable
        Callable function mapping $m/z$ to expected peak FWHM resolution.
    peak_type : {'gaussian', 'lorentzian', 'pseudo_voigt', 'custom'} or callable, default='gaussian'
        Peak shape model identifier string or direct callable function object.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if ``peak_type='custom'``.
    parent_mu : float, default=0.5
        Gaussian/Lorentzian mixing weight parameter used if ``peak_type='pseudo_voigt'``.
    cutoff : float, default=1e-4
        Minimum relative intensity threshold required to generate an isotopic peak.
    precomputed_pattern : sequence of tuple or None, default=None
        Pre-calculated sequence of ``(mass, relative_intensity)`` tuples to bypass computation.

    Returns
    -------
    isotope_signal : numpy.ndarray
        1D array containing the continuous instrument response signal generated by 
        all secondary isotopic variants matching the shape of `mass_axis`.
    """
    # Local import to strictly prevent circular import dependencies between peak_fitting and isotopes
    from opentof.peak_fitting import peak_function_selector

    # Legacy parameter intercept: Handles cases where a pre-resolved callable function is passed as peak_type
    if callable(peak_type):
        resolved_peak_func = peak_type
        peak_type = custom_shape if isinstance(custom_shape, str) else 'custom'
    else:
        # Resolve peak function callable from string identifier
        resolved_peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)

    # Obtain ideal dimensionless isotopic pattern (mass, relative_intensity)
    if precomputed_pattern is not None:
        iso_items = precomputed_pattern
    else:
        iso_pattern = calculate_isotope_masses(formula, cutoff=cutoff, debug=False)
        
        # Return all-zero signal vector if no valid isotopic pattern was calculated
        if not iso_pattern:
            return np.zeros_like(mass_axis)
            
        iso_items = iso_pattern.items()

    # Identify the monoisotopic parent mass (defined as the lowest mass in the distribution)
    parent_mass = min(iso_mass for iso_mass, _ in iso_items)

    # Scale secondary isotope amplitudes relative to fitted parent peak height
    isotope_peaks = []
    for iso_mass, rel_intensity in iso_items:
        # Skip the monoisotopic parent peak itself (only generate secondary isotope signal)
        if iso_mass == parent_mass:
            continue
            
        # Scale secondary isotope height: A_iso = A_parent * I_rel
        amplitude = parent_amplitude * rel_intensity
        isotope_peaks.append((iso_mass, amplitude))

    # Initialize continuous signal accumulation array
    isotope_signal = np.zeros_like(mass_axis)

    # Render each secondary isotope peak onto the mass axis grid
    for mass_c, amplitude in isotope_peaks:
        # Evaluate expected peak FWHM resolution at the isotope mass center
        fwhm = peak_width_function(mass_c)

        # Scale theoretical FWHM to empirical custom shape boundaries if applicable
        if peak_type == 'custom' and hasattr(resolved_peak_func, 'gauss_to_ps'):
            fwhm *= resolved_peak_func.gauss_to_ps

        # Render continuous line shape profile according to selected model
        if peak_type == 'pseudo_voigt':
            isotope_signal += resolved_peak_func(
                mass_axis, amplitude, mass_c, fwhm, 
                parent_mu if parent_mu is not None else 0.5
            )
        else:
            isotope_signal += resolved_peak_func(mass_axis, amplitude, mass_c, fwhm)

    return isotope_signal


def subtract_isotopes(
    spectrum,
    isotope_signal,
    threshold=0.01,
    parent_amplitude=1.0
):
    """
    Subtract calculated isotopic signal contributions from a raw intensity spectrum.

    Performs spectrum deisotoping by identifying region channels where calculated secondary 
    isotopic signal exceeds a threshold fraction of the parent amplitude, subtracting 
    the calculated profile, and enforcing non-negative intensity bounds.

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of original observed spectral intensity values.
    isotope_signal : numpy.ndarray
        1D array of calculated continuous isotopic signal to subtract.
    threshold : float, default=0.01
        Fractional threshold ($0.0$ to $1.0$) relative to `parent_amplitude` required 
        to trigger subtraction on a channel.
    parent_amplitude : float, default=1.0
        Peak height amplitude of the parent peak used to set threshold sensitivity.

    Returns
    -------
    corrected : numpy.ndarray
        1D array containing the deisotoped spectrum with non-negative physical bounds applied.
    """
    # Construct boolean mask isolating channels where calculated isotope signal exceeds threshold
    mask = isotope_signal >= (threshold * parent_amplitude)
    
    # Create explicit array copy to prevent mutating caller's original spectrum
    corrected = spectrum.copy()
    
    # Subtract secondary isotopic signal contributions across masked channels
    corrected[mask] -= isotope_signal[mask]

    # Enforce physical constraint: intensity cannot evaluate below zero
    corrected[corrected < 0] = 0.0

    return corrected


def reconstruct_total_isotope_signal(
    ms_index, 
    df, 
    peak_list, 
    mass_axis, 
    peak_width_function, 
    custom_shape=None
):
    """
    Reconstruct the total cumulative secondary isotopic signal across a mass axis for a single scan.

    Extracts fitted parent peak parameters (centers and amplitudes) for a target spectrum index 
    (`ms_index`) from a wide-format results DataFrame, evaluates secondary isotopic signal 
    profiles via :func:`isotope_signal_on_axis` for each chemical formula in `peak_list`, 
    and returns the combined cumulative isotopic background vector.

    Parameters
    ----------
    ms_index : int
        0-based index of the target spectrum / writebuf to reconstruct.
    df : pandas.DataFrame
        Wide-format fitted peak results DataFrame containing a `'MS_index'` column along 
        with per-peak parameter columns (e.g., ``f"{peak}_peak_center_mass"``, ``f"{peak}_peak_amplitude"``).
    peak_list : sequence of (str or float)
        Sequence of target chemical formulas or $m/z$ values. Only string chemical formulas 
        are evaluated for isotopic reconstruction.
    mass_axis : numpy.ndarray
        1D array of calibrated mass-to-charge ($m/z$) coordinates.
    peak_width_function : callable
        Callable function mapping $m/z$ to expected peak FWHM resolution.
    custom_shape : callable or None, default=None
        Custom peak shape callable required if empirical custom shapes are used.

    Returns
    -------
    total_reconstructed_signal : numpy.ndarray
        1D array representing the cumulative secondary isotopic background signal across `mass_axis`.
    """
    # Pre-allocate output signal accumulation vector
    total_reconstructed_signal = np.zeros_like(mass_axis)

    # Extract target row for ms_index from wide-format results DataFrame
    try:
        row = df.loc[df['MS_index'] == ms_index].iloc[0]
    except IndexError:
        print(f"Error: MS_index {ms_index} not found in DataFrame.")
        return total_reconstructed_signal

    # Iterate through all targets in the peak list
    for peak in peak_list:
        # Skip numeric mass floats (stoichiometry and isotopic variants are unknown)
        if not isinstance(peak, str):
            continue

        # Define column header names where fitted mass center and amplitude are stored
        center_col = f"{peak}_peak_center_mass"
        amp_col = f"{peak}_peak_amplitude"

        # Verify required parameter columns exist in the DataFrame
        if center_col not in df.columns or amp_col not in df.columns:
            continue

        # Extract fitted center mass and amplitude values for this spectrum
        center_mass = row[center_col]
        amplitude = row[amp_col]

        # Skip peaks that failed to fit or were filtered out (NaNs)
        if pd.isna(center_mass) or pd.isna(amplitude):
            continue

        # Generate continuous secondary isotopic signal profile for this parent peak
        peak_signal = isotope_signal_on_axis(
            formula=peak,
            mass_axis=mass_axis,
            parent_amplitude=amplitude,
            peak_width_function=peak_width_function,
            peak_type='custom' if custom_shape is not None else 'gaussian',
            custom_shape=custom_shape,
            cutoff=1e-4
        )

        # Accumulate peak isotopic profile into master signal vector
        total_reconstructed_signal += peak_signal

    return total_reconstructed_signal