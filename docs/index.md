# Welcome to OpenTof

**OpenTof** is an open-source Python module designed for the processing, calibration, and high-resolution analysis of Time-of-Flight (TOF) mass spectrometry data. 

Built to handle massive instrument datasets efficiently, OpenTof leverages out-of-core processing and provides a complete suite of tools: from raw data loading and mass calibration to fitting/integration and interactive visualization.

---

## Key Features

*   **Scalable Data Handling:** Seamlessly load and process gigabytes of `.h5` instrument data out-of-core using `dask`, preventing memory bottlenecks.
*   **Robust Mass Calibration:** Perform automated mass calibrations with advanced diagnostic plots,optional auto rejection, and support for dense (supersaturated) calibrant lists.
*   **Advanced Peak Fitting:** Resolve complex overlapping peaks using our Multi-Overlap Peak Fitting (MOPF) engine. Support for Gaussian, Lorentzian, Pseudo-Voigt, and empirically derived custom peak shapes.
*   **Native Isotope Modeling:** Built-in periodic table and isotopic footprint modeling. Automatically calculate, visualize, and subtract complex isotopic interferences from parent molecules.
*   **Signal Quantification:** Easily apply background subtraction (auto-zeros) and sensitivity calibrations to translate raw ion counts into quantified time-series concentrations.
*   **Interactive GUI:** Launch the built-in `SpectrumWizardGUI` to interactively define baselines, fit peaks, and manage your peak lists without writing a script.

---

## At a Glance

Getting started is as simple as pointing OpenTof at your data directory:

```python
import opentof as ot

# 1. Load your dataset
d = ot.Deployment.from_directory("path/to/h5/files")

# Define potential calibrants (this list assumes PTR chemistry)
calibrants={"(H2O)2H+": 37.028405, "(H2O)3H+": 55.03897, 
            "C3H6OH+": 59.049141, "C7H8H+" : 93.069876}

# 2. Run mass calibration
d.mass_calibration(calibrants)

# 3. Define instrument/spectra specific functions
d.determine_reference_spectrum()
d.determine_baseline()
d.determine_peak_width()
d.determine_peak_shape()

# # Optional second mass calibration with custom peak shape
# d.mass_calibration(calibrants,
#                    peak_type='custom',
#                    custom_shape=d.custom_peak_shape,)

# 4. Fit peaks and integrate
d.FFI_constrained(peak_type='pseudo_voigt')

# 5. View results
print(d.peak_data.head())