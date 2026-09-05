# Contributing to OpenTof

First off, thank you for considering contributing to OpenTof! 

OpenTof is an open-source framework designed for the processing, calibration, and analysis of Time-of-Flight (TOF) mass spectrometry data. Whether you are fixing a bug, proposing a new feature, improving documentation, or tackling one of our core optimization challenges, **all contributions are highly encouraged and welcome**.

This document outlines the framework for contributing and highlights specific areas where the project is actively seeking community expertise.

---

## Active Areas for Contribution

While we welcome PRs of all kinds, we are actively looking for help from the community in the following areas:

### 1. Parallelization and Performance Optimization
OpenTof handles massive datasets and relies heavily on `dask` for out-of-core processing of TOF data. Currently, the codebase is underutilizing available CPU and compute resources. We are looking for Dask experts to help profile and optimize our pipeline.

**Target Areas:** 
*   **Optimizing the unconstrained fitter**: The `map_worker` implementations in `peak_fitting.py` (particularly the for the "unconstrained" fitter) is **much** slower than its constrained counterpart 
*   The delayed sparse matrix multiplications (`_multiply_chunk_sparse`) in `mass_calibration.py` have been improved significantly but have also proved challening in the past to optimize for large datasets: `generate_averaged_dataset` and `define_reference_spectrum`. 
*    Additionally, the processing of **unsplit** TOF-MS data (such as from the VOCUS AIM) is currently very slow, likely due to how `dask` is interfacing with the large raw data files. Contributions focusing on speeding up how these files are processed could greatly benefit the data processing workflows for fast reagent-ion switching instruments.
    *   **Goal:** Maximize parallel CPU saturation, reduce memory bottlenecks during HDF5 I/O, and speed up the overall full-fitting integration pipeline.

### 2. Isotope Modeling
The current implementation for handling isotopes in `isotopes.py` attempts to calculate isotope masses, convolve elemental distributions, and recreate these isotopic signatures in signal space for easy subtraction. However, we would love for domain experts and computational chemists to double-check this logic. 
*   **Target Areas:** The `calculate_isotope_masses`, `convolve_distributions`, `isotope_signal_on_axis` and `reconstruct_total_isotope_signal` functions.
*   **Goal:** Validate the accuracy against existing isotopic databases, optimize the convolution performance, and ensure the code gracefully handles extreme edge cases.

### 3. Interactive GUI Enhancements
OpenTof features an interactive processing wizard (`SpectrumWizardGUI`) built with `customtkinter`. 
*   **Goal:** Contributions that improve the UI/UX, add new visualization panels, expand GUI capabilities, optimize the rendering speed of the `matplotlib` canvas, or make the application more responsive during heavy backend computations are highly desired.

### 4. Testing and Documentation
As a math-heavy, data-intensive library, maintaining accuracy is paramount.
*   **Goal:** Additional, Jupyter notebook tutorials demonstrating various use cases for OpenTof would be incredibly valuable for new users.

### 5. Signal Quantification
*   The `quantify_signal` function in `quantification.py` has proved to be inconsistent for quantifying signals when on-line auto-zeros and sensitivity calibrations are not performed during acquisition. 
    *   **Goal** These sensitivity issues can generally be manually corrected, but a more robust function for general purpose signal quantification seems warrented. Although perhaps one function supporting every instrument use case isn't possible? What is all needed for signal quantification? Are there standard variables in the instrument .h5 files that can be used to automate this process?

---

## How to Contribute

### Submitting Bug Reports or Feature Requests
If you find a bug or have an idea for a new feature, please open an issue on our GitHub repository. Provide as much detail as possible, including:
*   A clear, descriptive title.
*   Steps to reproduce the issue (for bugs).
*   Details about your environment (OS, Python version, OpenTof version).
*   Expected behavior vs. actual behavior.

### Making a Pull Request (PR)
1. **Fork the repository** and clone it to your local machine.
2. **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3. **Write your code**. Please ensure your code adheres to standard Python PEP8 formatting.
4. **Test your changes** to ensure they do not break existing functionality.
5. **Commit your changes** with clear, descriptive commit messages.
6. **Push to your fork** and submit a Pull Request against the `main` branch of the OpenTof repository.

### Code Style Guidelines
*   We follow standard Python PEP8 conventions.
*   Please include docstrings for new functions, classes, and modules.
*   Comment complex mathematical logic, particularly in the peak fitting and calibration modules, to help future maintainers understand the methodology.

Thank you for helping make OpenTof better!