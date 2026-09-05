# Quick Start

The first step is having OpenTof installed! Make sure to follow the instructions within [installation.md](../installation.md).

This basic skeleton can be used to run through a basic high resolution time series (HRTS) workflow:

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

# # Optional (but recommended) second mass calibration with custom peak shape
# d.mass_calibration(calibrants,
#                    peak_type='custom',
#                    custom_shape=d.custom_peak_shape,)

# 4. Fit peaks and integrate
d.FFI_constrained(peak_type='custom')

# 5. View results
print(d.peak_data.head())