# Quick Start

The first step is having OpenTof installed! Make sure to follow the instructions within [installation.md](../installation.md).

This basic skeleton can be used to run through a basic high resolution time series (HRTS) workflow:

```python
import opentof as ot

# 1. Load your dataset
d = ot.Deployment.from_directory("path/to/PTR/h5/files")

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

peak_list = [
        "C2H4OH+", # acetaldehyde
        "C4H6H+", # 1-3 butadiene
        "CH4SH+", # methanethiol
        "C2H3NH+", # acetonitrile
        "C3H6OH+", # acetone
        "C4H8OH+", # MEK
        "C6H12H+", # hexene-1
        "C3H6OH2OH+", # Acetone water cluster
        "C6H6+", # CT benzene
        "C6H7+", # protonated benzene
        "C7H8H+", # toluene
        "C2Cl4H+", # tetrachloroethylene,
        "C8H10H+", # m-xyline
        "C9H12H+", # 1-2-4 TMB
        ]

d.populate_peak_list_and_isotopes(peak_list)

# 4. Fit peaks and integrate
d.FFI_constrained(peak_type='custom')

# 5. View results
print(d.peak_data.head())