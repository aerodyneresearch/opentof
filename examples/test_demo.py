import pickle

import opentof as ot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_toggle = True

# Change to the directory that contains your h5 files.
deployment_dir_path = "path/to/demo_data"
d = ot.Deployment.from_directory(deployment_dir_path)

mass_calibration_filepath = "path/to/demo_data/2025_08_14_MassCal_H3O_hydroCarb_HS_v2 1.cal"
calibrants = ot.mass_calibration.load(mass_calibration_filepath)
calibrants


d.mass_calibration(calibrants, plot_flag=plot_toggle, averaging_interval=300)
d.calibration.keys()

d.define_reference_spectrum(plot_flag=plot_toggle, reference_peak_mass=ot.return_mass("C3H6OH+"))
d.reference.keys()

d.determine_baseline(plot_flag=plot_toggle)
d.baseline.keys()

d.determine_peak_width(plot_flag=plot_toggle)
print(d.peak_width_function(78))

d.determine_peak_shape(plot_flag=plot_toggle, tail_intensity_cutoff=0.03,
                       omega_r=0.7)
d.custom_peak_shape

d.mass_calibration(calibrants, 
                   peak_type='custom',
                   custom_shape=d.custom_peak_shape,
                   averaging_interval=300,
                   plot_flag=plot_toggle,
                   )

expected_compounds = ["C3H6OH+", "C3H7O(H2O)+", "C6H6+", 78.062485,
             "C7H9+", "C8H11+", "C9H13+", "C13H20H+", 
             "C6H18Si3O3H+", "C8H25Si4O4H+", "C10H31Si5O5+"]

d.populate_peak_list_and_isotopes(expected_compounds)

d.peak_list

d.FFI_constrained(peak_type='custom')

# This is equivalent to the code contained within the ---'s
# d.FFI_unconstrained(peak_type='custom')

# ---------------------------------------------------------------
# Change to the directory that contains your h5 files.
filepath = "path/to/demo_data/peak_data_constrained.pkl"
with open(filepath, 'wb') as file:
    pickle.dump(d.peak_data, file)

with open(filepath, 'rb') as file:
    d.peak_data = pickle.load(file)
# ---------------------------------------------------------------

# print("Begin populating time series")
# d.populate_time_series_unconstrained()

# d.time_series