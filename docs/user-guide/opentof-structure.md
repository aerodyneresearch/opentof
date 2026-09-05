# Overview of the OpenTof Abstraction Layers

There are **two** primary levels of abstraction within OpenTof.

1. HIGH-LEVEL: Functions contained within the "Deployment" class (within `deployment.py`) that operate directly on the Deployment object. 

The HIGH-LEVEL functions defined within `deployment.py` essentially just call the LOW-LEVEL functions contained within the individual `.py` files (`mass_calibration.py`, `peak_fitting.py`, etc.), with the appropriate inputs expected by these functions. 

If you are first starting out it is recommended to define the initial Deployment object as contained within the variable 'd' like the majority of this documentation assumes: 

```python
d = ot.Deployment.from_directory("some/dir/path")
```

Once the Deployment object has been initialized, ALL of the HIGH-LEVEL methods can be accessed via `d.` followed by the method name. For example:

```python
d.determine_baseline()
```

2. LOW-LEVEL: Functions are the funcitons contained within the individual .py files like `mass_calibration.py`, `peak_fitting.py`, `utils.py`, etc. Each function has been written with the capability to be called independently, but must be provided with its correct inputs! 

It is often convinient to have access to the LOW-LEVEL functions even if primarily working with the HIGH-LEVEL functions as some methods (like `ot.return_mass`) are accessible this way as on as you import OpenTof like so: 

```python
import opentof as ot
```

When building advanced and/or custom data processing pipelines these functions can provide convienent functions that can be used in sequence with the HIGH-LEVEL functions. Assuming you imported OpenTof in this way, the LOW-LEVEL functions can then be accessed using `ot.` followed by the method name. For example:

```python
ot.return_mass("C6H6+")
# returns: 78.046401
```

```python
ot.print_h5_structure("path/to/h5file")
# prints the contents of an .h5 file for quick viewing
```

```python
data = ppb_concentration_timeseries # For example
datetime64_timestamps = d.timestamps

results_dict = ot.activity_classifier(data, datetime64_timestamps)
# returns a dictionary with information pertaining to portions of the timeseries classifed as "active"
```

## Abstraction parallels

For a good example of the interplay between the two levels of abstraction in OpenTof we can use the process of determining a spectra's baseline.

The HIGH-LEVEL method for determining an appropriate spectral baseline is a minimum:

```python
# Determines the deployment baseline using the reference spectrum 
# Assumes it has already been defined via d.determine_reference_spectrum()
d.determine_baseline()
```

By default this method assumes the target spectra to use for the baseline determination procedure is the active reference spectra if one is defined. It then simply calls the LOW-LEVEL `ot.determine_baseline` function and then populates `Deployment.baseline` with information such as the reference spectra with the baseline subtracted out (`d.baseline['baseline_adjusted_spectrum']`), the denoised spectra processed through a low pass filter (`d.baseline['low_pass_data']`), and baseline itself (`d.baseline['adjusted_baseline']`)

In contrast, an equivalent call using the LOW-LEVEL `ot.determine_baseline` function using the reference spectra might look something like this:

```python
# Explicitly grab the 1D reference spectrum signal from the reference dictionary
reference_spectrum = d.reference['reference_spectrum']

# Call the LOW-LEVEL function providing the reference spectrum explicitly
results_dict = ot.determine_baseline(intensity_spectrum=reference_spectrum)

# Assign information to Deployment.baseline using the same HIGH-LEVEL keywords
d.baseline = {
            'baseline_adjusted_spectrum': results_dict[0],
            'low_pass_data': results_dict[1],
            'smoothed_baseline': results_dict[2],
            'adjusted_baseline': results_dict[3],
            'noise_level': results_dict[4],
            'med_diff': results_dict[5]
        }
```

Which is perhaps a trival example, but this can get more complicated for functions like `d.mass_calibration` for which the HIGH-LEVEL function just requires the user to pass a calibrant dictionary vs. the LOW-LEVEL `ot.run_mass_calibration` which requires the user to provide: 

- `tofdata` (multi-dimensional ions/s signal from instrument)
- `timestamps` (1D array of dtype datetime64 that align with the writebuf dimension in tofdata), 
- `standard_acq_data` (1D array binary mask that specifies periods of standard acquistion that align with the writebuf dimension in tofdata) 
- `massaxis_first_guess` (an initial guess of the the mass axis, usually from the instrument calibration)
- `tof_axis` (1D array of the the time-of-flight values aligning with the last dimension of tofdata calculated from the instrument start delay and sample interval)
- `calibrants` (dictionary of {"calibrant_formula" : calibrant_mass, ...} to be used/tried as calibrants)

The important thing to remember is that the HIGH-LEVEL functions are usually just "convienent" versions of some LOW-LEVEL functions! However, it is always possible to recreate the functionality of the HIGH-LEVEL functions from the LOW-LEVEL functions (it's just usually more work). 