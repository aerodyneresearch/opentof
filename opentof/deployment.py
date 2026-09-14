# deployment.py

from __future__ import annotations

import glob
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import os
import importlib.resources
from tqdm.auto import tqdm
import dask.array as da
import json
from scipy.interpolate import interp1d
import inspect
import warnings


class Deployment:
    """
    OpenTof's Primary data structure created from a previously aquired Time-of-Flight (TOF) deployment/experiment.

    Consolidates, sorts, and stores spectral data, metadata, and instrument state 
    information across single or multiple original raw `.h5` acquisition files. 
    Manages out-of-core Dask data structures and time-alignment indexing.
    
    The Deployment class contains all of the "high-level" wrapper functions that
    make up the "core" functionality of OpenTof. Many of these functions operate
    directly on the Deployment object and use it as a medium to store information.

    Parameters
    ----------
    file_handles : dict or None, default=None
        Dictionary mapping file indices to active `h5py.File` handles.
    file_timestamps : dict or None, default=None
        Dictionary mapping file indices to 2D/1D timestamp arrays.
    file_shapes : dict or None, default=None
        Dictionary mapping file indices to raw HDF5 dataset shapes.
    sort_indices : dict or None, default=None
        Dictionary mapping file indices to 1D integer sorting masks for chronological alignment.
    target_nbr_samples : int or None, default=None
        Minimum spectrum sample length across all loaded files used for truncation alignment.
    chunk_size : int, default=1000
        Number of spectra per chunk block for out-of-core Dask matrix computations.
    segment_profiles : dict or None, default=None
        Dictionary mapping reagent ion names to segment index numbers for multi-segment data.
    reagent_ion : str or None, default=None
        Name of the primary reagent ion selected for segment slicing (e.g., ``"I-"``).
    plot_dir : str or pathlib.Path or None, default=None
        Default directory path for saving diagnostic figures and summaries.

    Attributes
    ----------
    segment_profiles : dict or None
        Active segment profile configuration dictionary.
    reagent_ion : str or None
        Selected primary reagent ion string identifier.
    plot_dir : str or pathlib.Path or None
        Base directory path for exported plots and diagnostic summaries.
    target_nbr_samples : int or None
        Standardized sample bin count across all loaded files.
    chunk_size : int
        Block chunking size for parallel Dask array evaluations.
    tof_axis : numpy.ndarray or None
        1D physical time-of-flight axis in nanoseconds.
    first_guess_mass_axis : numpy.ndarray or None
        1D estimated mass-to-charge (m/z) axis generated from median acquisition parameters.
    instrument_type : str or None
        Detected instrument configuration string (e.g., ``"SingleSegment"`` or ``"MultiSegment"``).
    calibration : dict or None
        Mass calibration results dictionary returned by :meth:`mass_calibration`.
    reference : dict or None
        Reference spectrum results dictionary returned by :meth:`determine_reference_spectrum`.
    baseline : dict or None
        Baseline evaluation results dictionary returned by :meth:`determine_baseline`.
    peak_shape_interp : scipy.interpolate.interp1d or None
        1D cubic spline interpolator representing the empirical peak shape.
    custom_peak_shape : callable or None
        Callable function `f(x, A, x_c, FWHM)` for empirical peak shape modeling.
    peak_width_function : numpy.poly1d or None
        Polynomial function evaluating expected FWHM peak width as a function of m/z.
    peak_width_coeffs : numpy.ndarray or None
        Fitted polynomial coefficients for the peak width function.
    tofdata_subtracted : dask.array.Array or None
        2D out-of-core Dask array containing baseline-subtracted spectral intensity values.
    peak_data : pandas.DataFrame or None
        Wide-format pandas DataFrame storing deconvoluted peak fitting amplitudes and areas.
    time_series : pandas.DataFrame or None
        Time-aligned DataFrame storing extracted compound timeseries profiles.
    nm_data : dict
        Dictionary storing extracted intensity, derivative, and axis segments by nominal mass.
    diagnostic_fits : dict
        Dictionary storing information collected from running diagnostic nominal mass fitters
    peak_list : dict or None
        Target compound dictionary containing lists of formulas, center masses, and FWHM bounds.
    isotopes : dict or None
        Theoretical isotopic distributions calculated for all formulas in `peak_list`.
    apd_df : pandas.DataFrame
        Dataframe containing results from running automated_peak_discovery
    peak_assignment_db : pandas.DataFrame
        Reference formula lookup database loaded from packaged compound tables.
        Populated with formulas from the PubChem database
    peak_assignment_mass_array : numpy.ndarray
        Pre-sorted 1D array of exact masses for fast formula candidate lookups.
    """

    def __init__(self, file_handles=None, file_timestamps=None, file_shapes=None, 
                 sort_indices=None, target_nbr_samples=None, chunk_size=1000, 
                 segment_profiles=None, reagent_ion=None,
                 plot_dir=None):
        
        # Internal dictionaries storing raw file handles and file-level metadata
        self._file_handles = file_handles if file_handles is not None else {}
        self._file_timestamps = file_timestamps if file_timestamps is not None else {}
        self._file_shapes = file_shapes if file_shapes is not None else {}
        self._sort_indices = sort_indices if sort_indices is not None else {}

        # Instrument acquisition segment configurations and target reagent ion
        self.segment_profiles = segment_profiles
        self.reagent_ion = reagent_ion
        self.plot_dir = plot_dir

        # State flags and corruption tracking masks
        self._user_entered_cycling_flag = False
        self._corrupt_flags = None  # Tracks corrupted writebuf masks across all loaded files

        # Global chronological sorting index and cached timestamp array
        self._global_sort_idx = None
        self._timestamps_cache = None

        # Internal containers for the Global Averaged Dataset (GAD) cache
        self._averaged_dataset = None
        self._averaged_dataset_interval = None

        # Standardized minimum sample length across files for truncation alignment
        self.target_nbr_samples = target_nbr_samples

        # Internal property cache for dynamically loaded HDF5 groups (TPS2, Pressures, etc.)
        self._cache = {} 
        
        # Primary axis definitions and instrument metadata
        self.tof_axis = None
        self.first_guess_mass_axis = None
        self.instrument_type = None

        # Primary processing result containers
        self.calibration = None
        self.fitting_results = None
        self.reference = None
        self.baseline = None
        self.peak_shape_interp = None
        self.custom_peak_shape = None
        self.peak_width_function = None
        self.peak_width_coeffs = None
        self.tofdata_subtracted = None

        # Master DataFrames for extracted peak fitting results and time-series
        self.peak_data = None
        self.time_series = None

        # Block chunk size for parallel Dask matrix evaluations
        self.chunk_size = chunk_size

        # Dictionary container for user-extracted nominal mass segments
        self.nm_data = {}

        # Container for stored diagnostic fitting results
        self.diagnostic_fits = {}

        # Target peak list dictionary and pre-calculated isotope patterns
        self.peak_list = None
        self.isotopes = None

        # Load standard packaged molecular formula database and build pre-sorted exact mass array
        self.peak_assignment_db = self._load_default_formula_db()
        self.peak_assignment_mass_array = self.peak_assignment_db["ExactMass"].to_numpy()

    # Trying methods to save corrupted data
    @staticmethod
    def _find_last_valid_write(h5_file):
        """
        Locate the highest valid write index in 'FullSpectra/TofData' prior to physical truncation.

        Uses a binary search algorithm to probe HDF5 dataset slices and pinpoint the 
        exact boundary where writebuf data truncation or file corruption occurred.

        Parameters
        ----------
        h5_file : h5py.File
            Open read-only handle to an HDF5 raw acquisition file.

        Returns
        -------
        last_valid : int
            The highest 0-based write index that can be successfully read from disk, 
            or ``-1`` if the dataset is missing or empty.
        """
        # Guard clause: check if the dataset path exists in the HDF5 file structure
        if 'FullSpectra/TofData' not in h5_file:
            return -1
            
        dset = h5_file['FullSpectra/TofData']
        n_writes = dset.shape[0]
        
        # Return -1 if dataset contains zero write records
        if n_writes == 0:
            return -1
        
        # Initialize binary search boundaries across total write count
        low, high = 0, n_writes - 1
        last_valid = -1
        
        # Binary search loop to pinpoint the physical truncation line
        while low <= high:
            mid = (low + high) // 2
            try:
                # Attempt to probe a minimal dataset slice at the midpoint write index
                _ = dset[mid:mid+1, 0, ...]
                
                # Slicing succeeded: record midpoint as valid and search higher write indices
                last_valid = mid
                low = mid + 1
            except (OSError, IndexError):
                # Slicing failed (truncated/corrupted disk block): narrow search to lower write indices
                high = mid - 1
                
        return last_valid

    # ------------------------
    # Constructors
    # ------------------------
    @classmethod
    def from_directory(cls, directory, pattern="*.h5", chunk_size=1000, 
                       segment_profiles=None, reagent_ion=None,
                       plot_dir=None)-> Deployment:
        """
        Construct a Deployment object from a chronological sequence of .h5 files in a directory.

        Discovers raw HDF5 files matching `pattern`, parses acquisition timestamps and shapes, 
        detects file truncation / corruption boundaries, standardizes sample lengths, determines 
        instrument configuration type, populates cycling status dictionaries, builds out-of-core 
        Dask TOF arrays, calculates initial mass axes, enforces global chronological sorting, 
        and auto-imports existing Tofware processing files (``_IF.h5`` or ``_p.h5``) if discovered.

        Parameters
        ----------
        directory : str or pathlib.Path
            Path to the target directory containing raw `.h5` acquisition files.
        pattern : str, default="*.h5"
            Glob matching pattern to identify raw data files.
        chunk_size : int, default=1000
            Number of spectra per chunk block for out-of-core Dask matrix operations.
        segment_profiles : dict or None, default=None
            Dictionary mapping reagent ion names to segment index numbers for multi-segment data.
        reagent_ion : str or None, default=None
            Name of the primary reagent ion selected for segment slicing (e.g., ``"I-"``).
        plot_dir : str or pathlib.Path or None, default=None
            Directory path for exported diagnostic plots and summary files.

        Returns
        -------
        obj : Deployment
            Fully initialized `Deployment` instance populated with data and metadata.

        Raises
        ------
        OSError
            If no matching `.h5` files are discovered in the target directory.
        ValueError
            If maximum sample length exceeds 2x minimum sample length across files.
        """
        # Discover and sort matching file paths, filtering out Tofware state backups and processed OpenTof files
        raw_files = sorted(glob.glob(str(Path(directory) / pattern)))
        files = [f for f in raw_files if not (f.endswith("_IF.h5") or f.endswith("_p.h5") or f.endswith("opentof_deployment.h5"))]

        if len(files) == 0:
            raise OSError("Directory does not contain any .h5 files. Check path.")

        # Initialize tracking containers for multi-file metadata
        file_handles, file_timestamps, file_shapes, sort_indices = {}, {}, {}, {}
        nbr_samples_list = []
        warning_already_printed = False
        corrupt_mask_list = []

        print("Step 1/6: Loading .h5 files")
        # Loop through discovered files and parse metadata
        for idx, file in enumerate(files):
            try:
                h5, timestamps, shape, sort_idx = cls._load_and_parse_h5(file, idx)

                # Probe HDF5 dataset for physical writebuf truncation line
                last_valid_w = cls._find_last_valid_write(h5)
                n_bufs = shape[1] if len(shape) > 1 else 1
                total_writebufs = len(sort_idx)

                # Build boolean corruption mask for writebufs extending beyond truncation line
                file_corrupt_mask = np.zeros(total_writebufs, dtype=bool)
                if last_valid_w < (shape[0] - 1):
                    flat_writes = sort_idx // n_bufs
                    file_corrupt_mask = (flat_writes > last_valid_w)
                    print(f"⚠️ Notice: File {Path(file).name} truncated after write {last_valid_w}/{shape[0]-1}. "
                        f"Flagging {np.sum(file_corrupt_mask)} spectra as corrupted.")

                # Store file metadata in tracking containers
                corrupt_mask_list.append(file_corrupt_mask)
                file_handles[idx] = h5
                file_timestamps[idx] = timestamps
                file_shapes[idx] = shape
                sort_indices[idx] = sort_idx
                
                # Record sample channel counts across files
                if 'NbrSamples' in h5.attrs:
                    ns = int(cls._get_scalar_attr(h5, 'NbrSamples'))
                    nbr_samples_list.append(ns)

                target_nbr_samples = None
                if nbr_samples_list:
                    min_samples = min(nbr_samples_list)
                    max_samples = max(nbr_samples_list)

                    # Warn if sample counts vary across files and require truncation
                    if ns != max_samples:
                        if not warning_already_printed:
                            print(f"[WARNING] The 'NbrSamples' is inconsistent between files within the requested directory.")
                            warning_already_printed = True
                        print(f"{file} has {ns} 'NbrSamples' which is longer than: {min_samples} 'NbrSamples'. Excess will be truncated down to this length.")

                    # Enforce safety check to prevent merging fundamentally mismatched datasets
                    if max_samples > 2 * min_samples:
                        raise ValueError(
                            f"Fatal mismatch in NbrSamples across files. "
                            f"Max NbrSamples ({max_samples}) is more than twice the min NbrSamples ({min_samples}) accross all files. "
                            f"Check that the input files are from the same instrument or configuration. "
                        )
                    target_nbr_samples = min_samples
            except (KeyError, Exception) as e:
                print(f"⚠️ Warning: Skipping file {file}.")
                continue

        # Instantiate Deployment container with parsed file handles and parameters
        obj = cls(file_handles, file_timestamps, file_shapes, sort_indices, target_nbr_samples, chunk_size, segment_profiles, reagent_ion, plot_dir=plot_dir)

        # Concatenate corrupt flags across all loaded files
        if corrupt_mask_list:
            obj._corrupt_flags = np.concatenate(corrupt_mask_list)
        
        # Step 2: Determine single-segment or multi-segment instrument type
        print("Step 2/6: Determining Instrument Type")
        obj.get_instrument_type()

        # Step 3: Populate cycling status flags
        print("Step 3/6: Populating deployment.cycling_status")
        obj.build_cycling_status()

        # Step 4: Construct out-of-core Dask TOF array
        print("Step 4/6: Building deployment.tofdata")
        obj.build_tofdata()

        # Step 5: Compute initial first-guess mass axis
        print("Step 5/6: Calculating deployment.first_guess_mass_axis")
        obj.get_first_guess()

        # Step 6: Enforce global chronological timestamp order across files
        print("Step 6/6: Enforcing global chronological order")
        obj.enforce_global_time_sort()
        
        # Auto-seek and import existing Tofware processing files (_IF.h5 and _p.h5) if present
        try:
            dir_path = Path(directory)
            has_if = (dir_path / "IF").exists() or list(dir_path.glob("*_IF.h5"))
            has_p = (dir_path / "Processed").exists() or list(dir_path.glob("*_p.h5"))

            if has_if and has_p:
                print("\n[Auto-Seek] Associated Instrument Function (_IF) and Processed (_p) files detected. Initializing auto-import...")
                obj.import_from_h5(directory, driver="PIF")
            elif has_if:
                print("\n[Auto-Seek] Associated Instrument Function (_IF) files detected. Initializing auto-import...")
                obj.import_from_h5(directory, driver="IF")
            elif has_p:
                print("\n[Auto-Seek] Associated Processed (_p) files detected. Initializing auto-import...")
                obj.import_from_h5(directory, driver="P")
        except Exception as e:
            print(f"ℹ️ Note: Auto-loading existing configurations skipped: {e}")
        
        return obj

    @classmethod
    def single_file(cls, filepath, chunk_size=1000, segment_profiles=None, 
                    reagent_ion=None, plot_dir=None) -> Deployment:
        """
        Construct a Deployment object from a single raw .h5 file.

        Parses timestamps and dimensions, determines instrument type, constructs out-of-core 
        Dask TOF arrays, populates cycling status, calculates initial mass axes, and auto-seeks 
        associated Tofware files (``_IF.h5`` or ``_p.h5``) in adjacent or subfolder paths.

        Parameters
        ----------
        filepath : str or pathlib.Path
            File path to the target raw `.h5` acquisition file.
        chunk_size : int, default=1000
            Number of spectra per chunk block for Dask array operations.
        segment_profiles : dict or None, default=None
            Dictionary mapping reagent ion names to segment index numbers for multi-segment data.
        reagent_ion : str or None, default=None
            Name of the primary reagent ion selected for segment slicing (e.g., ``"I-"``).
        plot_dir : str or pathlib.Path or None, default=None
            Directory path for exported diagnostic plots and summary files.

        Returns
        -------
        obj : Deployment
            Fully initialized `Deployment` instance for the single file.
        """
        # Parse single HDF5 file metadata and timestamps
        h5, timestamps, shape, sort_idx = cls._load_and_parse_h5(filepath, 0)
        
        target_nbr_samples = None
        if 'NbrSamples' in h5.attrs:
            target_nbr_samples = int(cls._get_scalar_attr(h5, 'NbrSamples'))

        # Instantiate Deployment object with single-file dictionary mapping
        obj = cls({0: h5}, {0: timestamps}, {0: shape}, {0: sort_idx}, target_nbr_samples, chunk_size, segment_profiles, reagent_ion, plot_dir=plot_dir)

        # Run standard initialization pipeline steps
        obj.get_instrument_type()
        obj.build_cycling_status()
        obj.build_tofdata()
        obj.get_first_guess()

        # Auto-seek and import matching Tofware processing files in adjacent or subfolder paths
        try:
            file_path = Path(filepath)
            
            # Check for Instrument Function (_IF.h5) files
            if_target_subfolder = file_path.parent / "IF" / f"{file_path.stem}_IF.h5"
            if_target_samefolder = file_path.parent / f"{file_path.stem}_IF.h5"
            has_if = if_target_subfolder.exists() or if_target_samefolder.exists()

            # Check for Processed (_p.h5) files
            p_target_subfolder = file_path.parent / "Processed" / f"{file_path.stem}_p.h5"
            p_target_samefolder = file_path.parent / f"{file_path.stem}_p.h5"
            has_p = p_target_subfolder.exists() or p_target_samefolder.exists()

            # Execute auto-import depending on discovered Tofware files
            if has_if and has_p:
                print(f"\n[Auto-Seek] Associated Instrument Function and Processed files discovered. Importing...")
                obj.import_from_h5(if_target_subfolder if if_target_subfolder.exists() else if_target_samefolder, driver="IF")
                obj.import_from_h5(p_target_subfolder if p_target_subfolder.exists() else p_target_samefolder, driver="P")
            elif has_if:
                target = if_target_subfolder if if_target_subfolder.exists() else if_target_samefolder
                print(f"\n[Auto-Seek] Associated Instrument Function discovered. Importing: {target.name}")
                obj.import_from_h5(target, driver="IF")
            elif has_p:
                target = p_target_subfolder if p_target_subfolder.exists() else p_target_samefolder
                print(f"\n[Auto-Seek] Associated Processed file discovered. Importing: {target.name}")
                obj.import_from_h5(target, driver="P")
        except Exception as e:
            print(f"ℹ️ Note: Single-file configuration auto-load skipped: {e}")

        return obj

    @classmethod
    def from_ot_h5(cls, filepath) -> Deployment:
        """
        Construct a complete Deployment instance directly from an OpenTof state file (.h5).

        Parameters
        ----------
        filepath : str or pathlib.Path
            File path to the native OpenTof workspace state file (e.g., ``opentof_deployment.h5``).

        Returns
        -------
        obj : Deployment
            Restored `Deployment` instance populated with saved workspace attributes and results.
        """
        # Instantiate empty Deployment base object
        obj = cls()
        
        # Use native OpenTof driver ('OT') to import and populate workspace attributes from state file
        obj.import_from_h5(filepath, driver="OT")
        
        return obj

    @property
    def timestamps(self):
        """
        Return the combined, chronologically sorted acquisition timestamps array.

        Checks the internal cache `_timestamps_cache` first. If uncached, iterates through 
        all file timestamp arrays, applies file-level sorting masks, coerces entries to 
        ``datetime64[ns]``, applies global sorting across multi-file runs, and caches the result.

        Returns
        -------
        numpy.ndarray
            1D array of acquisition timestamps with dtype ``datetime64[ns]``.
        """
        # Return cached timestamp array if available (essential for imported OpenTof state files)
        if getattr(self, '_timestamps_cache', None) is not None:
            return self._timestamps_cache

        # Fallback: Compute timestamp sequence from underlying loaded HDF5 files
        if not hasattr(self, '_file_timestamps') or not self._file_timestamps:
            return np.array([], dtype='datetime64[ns]')

        sorted_ts = []
        for k in sorted(self._file_timestamps.keys()):
            ts = self._file_timestamps[k]
            flat = ts.flatten()
            
            # Apply file-level sort index if present
            if k in self._sort_indices:
                flat = flat[self._sort_indices[k]]

            # Skip empty arrays immediately
            if flat.size == 0:
                continue

            # Coerce non-datetime dtypes inline to standardized datetime64[ns]
            if flat.dtype.kind != 'M':
                flat = pd.to_datetime(flat).to_numpy(dtype='datetime64[ns]')

            sorted_ts.append(flat)

        # Concatenate file-level timestamp arrays
        ts_concat = np.concatenate(sorted_ts) if sorted_ts else np.array([], dtype='datetime64[ns]')

        # Apply global chronological sort index across multi-file boundaries
        if getattr(self, '_global_sort_idx', None) is not None:
            ts_concat = ts_concat[self._global_sort_idx]

        # Cache constructed timestamp array
        self._timestamps_cache = ts_concat
        return self._timestamps_cache

    @property
    def default_plot_subdir(self):
        """
        Generate a default plot sub-directory prefix string based on the run start time.

        Formats the earliest recorded timestamp into a ``YYYYMMDD_HHMMSS`` string label.

        Returns
        -------
        str
            Timestamp prefix string or ``"OpenTof_Run"`` fallback if timestamps are missing.
        """
        # Format the first timestamp entry into a standardized YYYYMMDD_HHMMSS string
        if getattr(self, '_timestamps_cache', None) is not None and len(self.timestamps) > 0:
            try:
                first_ts = pd.to_datetime(self.timestamps[0])
                return first_ts.strftime("%Y%m%d_%H%M%S")
            except Exception:
                pass
        return "OpenTof_Run"

    @property
    def shapes(self):
        """
        Return raw spectral dataset shapes across loaded HDF5 acquisition files.

        Returns
        -------
        numpy.ndarray
            2D array of shape ``(num_files, 2)`` where each row contains ``(num_writes, num_samples)``.
        """
        # Reshape file shape tuple dictionary into a 2D matrix
        return np.array(list(self._file_shapes.values())).reshape(-1, 2)

    @property
    def cycling_status(self):
        """
        Return the pre-computed instrument cycling status dictionary.

        Returns
        -------
        dict
            Dictionary containing 1D arrays: ``'type'`` (cycling mode flags) and ``'step'`` (sequence steps).
        """
        return self._cycling_status

    @property
    def tofdata(self):
        """
        Return the master time-of-flight spectral data matrix.

        Returns
        -------
        dask.array.Array or numpy.ndarray or None
            2D spectral dataset array of shape ``(num_spectra, num_samples)``.
        """
        return self._tofdata

    @property
    def standard_acquisition_data(self):
        """
        Return a boolean mask indicating standard acquisition sampling periods.

        Evaluates the binary inverse of :attr:`nonstandard_acquisition_data`.

        Returns
        -------
        numpy.ndarray of bool
            1D array where ``True`` indicates valid ambient sample acquisition scans 
            and ``False`` indicates non-standard periods (auto-zeros, cals, or corrupt scans).
        """
        # Invert the non-standard acquisition mask (True = Standard Data)
        return ~self.nonstandard_acquisition_data.astype(bool)

    @standard_acquisition_data.setter
    def standard_acquisition_data(self, arr):
        """
        Manually override the standard acquisition mask.

        Parameters
        ----------
        arr : array-like of bool
            1D boolean array where ``True`` indicates standard acquisition spectra.

        Raises
        ------
        ValueError
            If `arr` length does not match total deployment spectrum count (`len(self.timestamps)`).
        """
        arr = np.asarray(arr, dtype=bool)
        
        # Guard clause: Validate array length against total deployment spectrum count
        if len(arr) != len(self.timestamps):
            raise ValueError(
                f"Length mismatch: Custom array has length {len(arr)}, "
                f"but deployment has {len(self.timestamps)} mass spectra."
            )
            
        # Invert the standard mask and store internally as manual non-standard flags
        self._manual_nonstandard_flags = ~arr
        self._user_entered_cycling_flag = True

    @property
    def tps2(self):
        """
        Lazy-load and return TPS2 power supply telemetry variables.

        Returns
        -------
        dict
            Dictionary of TPS2 voltage and current telemetry arrays loaded from HDF5.
        """
        return self._load_group('TPS2')

    @property
    def water_regulation(self):
        """
        Lazy-load and return WaterRegulation temperature and flow variables.

        Returns
        -------
        dict
            Dictionary of water regulation telemetry arrays loaded from HDF5.
        """
        return self._load_group('WaterRegulation')

    @property
    def pressures(self):
        """
        Lazy-load and return vacuum chamber pressure gauge readings.

        Returns
        -------
        dict
            Dictionary of pressure sensor arrays loaded from HDF5.
        """
        return self._load_group('Pressures')

    @property
    def concentration(self):
        """
        Lazy-load and return reagent/sample concentration telemetry.

        Returns
        -------
        dict
            Dictionary of concentration arrays loaded from HDF5.
        """
        return self._load_group('Concentration')

    @property
    def btfo(self):
        """
        Lazy-load and return Big TOF Filter / Optics (BTFO) telemetry variables.

        Returns
        -------
        dict
            Dictionary of filter and optics telemetry arrays loaded from HDF5.
        """
        return self._load_group('BTFO')

    @property
    def nonstandard_acquisition_data(self):
        """
        Return a boolean mask isolating non-standard acquisition scans.

        Combines cycling status flags (auto-zeros, sensitivity calibrations), transient step 
        padding windows, synthesized correction flags, manual user overrides, and physical 
        writebuf truncation masks.

        Returns
        -------
        numpy.ndarray of bool
            1D array where ``True`` marks non-standard/bad scans (auto-zeros, cals, corrupt scans) 
            and ``False`` marks valid ambient sampling scans.
        """
        # Return manual user override flags immediately if active
        if getattr(self, '_user_entered_cycling_flag', False): 
            return self._manual_nonstandard_flags

        data = self.cycling_status
        
        # Check if valid cycling data is present (any flag != -1)
        has_cycling = (data['type'] != -1).any()

        if has_cycling:
            # Mark any scan where type != 0 (non-standard sampling) as True
            flags = (data['type'] != 0)

            # Apply transient padding logic for native instrument cycling steps
            is_step_two = (data['step'] == 2)
            if is_step_two.any():
                # Flag startup transient period (scans before the first step 2 event)
                first_step_two_idx = is_step_two.argmax()
                flags[:first_step_two_idx] = True

            # Handle Step 5 (Sensitivity Calibration) transients with rolling padding
            is_step_five = (data['step'] == 5)
            if is_step_five.any():
                pad_before = np.roll(is_step_five, -2)
                pad_after = np.roll(is_step_five, 2)
                
                # Prevent wrap-around boundary padding
                pad_before[-1] = False 
                pad_after[0] = False
                
                # Combine step 5 and adjacent transient buffer scans
                flags = flags | is_step_five | pad_before | pad_after

        else:
            # Fallback: Assume all spectra are standard acquisition scans if cycling telemetry is missing
            length = self.tofdata.shape[0] if self.tofdata is not None else 0
            flags = np.zeros(length, dtype=bool)

        # Combine corrupt writebuf mask into non-standard acquisition flags via bitwise OR
        if getattr(self, '_corrupt_flags', None) is not None and len(self._corrupt_flags) == len(flags):
            flags = flags | self._corrupt_flags

        return flags

    @nonstandard_acquisition_data.setter
    def nonstandard_acquisition_data(self, arr):
        """
        Manually override the non-standard acquisition mask.

        Parameters
        ----------
        arr : array-like of bool
            1D boolean array where ``True`` indicates non-standard acquisition spectra.

        Raises
        ------
        ValueError
            If `arr` length does not match total deployment spectrum count (`len(self.timestamps)`).
        """
        arr = np.asarray(arr, dtype=bool)
        
        # Guard clause: Validate array length against total deployment spectrum count
        if len(arr) != len(self.timestamps):
            raise ValueError(
                f"Length mismatch: Custom array has length {len(arr)}, "
                f"but deployment has {len(self.timestamps)} mass spectra."
            )
            
        # Store manual non-standard flags and set user flag override marker
        self._manual_nonstandard_flags = arr
        self._user_entered_cycling_flag = True

    def _load_group(self, group_name):
        """
        Lazy-load and concatenate variable datasets from an HDF5 group structure.

        Reads HDF5 groups containing ``'TwData'`` and ``'TwInfo'`` arrays across all loaded 
        acquisition files, decodes byte-encoded variable names (handling UTF-8 and 
        Latin-1 encodings), flattens multi-dimensional write/buf matrices into 1D time-series, 
        applies per-file and global chronological sorting masks, and caches the results.

        Parameters
        ----------
        group_name : str
            Name of the HDF5 group to load (e.g., ``'TPS2'``, ``'Pressures'``, ``'WaterRegulation'``).

        Returns
        -------
        dict
            Dictionary mapping variable name strings to 1D concatenated NumPy arrays.
        """
        # Return cached dictionary if group has already been loaded into memory
        if group_name in self._cache:
            return self._cache[group_name]

        combined = {}
        found_any = False

        # Iterate through files in sorted order
        for k in sorted(self._file_handles.keys()):
            h5 = self._file_handles[k]
            
            # Skip file if target group is missing
            if group_name not in h5:
                continue
            
            found_any = True
            
            try:
                # Extract raw data matrix and variable metadata array
                twdata_raw = h5[group_name]['TwData'][:]
                twinfo_raw = np.atleast_1d(h5[group_name]['TwInfo'])
                
                # Decode variable names from byte strings (handling UTF-8 and Latin-1 fallback)
                var_names = []
                for n in twinfo_raw:
                    if isinstance(n, bytes):
                        try:
                            # Attempt standard UTF-8 decoding
                            var_names.append(n.decode('utf-8'))
                        except UnicodeDecodeError:
                            # Fallback to Latin-1 for special characters (e.g., degree symbols)
                            var_names.append(n.decode('latin-1'))
                    else:
                        var_names.append(str(n))

                # Flatten 3D write/buf matrices into 1D arrays and apply per-file sort index
                for idx, name in enumerate(var_names):
                    flat_data = twdata_raw[:, :, idx].ravel()
                    
                    if k in self._sort_indices:
                        flat_data = flat_data[self._sort_indices[k]]
                    
                    combined.setdefault(name, []).append(flat_data)

            except Exception as e:
                print(f"❌ Error extracting '{group_name}' in file {k}: {e}")

        # Concatenate arrays across files and cache results
        if not found_any:
            print(f"⚠️ Warning: Dataset '{group_name}' not found in deployment.")
            self._cache[group_name] = {}
        else:
            concatenated = {k: np.concatenate(v) for k, v in combined.items()}
            
            # Apply global chronological sorting mask if active
            if getattr(self, '_global_sort_idx', None) is not None:
                for k in concatenated:
                    concatenated[k] = concatenated[k][self._global_sort_idx]
            
            self._cache[group_name] = concatenated
            
        return self._cache[group_name]


    def build_cycling_status(self):
        """
        Build the combined 1D cycling status dictionary across all loaded acquisition files.

        Extracts native cycling flags (`'type'`, `'step'`, etc.) from raw files. If native 
        cycling telemetry is absent (all ``-1`` values), synthesizes auto-zero (type ``1``) 
        and sensitivity calibration (type ``4``) event masks directly from HDF5 
        ``'Corrections'`` dataset tables.

        Returns
        -------
        None
            Populates `self._cycling_status` in-place.
        """
        combined = {}

        # Extract raw per-file cycling flags into self.cycling_values
        self.extract_cycling_flags()

        # Flatten and collect cycling arrays across files
        for data in self.cycling_values.values():
            for key, arr in data.items():
                arr_flat = arr.ravel()
                combined.setdefault(key, []).append(arr_flat)

        # Concatenate file-level cycling arrays
        self._cycling_status = {key: np.concatenate(arrs) for key, arrs in combined.items()}

        # --- FALLBACK HIERARCHY FOR SYNTHESIZING CYCLING FLAGS ---
        # If native type array contains all -1s (missing telemetry), synthesize flags from 'Corrections' group
        if (self._cycling_status['type'] == -1).all():
            
            # Check if any file handle contains a 'Corrections' HDF5 group
            has_corrections = any('Corrections' in h5 for h5 in self._file_handles.values())

            if has_corrections:
                synth_type_list = []
                
                for k in sorted(self._file_handles.keys()):
                    h5 = self._file_handles[k]
                    
                    # Extract dataset dimensions to convert (write, buf) tuples to flat 1D indices
                    d1, d2 = h5['FullSpectra/TofData'].shape[:2]
                    num_bufs = d2
                    total_len = d1 * d2
                    
                    file_synth_type = np.zeros(total_len, dtype=np.int32)
                    
                    # Parse background/auto-zero intervals from Corrections/Background/Data
                    if 'Corrections' in h5 and 'Background/Data' in h5['Corrections']:
                        bg_data = h5['Corrections/Background/Data'][:]
                        for row in bg_data:
                            # Map (write_start, buf_start) -> 1D start index; (write_stop, buf_stop) -> 1D stop index
                            start_idx = int(row[1]) * num_bufs + int(row[2])
                            end_idx = int(row[7]) * num_bufs + int(row[8])
                            
                            # Assign type flag 1 for auto-zero / background
                            file_synth_type[start_idx:end_idx + 1] = 1
                            
                    # Parse sensitivity calibration intervals from Corrections/Calibration/Data
                    if 'Corrections' in h5 and 'Calibration/Data' in h5['Corrections']:
                        cal_data = h5['Corrections/Calibration/Data'][:]
                        for row in cal_data:
                            start_idx = int(row[1]) * num_bufs + int(row[2])
                            end_idx = int(row[7]) * num_bufs + int(row[8])
                            
                            # Assign type flag 4 for sensitivity calibration
                            file_synth_type[start_idx:end_idx + 1] = 4
                            
                    # Apply per-file timestamp sorting mask
                    file_synth_type = file_synth_type[self._sort_indices[k]]
                    synth_type_list.append(file_synth_type)
                    
                # Overwrite default -1 array with synthesized cycling type flags
                self._cycling_status['type'] = np.concatenate(synth_type_list)

    def build_tofdata(self):
        """
        Construct the master 2D out-of-core Dask spectral array and physical TOF axis.

        Extracts acquisition timing attributes (`StartDelay`, `SampleInterval`, `NbrWaveforms`, 
        `Single Ion Signal`, `TofPeriod`), applies segment profile filtering for selected 
        reagent ions, wraps raw HDF5 dataset slices with `HDF5ArrayWrapper`, constructs 
        a 2D out-of-core `dask.array.Array` scaled to ions/second, and calculates the 
        physical time-of-flight axis in nanoseconds.

        Returns
        -------
        None
            Populates `self._tofdata`, `self.sample_index_axis`, and `self.tof_axis` in-place.

        Raises
        ------
        KeyError
            If `self.reagent_ion` is set but missing from `self.segment_profiles`.
        """
        # Extract timing and sample interval attributes from the first loaded file
        first_h5 = next(iter(self._file_handles.values()))
        self.start_delay = int(self._get_scalar_attr(first_h5['TimingData'], 'StartDelay'))
        self.sample_interval = float(self._get_scalar_attr(first_h5['FullSpectra'], 'SampleInterval'))
        
        # Convert sample frequency (MHz) to time interval (seconds) if needed
        if self.sample_interval > 1.0:
            self.sample_interval = 1.0 / self.sample_interval

        # Determine segment slicing filter for target reagent ion in multi-segment setups
        seg_list = None
        if self.segment_profiles is not None and self.reagent_ion is not None:
            if self.reagent_ion not in self.segment_profiles:
                raise KeyError(f"Reagent ion '{self.reagent_ion}' not found in provided segment_profiles.")
            raw_seg = self.segment_profiles[self.reagent_ion]
            seg_list = [int(raw_seg)] if isinstance(raw_seg, (int, np.integer)) else [int(x) for x in raw_seg]

        dask_arrays = []

        # Process each HDF5 file to construct chunked Dask array wrappers
        for k in sorted(self._file_handles.keys()):
            h5 = self._file_handles[k]
            filepath = h5.filename
            
            # Calculate total waveform accumulation multiplier across memories and blocks
            nbr_waveforms = np.float32(self._get_scalar_attr(h5, 'NbrWaveforms'))
            nbr_memories = self._get_scalar_attr(h5, 'NbrMemories')
            if nbr_memories is not None and 'TofData' in h5['FullSpectra'] and h5['FullSpectra/TofData'].ndim >= 3 and h5['FullSpectra/TofData'].shape[2] == 1:
                nbr_waveforms *= np.float32(nbr_memories)
            nbr_blocks = self._get_scalar_attr(h5, 'NbrBlocks')
            if nbr_blocks is not None:
                nbr_waveforms *= np.float32(nbr_blocks)

            SIS = np.float32(self._get_scalar_attr(h5['FullSpectra'], 'Single Ion Signal'))
            tof_period = np.float32(self._get_scalar_attr(h5['TimingData'], 'TofPeriod'))

            # Instantiate lazy custom array wrapper for on-demand chunk loading and scaling
            wrapper = HDF5ArrayWrapper(
                filepath=filepath,
                key='FullSpectra/TofData',
                sort_idx=self._sort_indices[k],
                seg_list=seg_list,
                nbr_waveforms=nbr_waveforms,
                SIS=SIS,
                tof_period=tof_period,
                target_nbr_samples=self.target_nbr_samples
            )

            # Convert lazy wrapper into a 2D chunked Dask array
            d_k = da.from_array(wrapper, chunks=(self.chunk_size, wrapper.n_samples))
            dask_arrays.append(d_k)

        # Concatenate Dask array blocks along the spectrum axis (axis 0)
        self._tofdata = da.concatenate(dask_arrays, axis=0)

        # Construct 1D sample index channel axis and physical TOF axis in nanoseconds
        tof_axis_length = self._tofdata.shape[-1]
        self.sample_index_axis = np.arange(tof_axis_length)
        self.tof_axis = (self.sample_index_axis + self.start_delay) * self.sample_interval * 1e9

    def get_first_guess(self):
        """
        Generate an initial median mass-to-charge (m/z) coordinate axis.

        Extracts mass calibration parameters (p_1, ..., p_5) and calibration modes 
        across loaded acquisition files. Filters out short files containing fewer than 
        100 writebufs to avoid transient parameter outliers, computes the median parameter 
        vector across files, and generates `self.first_guess_mass_axis` via :func:`apply_mass_calibration`.

        Returns
        -------
        None
            Populates `self.first_guess_mass_axis` in-place.
        """
        from .mass_calibration import apply_mass_calibration

        valid_params = []
        fallback_params = []

        # Iterate over loaded file handles to extract calibration metadata
        for k, h5 in self._file_handles.items():
            # Determine total writebufs contained in TofData
            d1, d2 = h5['FullSpectra/TofData'].shape[:2]
            writebufs = d1 * d2

            # Extract dynamic calibration parameters (p1 through p5)
            params = []
            for i in range(1, 6):
                attr_name = f'MassCalibration p{i}'
                if attr_name in h5['FullSpectra'].attrs:
                    params.append(self._get_scalar_attr(h5['FullSpectra'], attr_name))
            
            # Determine mass calibration mode (default to mode 2 if omitted)
            mode = 2 
            if 'MassCalibMode' in h5['FullSpectra'].attrs:
                mode = int(self._get_scalar_attr(h5['FullSpectra'], 'MassCalibMode'))
            elif len(params) == 2:
                mode = 0  # Fallback mode for legacy 2-parameter calibration equation
                
            nbr_samples = int(self._get_scalar_attr(h5, 'NbrSamples')) 
            
            param_tuple = (mode, params, nbr_samples)
            
            # Record parameter tuples into fallback container and valid container (writebufs >= 100)
            fallback_params.append(param_tuple)
            if writebufs >= 100:
                valid_params.append(param_tuple)

        # Fallback to all files if no individual file contained >= 100 writebufs
        if not valid_params:
            print("⚠️ Warning: Input files contain <100 writebufs total. Using all writebufs for defining first guess mass axis.")
            params_to_use = fallback_params
        else:
            params_to_use = valid_params

        # Assume consistent calibration mode and parameter count across files
        mode_to_use = params_to_use[0][0]
        num_params = len(params_to_use[0][1])

        # Compute median value for each calibration parameter
        median_params = [np.median([p[1][i] for p in params_to_use]) for i in range(num_params)]
        
        # Override sample count with target_nbr_samples to ensure alignment with truncated data
        if self.target_nbr_samples is not None:
            median_samples = self.target_nbr_samples
        else:
            median_samples = int(np.median([p[2] for p in params_to_use]))

        # Generate sample channel index array [0, 1, ..., N-1]
        i_array = np.arange(median_samples)
        
        # Calculate initial first-guess mass-to-charge (m/z) axis
        self.first_guess_mass_axis = apply_mass_calibration(i_array, median_params, mode=mode_to_use)

    def get_instrument_type(self):
        """
        Detect and record the TOF instrument operational configuration.

        Inspects the HDF5 group hierarchy across loaded files. Assigns ``"SingleSegment"`` 
        if the ``'Ionization'`` group is missing or contains numeric mode attributes, or 
        ``"MultiSegment"`` if string-encoded mode attributes are present.

        Returns
        -------
        None
            Populates `self.instrument_type` in-place with ``"SingleSegment"`` or ``"MultiSegment"``.
        """
        found_types = set()

        for h5 in self._file_handles.values():
            # Check for existence of 'Ionization' HDF5 group
            if 'Ionization' not in h5:
                found_types.add("SingleSegment")
                continue

            # Differentiate MultiSegment vs. SingleSegment using 'Mode1' attribute type
            try:
                mode_attr = self._get_scalar_attr(h5['Ionization'], 'Mode1')
                
                # Check for text string / byte string types indicating MultiSegment configuration
                is_string = isinstance(mode_attr, (str, bytes))
                is_numpy_str = hasattr(mode_attr, 'dtype') and mode_attr.dtype.kind in ('S', 'U')

                if is_string or is_numpy_str:
                    found_types.add("MultiSegment")
                else:
                    found_types.add("SingleSegment")

            except Exception:
                # Fallback to SingleSegment on missing attributes
                print("⚠️ Warning: Unable to unambiguously assign instrument from data file. Defaulting to SingleSegment.")
                found_types.add("SingleSegment")

        if len(found_types) == 0:
            self.instrument_type = None
            return

        # Assign consolidated instrument type string
        self.instrument_type = found_types.pop()


    # ------------------------
    # Private helpers
    # ------------------------
    @staticmethod
    def _convert_filetime_to_pandas_datetime(acquisition_time_zero):
        """
        Convert a 64-bit Windows FILETIME integer timestamp into UTC pandas Timestamps.

        Parameters
        ----------
        acquisition_time_zero : int, float, or numpy.ndarray
            Windows FILETIME integer representing 100-nanosecond intervals since January 1, 1601 UTC.

        Returns
        -------
        pandas.DatetimeIndex or pandas.Timestamp
            Converted UTC timestamp(s) in pandas datetime format.
        """
        FILETIME_EPOCH = datetime(1601, 1, 1)
        SECONDS_IN_100_NANOSECONDS = 10**7
        
        # Calculate seconds elapsed since January 1, 1601
        seconds_since_1601 = acquisition_time_zero / SECONDS_IN_100_NANOSECONDS
        
        # Vectorize conversion to Python datetime objects
        utc_datetime = np.vectorize(lambda x: FILETIME_EPOCH + timedelta(seconds=x))(seconds_since_1601)
        
        # Return converted pandas datetime array
        return pd.to_datetime(utc_datetime.tolist())

    @classmethod
    def _load_and_parse_h5(cls, file, idx):
        """
        Parse timing metadata, writebuf shapes, and sorting indices from a raw HDF5 file.

        Parameters
        ----------
        file : str or pathlib.Path
            Path to the raw HDF5 acquisition file.
        idx : int
            0-based integer file index assigned during directory scanning.

        Returns
        -------
        h5 : h5py.File
            Open read-only HDF5 file handle.
        timestamps : numpy.ndarray
            2D array of absolute pandas acquisition timestamps matching `BufTimes` dimensions.
        shape : tuple of int
            2D shape tuple ``(num_writes, num_bufs)`` of the `BufTimes` dataset.
        sort_idx : numpy.ndarray
            1D integer sorting mask mapping flattened writebufs to chronological order.

        Raises
        ------
        ValueError
            If the target file is a native OpenTof backup state file.
        KeyError
            If required timing groups or attributes are missing.
        """
        # Open HDF5 file handle in read-only mode
        h5 = h5py.File(file, 'r')

        # Bypass native OpenTof workspace state files
        if "OpenTofFileType" in h5.attrs and h5.attrs["OpenTofFileType"] == "DeploymentState":
            h5.close()
            raise ValueError("Skipping processed native OpenTof state backup workspace file.")

        # Ensure required TimingData group exists
        if 'TimingData' not in h5:
            print(f"⚠️ Warning: {file} is missing 'TimingData' group.")
            raise KeyError
    
        timing = h5['TimingData']

        # Ensure AcquisitionTimeZero attribute exists
        if 'AcquisitionTimeZero' not in timing.attrs:
            print(f"⚠️ Warning: {file} is missing 'AcquisitionTimeZero' attribute.")
            raise KeyError

        # Convert 64-bit Windows FILETIME epoch to pandas datetime
        time_zero = cls._convert_filetime_to_pandas_datetime(
            timing.attrs['AcquisitionTimeZero']
        )
        
        # Extract relative buffer acquisition times in seconds
        time_data = np.array(timing['BufTimes']).flatten()
        
        # Construct absolute pandas timestamps for each writebuf
        timestamps = np.array([
            time_zero + pd.to_timedelta(s, unit='s') for s in time_data
        ]).reshape(timing['BufTimes'].shape)

        flat_timestamps = timestamps.flatten()

        # Identify valid writebufs (relative time > 0 or first buffer channel)
        valid_mask = (time_data > 0) | (np.arange(len(time_data)) == 0)
        valid_indices = np.where(valid_mask)[0]
        
        # Sort valid writebuf indices by absolute timestamp
        sorted_relative_idx = np.argsort(flat_timestamps[valid_indices])
        
        # Map sorted relative indices back to original flat writebuf positions
        sort_idx = valid_indices[sorted_relative_idx]

        # Return file handle, timestamps, shape tuple, and sorting mask
        return h5, timestamps, timing['BufTimes'].shape, sort_idx

    def extract_cycling_flags(self):
        """
        Extract hardware cycling telemetry flags aligned with writebufs across files.

        Reads fields (`cycle_active`, `step`, `type`, `duration`, `profile`) from the 
        `Cycling_Status` HDF5 group for each file. If cycling telemetry is absent, populates 
        arrays filled with ``-1`` default values. Applies per-file sorting masks.

        Returns
        -------
        cycling_values : dict
            Dictionary mapping file indices to sub-dictionaries of 1D sorted cycling flag arrays.
        """
        cycling_values = {}

        for k, h5 in self._file_handles.items():
            # Fallback: Populate default -1 arrays if Cycling_Status group is absent
            if 'Cycling_Status' not in h5:
                d1, d2 = h5['FullSpectra/TofData'].shape[:2]
                data = {
                    'cycle_active': np.full((d1 * d2,), -1, dtype=np.int32),
                    'step': np.full((d1 * d2,), -1, dtype=np.int32),
                    'type': np.full((d1 * d2,), -1, dtype=np.int32),
                    'duration': np.full((d1 * d2,), -1, dtype=np.int32),
                    'profile': np.full((d1 * d2,), -1, dtype=np.int32),
                }
                # Apply per-file timestamp sorting mask to default arrays
                for key in data:
                    data[key] = data[key][self._sort_indices[k]]
                cycling_values[k] = data
                continue

            twdata = h5['Cycling_Status']['TwData']
            twinfo = np.atleast_1d(h5['Cycling_Status']['TwInfo'])

            # Local helper to resolve variable column index in TwInfo
            def get_index(name): 
                return np.where(twinfo == name.encode())[0]

            indices = {
                'cycle_active': get_index('cycle_active'),
                'step': get_index('step'),
                'type': get_index('type'),
                'duration': get_index('duration'),
                'profile': get_index('profile'),
            }

            # Extract 1D flattened arrays for each cycling telemetry variable
            data = {
                key: (
                    np.squeeze(twdata[:, :, idx]).reshape(-1)
                    if idx.size > 0 
                    else np.full(twdata.shape[:2], -1).reshape(-1)
                )
                for key, idx in indices.items()
            }

            # Apply per-file timestamp sorting mask
            for key in data:
                data[key] = data[key][self._sort_indices[k]]

            cycling_values[k] = data

        self.cycling_values = cycling_values
        return self.cycling_values


    def _extract_and_stack_group(self, group_name):
        """
        Extract, decode, flatten, sort, and concatenate HDF5 datasets across loaded files.

        Parameters
        ----------
        group_name : str
            Target HDF5 group containing ``'TwData'`` and ``'TwInfo'`` datasets (e.g., ``'WaterRegulation'``).

        Returns
        -------
        dict
            Dictionary mapping variable name strings to 1D concatenated NumPy arrays.
        """
        combined = {}
        found_any = False

        # Iterate through files in sorted index order
        for k in sorted(self._file_handles.keys()):
            h5 = self._file_handles[k]
            
            # Verify group presence in file
            if group_name not in h5:
                print(f"⚠️ Warning: Group '{group_name}' not found in file index {k}")
                continue
            
            found_any = True
            
            try:
                twdata_raw = h5[group_name]['TwData'][:] 
                twinfo = np.atleast_1d(h5[group_name]['TwInfo'])
                
                # Decode byte-encoded variable names
                var_names = []
                for n in twinfo:
                    if isinstance(n, bytes):
                        var_names.append(n.decode('utf-8'))
                    else:
                        var_names.append(str(n))

                # Flatten write/buf matrices and apply per-file sorting masks
                for idx, name in enumerate(var_names):
                    flat_data = twdata_raw[:, :, idx].ravel()
                    
                    if k in self._sort_indices:
                        flat_data = flat_data[self._sort_indices[k]]
                    
                    combined.setdefault(name, []).append(flat_data)

            except Exception as e:
                print(f"❌ Error extracting '{group_name}' in file {k}: {e}")

        if not found_any:
            print(f"⚠️ Warning: Dataset '{group_name}' was not found in any loaded files.")
            return {}

        # Concatenate per-file arrays into unified 1D time-series vectors
        return {key: np.concatenate(arrs) for key, arrs in combined.items()}

    def enforce_global_time_sort(self):
        """
        Verify and enforce global chronological ordering of timestamps across all files.

        Checks if the concatenated time-series timestamps are monotonically increasing across 
        multi-file dataset boundaries. If out-of-order files or timestamp overlaps are detected, 
        computes a 1D global sorting index (`_global_sort_idx`) and re-orders the cached timestamp 
        array, cycling status arrays, and corruption tracking masks in-place.

        Returns
        -------
        None
            Updates `self._global_sort_idx`, `self._timestamps_cache`, `self._cycling_status`, 
            and `self._corrupt_flags` in-place.
        """
        # Retrieve concatenated timestamp vector across all loaded files
        ts = self.timestamps 
        
        # Check for monotonic time progression across multi-file boundaries using pandas
        if not pd.Series(ts).is_monotonic_increasing:
            # Calculate 1D global chronological sorting permutation mask
            self._global_sort_idx = np.argsort(ts)
            
            # Apply global chronological re-ordering to the cached timestamp array
            self._timestamps_cache = ts[self._global_sort_idx]
            
            # Re-sort all 1D arrays stored within the cycling status dictionary
            if getattr(self, '_cycling_status', None) is not None:
                for key in self._cycling_status:
                    self._cycling_status[key] = self._cycling_status[key][self._global_sort_idx]

            # Re-sort the corrupted writebuf tracking mask array
            if getattr(self, '_corrupt_flags', None) is not None:
                self._corrupt_flags = self._corrupt_flags[self._global_sort_idx]

    # ------------------------
    # Wrapper Functions
    # ------------------------
    def mass_calibration(self, calibrants=None, averaging_interval=300, plot_flag=True, 
                         show_plot_flag=True, overwrite_calibration=True, recompute_gad=False, 
                         output_dir=None, plot_subdir=None, **kwargs) -> Deployment:
        """
        Execute or reconstruct mass calibration across the deployment time-series.

        Computes time-resolved mass calibration parameters by optimizing peak fits 
        against known calibrant masses over time intervals (`averaging_interval`). 
        If pre-computed or imported calibration parameters exist (e.g., loaded from 
        an ``_IF.h5`` file), automatically reconstructs interval indices and mass axes 
        without re-running the optimization solver unless `overwrite_calibration=True`.

        Parameters
        ----------
        calibrants : dict or None, default=None
            Dictionary mapping calibrant compound strings to theoretical exact m/z values 
            (e.g., ``{"C2H5+": 29.0386, "C6H6+": 78.0464}``). Required if running optimization loops.
        averaging_interval : int, default=300
            Time window duration in seconds for building the Global Averaged Dataset (GAD).
        plot_flag : bool, default=True
            If ``True``, generates diagnostic mass calibration summary, TOF drift, and 
            mass-dependent error plots.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        overwrite_calibration : bool, default=True
            If ``True``, ignores pre-existing or imported calibration attributes and forces 
            re-execution of the non-linear optimization pipeline.
        recompute_gad : bool, default=False
            If ``True``, ignores internal GAD caches and forces recalculation of the 
            averaged dataset.
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        **kwargs : dict
            Additional keyword arguments passed to :func:`run_mass_calibration` or 
            auxiliary plotting routines (e.g., `precomputed_gad`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.calibration` populated and 
            `self.first_guess_mass_axis` updated to the median mass axis across all intervals.

        Raises
        ------
        ValueError
            If `calibrants` dictionary is omitted when re-running calibration loops.
        RuntimeError
            If an invalid external `precomputed_gad` matrix is supplied via `kwargs`.
        """
        # Resolve base output directory from instance attribute or default utility
        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        # Resolve plot sub-directory prefix from instance property if unassigned
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir
            
        # Combine base output directory with sub-directory prefix
        output_dir = os.path.join(output_dir, plot_subdir)
        
        # Configure keyword argument overrides for downstream plotting routines
        kwargs['output_dir'] = output_dir
        kwargs['show_plot_flag'] = show_plot_flag
        kwargs['mc_summary_plot_name'] = "mc_summary.png"
        kwargs['tof_drift_plot_name'] = "tof_drift.png"

        # --- STEP 1: DETECT AND RECONSTRUCT IMPORTED MASS CALIBRATION PARAMETERS ---
        # If calibration data was imported from Tofware/IF files, bypass fitting optimization
        if self.calibration is not None and "batch_params" in self.calibration and not overwrite_calibration:
            print("ℹ️ Found existing or imported mass calibration parameters. Set overwrite_calibration=True to recalculate.")
            from opentof.mass_calibration import apply_mass_calibration
            
            # Reconstruct interval_indices via vectorized nearest-neighbor timestamp matching
            imported_ts = np.atleast_1d(self.calibration["averaged_timestamps"])
            if len(imported_ts) > 0:
                # Normalize numeric timestamp formatting if stored as floating-point days
                if imported_ts.dtype.kind != 'M':
                    imported_ts_dt = np.array([np.datetime64('1970-01-01T00:00:00') + np.timedelta64(int(t * 86400), 's') for t in imported_ts])
                    self.calibration["averaged_timestamps"] = imported_ts_dt
                    imported_ts = imported_ts_dt
                
                # Binary search matching to assign each spectrum to its nearest calibration interval block
                indices = np.searchsorted(imported_ts, self.timestamps)
                indices = np.clip(indices, 0, len(imported_ts) - 1)
                left_indices = np.clip(indices - 1, 0, len(imported_ts) - 1)
                
                dist_right = np.abs(self.timestamps - imported_ts[indices])
                dist_left = np.abs(self.timestamps - imported_ts[left_indices])
                mask = dist_right > dist_left
                indices[mask] = left_indices[mask]
                
                self.calibration["interval_indices"] = indices
            else:
                self.calibration["interval_indices"] = np.zeros(len(self.timestamps), dtype=int)
            
            # Reconstruct mass axes for each interval parameter set
            batch_massaxes = []
            for params in self.calibration["batch_params"]:
                m_axis = apply_mass_calibration(self.sample_index_axis, params, mode=self.calibration["mass_cal_mode"])
                batch_massaxes.append(m_axis)

            self.calibration["batch_massaxes"] = batch_massaxes
            
            # Synchronize first_guess_mass_axis to median reconstructed mass profile
            self.first_guess_mass_axis = np.median(self.calibration['batch_massaxes'], axis=0)
            return self

        # --- STEP 2: STANDARD FITTING PIPELINE VALIDATION ---
        # Enforce requirement for calibrants dictionary when running optimization loops
        if calibrants is None:
            raise ValueError("calibrants dict must be provided to run mass calibration fitting loops.")

        # Validate custom external Global Averaged Dataset (GAD) if supplied via kwargs
        external_gad = kwargs.get('precomputed_gad', None)
        if external_gad is not None:
            print("🔍 Validating user-supplied custom 'precomputed_gad' tuple...")
            try:
                avg_data, avg_ts, int_idx = external_gad
                
                # Verify sample channel dimensions match raw TOF dataset
                if avg_data.shape[1] != self.tofdata.shape[-1]:
                    raise ValueError(f"TOF bin dimension mismatch! GAD bins: {avg_data.shape[1]}, Raw TofData bins: {self.tofdata.shape[-1]}")
                if len(avg_ts) != avg_data.shape[0]:
                    raise ValueError(f"Timestamp length ({len(avg_ts)}) doesn't match GAD rows ({avg_data.shape[0]}).")
                
                # Verify time step delta matches requested averaging_interval
                if len(avg_ts) > 1:
                    ts_seconds = pd.to_datetime(avg_ts).astype(np.int64) / 1e9
                    detected_delta = int(np.round(np.median(np.diff(ts_seconds))))
                    
                    if abs(detected_delta - averaging_interval) > 5:  # 5-second tolerance for sample jitter
                        raise ValueError(
                            f"Time interval mismatch! The provided precomputed_gad uses ~{detected_delta}s windows, "
                            f"but your mass_calibration call is requesting {averaging_interval}s windows. "
                            f"Please re-generate your custom GAD tuple or remove the override flag."
                        )
                print("  -> Custom GAD tuple structure verified successfully.")
            except Exception as e:
                raise RuntimeError(f"❌ Mass Calibration Denied due to an invalid external precomputed_gad: {e}")

        else:
            # Generate or update internal averaged dataset cache
            self.generate_averaged_dataset(averaging_interval=averaging_interval, recalculate=recompute_gad, **kwargs)

        # --- STEP 3: RUN BACKEND CALIBRATION PIPELINE ---
        from opentof.mass_calibration import (
            run_mass_calibration, 
            combined_mc_aux_plots, 
            mass_dependent_error_plot
        )
        
        # Filter kwargs to match signature parameters of run_mass_calibration
        cal_sig = inspect.signature(run_mass_calibration).parameters
        cal_kwargs = {k: v for k, v in kwargs.items() if k in cal_sig}

        # Select active GAD tuple (external override or managed internal cache)
        active_gad = external_gad if external_gad is not None else self._averaged_dataset

        # Execute mass calibration fitting routine across time intervals
        results = run_mass_calibration(
            tofdata=self.tofdata,
            timestamps=self.timestamps,
            standard_acq_data=self.standard_acquisition_data,
            massaxis_first_guess=self.first_guess_mass_axis,
            calibrants=calibrants,
            tof_axis=self.tof_axis,
            averaging_interval=averaging_interval,
            precomputed_gad=active_gad, 
            plot_flag=plot_flag,
            chunk_size=self.chunk_size,
            **cal_kwargs
        )
        
        # --- STEP 4: RENDER DIAGNOSTIC PLOTS ---
        if plot_flag:
            kwargs['plot_subdir'] = plot_subdir 
            
            # Filter kwargs matching combined_mc_aux_plots signature
            plot_sig = inspect.signature(combined_mc_aux_plots).parameters
            plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_sig}

            combined_mc_aux_plots(results, **plot_kwargs)

            # Generate mass-dependent error spline plots
            self.mde_median_spline, self.mde_tde_splines = mass_dependent_error_plot(
                results, 
                output_dir=output_dir,
                show_plot_flag=show_plot_flag,
            )

        # Store results dictionary on object instance
        self.calibration = results

        print("Calibration results stored in Deployment.calibration")

        # Synchronize first_guess_mass_axis to median profile across all fitted intervals
        self.first_guess_mass_axis = np.median(self.calibration['batch_massaxes'], axis=0)
        
        # Return self instance to enable method chaining
        return self

    def determine_reference_spectrum(self, output_dir=None, plot_subdir=None, 
                                     plot_flag=True, show_plot_flag=True, **kwargs) -> Deployment:
        """
        Calculate the global reference spectrum and positional mass offsets across time.

        Averages spectra across standard acquisition periods, calculates dynamic 
        positional peak drift (in PPM and absolute m/z), and populates the 
        `self.reference` dictionary. Must be executed after :meth:`mass_calibration`.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        plot_flag : bool, default=True
            If ``True``, renders and saves diagnostic reference spectrum and TOF drift figures.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        **kwargs : dict
            Additional keyword arguments passed to :func:`define_reference_spectrum` 
            (e.g., `reference_peak_mass`, `rolling_window`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.reference` dictionary populated containing:
                * ``'reference_spectrum'`` : 1D array of mean reference spectrum intensity values.
                * ``'rs_mass_axis'`` : 1D reference mass-to-charge (m/z) axis array.
                * ``'ppm_positional_offset'`` : 1D array of relative positional mass drift in PPM across time.
                * ``'absolute_positional_offset'`` : 1D array of absolute positional mass drift in m/z across time.
                * ``'rolling_average'`` : 2D array of rolling-averaged spectral intensity values.
                * ``'rolling_average_time_mask'`` : 1D boolean array indicating valid un-gapped rolling windows.

        Raises
        ------
        ValueError
            If :meth:`mass_calibration` has not been executed (`self.calibration is None`).
        """
        # Guard clause: Enforce requirement that mass calibration must be run first
        if getattr(self, 'calibration', None) is None:
            raise ValueError("Mass calibration has not been run. Please call `calibrate_mass()` first.")

        # Resolve base output directory path
        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        # Resolve plot sub-directory prefix
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir
            
        # Combine base output directory with sub-directory prefix
        output_dir = os.path.join(output_dir, plot_subdir)
        kwargs['output_dir'] = output_dir
        kwargs['plot_flag'] = plot_flag
        kwargs['show_plot_flag'] = show_plot_flag

        # Local import to prevent circular module dependencies
        from opentof.mass_calibration import define_reference_spectrum 
        
        # Inject precomputed global average spectrum cache if reference_peak_mass is unassigned
        if kwargs.get('reference_peak_mass') is None and self._averaged_dataset is not None:
            kwargs['precomputed_global_average'] = np.mean(self._averaged_dataset[0], axis=0)

        # Filter keyword arguments matching define_reference_spectrum parameter signature
        ref_sig = inspect.signature(define_reference_spectrum).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ref_sig}

        # Execute reference spectrum calculation routine
        ref_spec, rs_mass_axis, ppm_offset, abs_offset, roll_avg, roll_mask = define_reference_spectrum(
            calibration_results=self.calibration,
            tofdata=self.tofdata,
            standard_acq_data=self.standard_acquisition_data,
            timestamps=self.timestamps,
            chunk_size=self.chunk_size,
            **filtered_kwargs
        )
        
        # Populate reference dictionary attribute
        self.reference = {
            'reference_spectrum': ref_spec,
            'rs_mass_axis': rs_mass_axis,
            'ppm_positional_offset': ppm_offset,
            'absolute_positional_offset': abs_offset,
            'rolling_average': roll_avg,
            'rolling_average_time_mask': roll_mask
        }
        
        # Return self instance to enable method chaining
        return self

    
    def determine_baseline(self, target_spectrum=None, plot_flag=True, 
                           output_dir=None, plot_subdir=None, show_plot_flag=True, 
                           **kwargs) -> Deployment:
        """
        Calculate the baseline and global noise level for a spectrum.

        Evaluates baseline continuum drift and standard deviation noise levels across 
        the target spectrum (defaulting to `self.reference['reference_spectrum']`). 
        Populates `self.baseline` with adjusted spectrum arrays and subtracts 
        the baseline profile from `self.tofdata` to build `self.tofdata_subtracted`.

        Parameters
        ----------
        target_spectrum : numpy.ndarray or None, default=None
            1D intensity spectrum array to analyze. Defaults to the calculated reference 
            spectrum (`self.reference['reference_spectrum']`) if ``None``.
        plot_flag : bool, default=True
            If ``True``, generates diagnostic baseline correction figures.
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        **kwargs : dict
            Additional keyword arguments passed to :func:`determine_baseline` 
            (e.g., `cutoff_freq`, `window_size`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.baseline` dictionary populated and 
            `self.tofdata_subtracted` constructed containing:
                * ``'baseline_adjusted_spectrum'`` : 1D baseline-subtracted reference spectrum array.
                * ``'low_pass_data'`` : 1D low-pass filtered trendline array.
                * ``'smoothed_baseline'`` : 1D Savitzky-Golay smoothed baseline array.
                * ``'adjusted_baseline'`` : 1D final adjusted baseline profile array.
                * ``'global_noise_level'`` : Calculated baseline noise standard deviation (σ).
                * ``'med_diff'`` : Median residual difference.

        Raises
        ------
        ValueError
            If `target_spectrum` is ``None`` and `self.reference` has not been defined.
        """
        # Safety check: Default to the reference spectrum if no target is provided
        if target_spectrum is None:
            if getattr(self, 'reference', None) is None:
                raise ValueError(
                    "No reference spectrum found. Please call `define_reference_spectrum()` first, "
                    "or explicitly provide a `target_spectrum` array."
                )
            target_spectrum = self.reference['reference_spectrum']

        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir
            
        output_dir = os.path.join(output_dir, plot_subdir)
        kwargs['output_dir'] = output_dir
        kwargs['show_plot_flag'] = show_plot_flag

        # Import inside the method to prevent circular import issues
        from opentof.mass_calibration import determine_baseline as determine_baseline
        
        # Run the external function
        # All extra parameters (cutoff_freq, window_size, etc.) can be passed via **kwargs
        results = determine_baseline(
            intensity_spectrum=target_spectrum,
            plot_flag=plot_flag,
            **kwargs
        )
        
        # determine_baseline returns a tuple of 6 items:
        # (baseline_adjusted_spectrum, low_pass_data, smoothed_baseline, adjusted_baseline, noise_level, med_diff)
        self.baseline = {
            'baseline_adjusted_spectrum': results[0],
            'low_pass_data': results[1],
            'smoothed_baseline': results[2],
            'adjusted_baseline': results[3],
            'global_noise_level': results[4],
            'med_diff': results[5],
        }

        self.tofdata_subtracted = self.tofdata - self.baseline['adjusted_baseline'].reshape(1, -1)

        return self

    
    def determine_peak_width(self, plot_flag=True, mode='ransac', 
                             output_dir=None, plot_subdir=None, 
                             show_plot_flag=True, **kwargs) -> Deployment:
        """
        Fit peak width (FWHM) resolution functions across the mass spectrum.

        Analyzes local peak shapes across the baseline-adjusted reference spectrum using 
        RANSAC regression (`mode='ransac'`) or iterative peak fitting (`mode='iterative'`). 
        Derives linear peak width polynomials FWHM}(m) = S * m + B and mass-dependent 
        resolving power functions RP(m) = a * m^b.

        Parameters
        ----------
        plot_flag : bool, default=True
            If ``True``, generates diagnostic peak width and resolving power regression figures.
        mode : {'ransac', 'iterative'}, default='ransac'
            Regression algorithm mode:
            
            * ``'ransac'``: Uses RANSAC robust regression to exclude overlapping or satellite peak outliers.
            * ``'iterative'``: Uses iterative outlier rejection refinement.
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        **kwargs : dict
            Additional keyword arguments passed to underlying peak width fitting routines.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with peak width attributes populated:
                * ``peak_width_function`` : `numpy.poly1d` polynomial evaluating expected FWHM.
                * ``peak_width_coeffs`` : 1D array of polynomial coefficients [S, B].
                * ``min_peak_width_function`` : `numpy.poly1d` polynomial evaluating minimum FWHM floor.
                * ``min_peak_width_coeffs`` : 1D array of minimum FWHM polynomial coefficients.
                * ``resolving_power_function`` : Callable function `f(m)` evaluating resolving power (m / dm).

        Raises
        ------
        ValueError
            If :meth:`determine_baseline` has not been executed (`self.baseline is None`).
        """
        # Guard clause: Enforce requirement that baseline determination must be run first
        if getattr(self, 'baseline', None) is None:
            raise ValueError("No reference spectrum baseline found. Please call `determine_baseline()` first.")

        # Resolve base output plot directory
        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        # Resolve plot sub-directory prefix
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir

        # Combine base output directory with sub-directory prefix
        output_dir = os.path.join(output_dir, plot_subdir)
        kwargs['output_dir'] = output_dir
        kwargs['show_plot_flag'] = show_plot_flag
            
        # Local imports to prevent circular module dependencies
        from opentof.peak_width_shape import peak_width, ransac_peak_width

        # Execute selected peak width parameterization algorithm
        if mode == 'ransac':
            print("Determining peak width using RANSAC regression...")
            fit_coeffs, offset_coeffs, rp_popt = ransac_peak_width(
                reference_spectrum=self.baseline['baseline_adjusted_spectrum'],
                rs_mass_axis=self.reference['rs_mass_axis'],
                plot_flag=plot_flag,
                **kwargs
            )
        else:
            print("Determining peak width using iterative refinement...")
            fit_coeffs, rp_popt = peak_width(
                reference_spectrum=self.baseline['baseline_adjusted_spectrum'],
                rs_mass_axis=self.reference['rs_mass_axis'],
                plot_flag=plot_flag,
                **kwargs
            )
            offset_coeffs = fit_coeffs

        # Instantiate 1D NumPy polynomial objects from fitted coefficients
        fit_pwf = np.poly1d(fit_coeffs)
        off_pwf = np.poly1d(offset_coeffs)
        
        # Store primary peak width polynomial functions and coefficients on instance
        self.peak_width_function = fit_pwf
        self.peak_width_coeffs = fit_coeffs

        # Store offset adjusted minimum peak width floor and coefficients
        self.min_peak_width_function = off_pwf
        self.min_peak_width_coeffs = offset_coeffs

        # Construct mass-dependent resolving power callable function RP(m) = a * m^b
        self.resolving_power_function = lambda m: rp_popt[0] * (m ** rp_popt[1])
        
        # Return self instance to enable method chaining
        return self

    def determine_peak_shape(self, plot_flag=True, output_dir=None, 
                             plot_subdir=None, show_plot_flag=True,
                             **kwargs) -> Deployment:
        """
        Extract the empirical instrument peak line shape from the reference spectrum.

        Selects well-isolated, high-SNR peaks across the reference spectrum, aligns and 
        normalizes them onto a standardized z-space domain (z = (x - x_c) / FWHM), 
        fits a 1D cubic spline interpolator, and constructs a live callable empirical peak 
        shape function.

        Parameters
        ----------
        plot_flag : bool, default=True
            If ``True``, generates diagnostic empirical peak shape overlay figures.
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        **kwargs : dict
            Additional keyword arguments passed to :func:`peak_shape`.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with empirical peak shape attributes populated:
                * ``peak_shape_interp`` : `scipy.interpolate.interp1d` 1D cubic spline interpolator.
                * ``custom_peak_shape`` : Live callable function `f(x, A, x_c, FWHM)` for peak modeling.
                * ``custom_shape_area_ratio`` : Integrated area scaling ratio relative to a standard Gaussian profile.

        Raises
        ------
        ValueError
            If :meth:`determine_baseline` has not been executed (`self.baseline is None`).
        """
        # Guard clause: Enforce requirement that baseline determination must be run first
        if getattr(self, 'baseline', None) is None:
            raise ValueError("No reference spectrum baseline found. Please call `determine_baseline()` first.")

        # Resolve base output plot directory
        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        # Resolve plot sub-directory prefix
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir

        # Combine base output directory with sub-directory prefix
        output_dir = os.path.join(output_dir, plot_subdir)
        kwargs['output_dir'] = output_dir
        kwargs['show_plot_flag'] = show_plot_flag

        # Local import to prevent circular module dependencies
        from opentof.peak_width_shape import peak_shape

        # Execute empirical peak shape extraction algorithm
        custom_interp, custom_shape, area_ratio = peak_shape(
            reference_spectrum=self.baseline['baseline_adjusted_spectrum'],
            plot_flag=plot_flag,
            plot_name="peak_shape.png",
            shape_filename="custom_shape",
            **kwargs
        )
        
        # Store 1D spline interpolator, callable shape function, and area conversion ratio
        self.peak_shape_interp = custom_interp
        self.custom_peak_shape = custom_shape  # Live callable function f(x, A, x_c, FWHM)
        self.custom_shape_area_ratio = area_ratio
        
        # Return self instance to enable method chaining
        return self
    
    def populate_peak_list_and_isotopes(self, peak_list, **kwargs) -> Deployment:
        """
        Populate the target peak list dictionary and pre-calculate theoretical isotopic distributions.

        Calculates exact theoretical mass centers (m/z) and expected FWHM resolution widths 
        for each peak target in `peak_list`. Calls :func:`batch_isotope_masses` to construct 
        theoretical isotopic abundance patterns for all string chemical formulas.

        Parameters
        ----------
        peak_list : list of (str or float)
            Sequence of target chemical formulas (e.g., ``"C6H6+"``) or exact numeric m/z values.
        **kwargs : dict
            Additional keyword arguments passed to :func:`batch_isotope_masses` (e.g., `cutoff`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.peak_list` and `self.isotopes` populated:
                * ``peak_list`` : Dictionary containing keys `'peaks'` (input list), `'centers'` (1D m/z array), and `'fwhms'` (1D FWHM array).
                * ``isotopes`` : Dictionary mapping chemical formulas to lists of ``(mass, relative_intensity)`` tuples.
        """
        # Warning guard clause if peak width function has not been established prior to execution
        if getattr(self, 'peak_width_function', None) is None:
            print("[WARNING!] peak_width_function not defined.")
            print(" deployment.peak_list['fwhms'] will be undefined!")
            print(" Call `determine_peak_width()` or populate manually")
            print(" prior to fully constrained fitting!")

        # Local imports to prevent circular dependencies
        from opentof.isotopes import batch_isotope_masses
        from opentof.utils import return_mass

        # Compute theoretical isotopic abundance patterns for all peak list items
        isotope_mapping = batch_isotope_masses(
            peak_list=peak_list,
            **kwargs
        )

        peak_dict = {}
        peak_dict['peaks'] = peak_list

        peak_centers = []
        peak_fwhms = []
        
        # Calculate theoretical mass center and expected FWHM resolution for each peak target
        for peak in peak_list:
            pc = return_mass(peak)  # Calculate exact theoretical mass (m/z)
            peak_centers.append(pc)
            
            # Evaluate expected FWHM resolution width at mass center if function exists
            if self.peak_width_function is not None:
                peak_fwhms.append(self.peak_width_function(pc))
            else:
                peak_fwhms.append(np.nan)

        # Store numpy arrays of peak centers and FWHMs in dictionary
        peak_dict['centers'] = np.array(peak_centers)
        peak_dict['fwhms'] = np.array(peak_fwhms)

        # Assign peak list dictionary and isotope mappings to instance attributes
        self.peak_list = peak_dict
        self.isotopes = isotope_mapping
        
        # Return self instance to enable method chaining
        return self

    def _apply_isotope_reallocation(self, df):
        """
        Scale parent peak areas to reallocate total molecular isotopic abundance.

        Divides fitted monoisotopic parent peak area columns by their theoretical monoisotopic 
        abundance fraction (0.0 < f ≤ 1.0), scaling parent areas to reflect total molecular 
        signal sum across all isotopic variants.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing fitted peak result columns (e.g., ``f"{peak}_area"``).

        Returns
        -------
        df : pandas.DataFrame
            Updated DataFrame with scaled peak area columns reflecting total isotopic signal.
        """
        # Local import to prevent circular dependencies
        from opentof.isotopes import get_parent_fraction
        
        # Iterate over all targets in the active peak list
        for peak in self.peak_list['peaks']:
            # Theoretical isotope scaling applies strictly to chemical formula strings
            if isinstance(peak, str): 
                # Retrieve monoisotopic parent peak abundance fraction (0.0 to 1.0)
                fraction = get_parent_fraction(peak)
                area_col = f"{peak}_area"
                
                # Rescale monoisotopic area column to total molecular area: Area_total = Area_parent / fraction
                if area_col in df.columns and fraction > 0:
                    df[area_col] = df[area_col] / fraction
                    
        return df

    # TODO: progress bar is nonlinear!!
    def FFI_unconstrained(self, peak_type='gaussian', 
                          noise_level=None, 
                          noise_std_mult=10.0,
                          use_diagnostic_overrides=True,
                          verbose=True, tol=1e-4, **kwargs) -> Deployment:
        """
        Execute parallel unconstrained peak fitting and integration across all dataset spectra.

        Solves for peak amplitudes, mass centers (m/z), and FWHM widths simultaneously using 
        non-linear least squares optimization across all target peaks in `self.peak_list`. 
        Applies automated isotope abundance reallocation scaling to fitted peak areas and 
        stores the resulting wide-format DataFrame in `self.peak_data`.

        Parameters
        ----------
        peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
            Model line shape identifier passed to the fitting backend.
        noise_level : float or None, default=None
            Baseline noise standard deviation threshold (σ) used for noise gating. 
            If ``None``, noise_level is estimated downstream
        verbose : bool, default=True
            If ``True``, prints progress bars and diagnostic status updates.
        **kwargs : dict
            Additional keyword arguments passed to :func:`full_fitting_integration_unconstrained` 
            (e.g., `nm_search_range`, `max_iter`, `tol`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.peak_data` DataFrame populated.

        Raises
        ------
        ValueError
            If any required prerequisite pipeline step (:meth:`mass_calibration`, 
            :meth:`determine_baseline`, :meth:`determine_peak_width`, 
            :meth:`populate_peak_list_and_isotopes`, or :meth:`determine_peak_shape`) 
            has not been executed prior to calling this method.
        """
        # Safety Check 1: Ensure mass calibration parameters are populated
        if getattr(self, 'calibration', None) is None:
            raise ValueError("Mass calibration not run. Call `calibrate_mass()` first.")
            
        # Safety Check 2: Ensure baseline subtraction profile is calculated
        if getattr(self, 'baseline', None) is None:
            raise ValueError("Baseline not determined. Call `determine_baseline()` first.")
            
        # Safety Check 3: Ensure peak width function is parameterized
        if getattr(self, 'peak_width_function', None) is None:
            raise ValueError("Peak width not determined. Call `determine_peak_width()` first.")
            
        # Safety Check 4: Ensure target peak list and isotope mappings are populated
        if getattr(self, 'peak_list', None) is None:
            raise ValueError("Peak List not determined. Call `populate_peak_list_and_isotopes()` first.")
        
        # Local import to prevent circular module dependencies
        from opentof.peak_fitting import full_fitting_integration_unconstrained

        # Retrieve empirical custom peak shape callable if peak_type='custom'
        custom_shape = kwargs.pop('custom_shape', None)
        if peak_type == 'custom':
            if getattr(self, 'custom_peak_shape', None) is None:
                raise ValueError("Peak type is 'custom' but no custom shape is defined. "
                                 "Call `determine_peak_shape()` first.")
            custom_shape = self.custom_peak_shape

        # Check diagnostic overrides to update target peak list centers or initial guesses
        active_peaks = list(self.peak_list['peaks'])
        if use_diagnostic_overrides and hasattr(self, 'diagnostic_fits'):
            for key, diag_res in self.diagnostic_fits.items():
                if key.startswith("unconstrained_") and 'peaks' in diag_res:
                    for p in diag_res['peaks']:
                        if p['name'] not in active_peaks and not p['name'].startswith("Peak_"):
                            active_peaks.append(p['name'])
            
            # Synchronize peak list if custom peaks were added during diagnostic testing
            if len(active_peaks) != len(self.peak_list['peaks']):
                self.populate_peak_list_and_isotopes(active_peaks)

        # Execute out-of-core parallel unconstrained peak fitting driver
        result_df = full_fitting_integration_unconstrained(
            peak_list=self.peak_list['peaks'],
            tofdata_subtracted=self.tofdata_subtracted,
            calibration_results=self.calibration,
            sample_index_axis=self.sample_index_axis,
            tof_axis=self.tof_axis,
            peak_width_function=self.peak_width_function,
            peak_type=peak_type,
            custom_shape=custom_shape,
            noise_level=noise_level,
            noise_std_mult=noise_std_mult,
            chunk_size=self.chunk_size,
            verbose=verbose,
            tol=tol,
            **kwargs
        )

        # Scale monoisotopic peak areas to account for total molecular isotopic abundance
        result_df = self._apply_isotope_reallocation(result_df)
        
        # Store result DataFrame on instance
        self.peak_data = result_df

        # Return self instance to enable method chaining
        return self
    
    # TODO: progress bar is nonlinear!!
    def FFI_constrained(self, peak_type='gaussian', verbose=True, **kwargs) -> Deployment:
        """
        Execute parallel fully constrained peak fitting and integration across all dataset spectra.

        Holds target peak mass centers (m/z) and widths (FWHM) fixed at theoretical values 
        while solving for peak height amplitudes using Non-Negative Least Squares (NNLS). 
        Applies automated isotope abundance reallocation scaling to fitted peak areas and 
        stores the resulting wide-format DataFrame in `self.peak_data`.

        Parameters
        ----------
        peak_type : {'pseudo_voigt', 'gaussian', 'lorentzian', 'custom'}, default='pseudo_voigt'
            Model line shape identifier passed to the fitting backend.
        **kwargs : dict
            Additional keyword arguments passed to :func:`full_fitting_integration_constrained` 
            (e.g., `nm_search_range`, `verbose`).

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.peak_data` DataFrame populated.

        Raises
        ------
        ValueError
            If any required prerequisite pipeline step (:meth:`mass_calibration`, 
            :meth:`determine_baseline`, :meth:`determine_peak_width`, 
            :meth:`populate_peak_list_and_isotopes`, or :meth:`determine_peak_shape`) 
            has not been executed prior to calling this method.
        """
        # Safety Check 1: Ensure mass calibration parameters are populated
        if getattr(self, 'calibration', None) is None:
            raise ValueError("Mass calibration not run. Call `calibrate_mass()` first.")
            
        # Safety Check 2: Ensure baseline subtraction profile is calculated
        if getattr(self, 'baseline', None) is None:
            raise ValueError("Baseline not determined. Call `determine_baseline()` first.")
            
        # Safety Check 3: Ensure peak width function is parameterized
        if getattr(self, 'peak_width_function', None) is None:
            raise ValueError("Peak width not determined. Call `determine_peak_width()` first.")
            
        # Safety Check 4: Ensure target peak list and isotope mappings are populated
        if getattr(self, 'peak_list', None) is None:
            raise ValueError("Peak List not determined. Call `populate_peak_list_and_isotopes()` first.")
        
        # Local import to prevent circular module dependencies
        from opentof.peak_fitting import full_fitting_integration_constrained

        # Retrieve empirical custom peak shape callable if peak_type='custom'
        custom_shape = kwargs.pop('custom_shape', None)
        if peak_type == 'custom':
            if getattr(self, 'custom_peak_shape', None) is None:
                raise ValueError("Peak type is 'custom' but no custom shape is defined. "
                                 "Call `determine_peak_shape()` first.")
            custom_shape = self.custom_peak_shape

        # Execute out-of-core parallel constrained NNLS fitting driver
        result_df = full_fitting_integration_constrained(
            peak_list=self.peak_list['peaks'],
            tofdata_subtracted=self.tofdata_subtracted,
            calibration_results=self.calibration,
            sample_index_axis=self.sample_index_axis,
            tof_axis=self.tof_axis,
            peak_width_function=self.peak_width_function,
            peak_type=peak_type,
            custom_shape=custom_shape,
            chunk_size=self.chunk_size,
            verbose=verbose,
            **kwargs
        )

        # Scale monoisotopic peak areas to account for total molecular isotopic abundance
        result_df = self._apply_isotope_reallocation(result_df)

        # Store result DataFrame on instance
        self.peak_data = result_df

        # Return self instance to enable method chaining
        return self
    
    def generate_averaged_dataset(self, averaging_interval=300, 
                                  recalculate=False, **kwargs) -> Deployment:
        """
        Generate and cache the Global Averaged Dataset (GAD) across time intervals.

        Averages TOF spectral intensity matrices across standard acquisition scans inside 
        fixed time windows (`averaging_interval` seconds). Uses cached results if the requested 
        interval matches `self._averaged_dataset_interval` unless `recalculate=True`.

        Parameters
        ----------
        averaging_interval : int, default=300
            Time window duration in seconds for time-series spectrum averaging.
        recalculate : bool, default=False
            If ``True``, forces recomputation of the averaged dataset even if a cached tuple exists.
        **kwargs : dict
            Additional keyword arguments passed to :func:`generate_averaged_dataset` in `opentof.mass_calibration`.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self._averaged_dataset` populated as a tuple 
            ``(averaged_data, midpoint_timestamps, interval_indices)``.
        """
        # --- Step 1: Cache Validation & Drift Guard ---
        # Reuse existing cache if interval matches and recalculation is not forced
        if self._averaged_dataset is not None:
            if self._averaged_dataset_interval == averaging_interval and not recalculate:
                print(f"ℹ️ Using cached averaged dataset ({averaging_interval}s interval).")
                return self
            elif self._averaged_dataset_interval != averaging_interval:
                print(f"⚠️ New averaging_interval requested. Existing averaged dataset is {self._averaged_dataset_interval}s, "
                      f"but you requested {averaging_interval}s. Forcing recomputation...")
                recalculate = True

        # --- Step 2: Parameter Inspection & Backend Execution ---
        from opentof.mass_calibration import generate_averaged_dataset as backend_generate_averaged_dataset
        import inspect

        # Inspect signature to filter relevant keyword arguments
        gad_sig = inspect.signature(backend_generate_averaged_dataset).parameters
        gad_kwargs = {k: v for k, v in kwargs.items() if k in gad_sig}

        print(f"⏳ Generating interval-averaged dataset ({averaging_interval} sec windows)...")
        
        # Execute backend dataset averaging routine
        averaged_data, midpoint_timestamps, interval_indices = backend_generate_averaged_dataset(
            tofdata=self.tofdata,
            timestamps=self.timestamps,
            standard_acq_data=self.standard_acquisition_data,
            averaging_interval=averaging_interval,
            chunk_size=self.chunk_size,
            **gad_kwargs
        )

        # --- Step 3: Populate Instance Cache ---
        # Cache averaged matrix, midpoint timestamps, and interval mapping index array
        self._averaged_dataset = (averaged_data, midpoint_timestamps, interval_indices)
        self._averaged_dataset_interval = averaging_interval

        # Populate or update calibration dictionary metadata
        if getattr(self, 'calibration', None) is None:
            print("[Warning] typical Deployment.calibration dataset can only be partially populated.")
            self.calibration = {
                "interval_indices": interval_indices,
                "midpoint_timestamps": midpoint_timestamps,
            }
        else:
            if "interval_indices" not in self.calibration:
                self.calibration["interval_indices"] = interval_indices
            if "midpoint_timestamps" not in self.calibration:
                self.calibration["midpoint_timestamps"] = midpoint_timestamps
        
        print("✅ Averaged dataset successfully stored in deployment object. (Deployment._averaged_dataset and Deployment._averaged_dataset_interval)")
        
        # Return self instance to enable method chaining
        return self

    def plot_peak_data_for(self, peak_name, parameter="area",
                           use_std_acq_data=True,
                           color="deepskyblue",
                           dpi=150):
        """
        Plot the time-series profile of a specific peak parameter from fitting results.

        Extracts and displays time-series trends for integrated area, amplitude, center 
        mass, or FWHM resolution from `self.peak_data` for a target compound string.

        Parameters
        ----------
        peak_name : str
            Target compound or ion formula string matching an entry in `self.peak_list['peaks']`.
        parameter : {'area', 'amplitude', 'center', 'fwhm'}, default='area'
            Target peak parameter to visualize:
            
            * ``'area'``: Integrated peak area (ions/s).
            * ``'amplitude'``: Peak height amplitude (ions/s).
            * ``'center'``: Fitted peak center mass (m/z).
            * ``'fwhm'``: Fitted FWHM width in mass units.
        use_std_acq_data : bool, default=True
            If ``True``, filters out non-standard acquisition scans (auto-zeros, cals, corrupt scans) 
            using `self.standard_acquisition_data`.
        color : str, default="deepskyblue"
            Matplotlib line color string.
        dpi : int, default=150
            Figure resolution dots-per-inch.

        Raises
        ------
        ValueError
            If `self.peak_data` is unpopulated or if the requested parameter key is missing 
            from DataFrame columns.
        """
        # Guard clause: Verify peak_data DataFrame is populated
        if getattr(self, 'peak_data', None) is None:
            raise ValueError("deployment.peak_data not populated. Please call `FFI_unconstrained()` or `FFI_constrained()` first.")

        # Map requested parameter string to DataFrame column suffix and axis label
        if parameter == "fwhm":
            parameter_extension = "_fwhm_mass"
            y_axis_label = "FWHM (mass space)"
        elif parameter == "center":
            parameter_extension = "_center_mass"
            y_axis_label = "Peak Center (mass space)"
        elif parameter == "amplitude":
            parameter_extension = "_amplitude"
            y_axis_label = "Peak Amplitude (ions/s)"
        else:
            parameter_extension = "_area"
            y_axis_label = "Integrated Area (ions/s)"
        
        # Resolve target DataFrame column key name
        key_name = peak_name + parameter_extension
        if key_name not in self.peak_data.columns:
            raise ValueError(f"Column key: '{key_name}' not found in peak_data columns."
                             "Check that the peak string matches exactly or perform "
                             "unconstrained fitting to populate additional fields.")

        # Filter time-series vectors using standard acquisition mask if requested
        if use_std_acq_data:
            print("Filtering Time Series using deployment.standard_acquisition_data")
            y = self.peak_data[key_name][self.standard_acquisition_data]
            x = self.timestamps[self.standard_acquisition_data]
        else:
            y = self.peak_data[key_name]
            x = self.timestamps

        # Render time-series line plot
        plt.figure(figsize=(12, 6), dpi=dpi)
        plt.plot(x, y, linestyle='-', color=color)
        plt.title(f"Deployment.peak_data: {key_name}")
        plt.xlabel("Timestamps")
        plt.ylabel(y_axis_label)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.show()


    def fit_nm_constrained(self, nominal_mass, ms_i=None, initial_masses=None, 
                           peak_type='custom', target_spectrum=None, 
                           target_mass_axis=None, target_baseline=None,
                           nm_search_range=0.5, subtract_isotopes=True,
                           plot_flag=True, show_plot_flag=True, save_plot_flag=False, 
                           output_dir=None, plot_filename=None, **kwargs):
        """
        Perform diagnostic constrained (Non-Negative Least Squares) peak fitting on a sliced nominal mass segment.

        Slices a target nominal mass window (:math:`m/z ~NM +- nm_search_range}`) from a target spectrum,
        reconstructs and subtracts secondary isotopic interference signals on the fly, resolves target peak centers and widths 
        from `self.peak_list` (or accepts custom `initial_masses`), and solves for peak height amplitudes using Non-Negative 
        Least Squares (NNLS).

        Parameters
        ----------
        nominal_mass : int or float
            Target nominal mass integer or center value (:math:`m/z`) to inspect.
        ms_i : int or None, default=None
            0-based writebuf/spectrum index along the dataset time-series axis to inspect. If ``None``, defaults to `target_spectrum` or `self.reference['reference_spectrum']`.
        initial_masses : sequence of (str or float) or None, default=None
            Sequence of chemical formula strings or exact numeric $m/z$ values to constrain. If ``None``, resolves targets from `self.peak_list`.
        peak_type : {'custom', 'pseudo_voigt', 'gaussian', 'lorentzian'}, default='custom'
            Model line shape identifier passed to the basis matrix generator.
        target_spectrum : numpy.ndarray or None, default=None
            1D array of spectral intensity values to evaluate. Defaults to `self.reference['reference_spectrum']` if ``None``.
        target_mass_axis : numpy.ndarray or None, default=None
            1D array of calibrated mass-to-charge ($m/z$) coordinates matching `target_spectrum`. Defaults to reference or first-guess mass axis if ``None``.
        target_baseline : numpy.ndarray or None, default=None
            1D array representing baseline continuum intensity to subtract from `target_spectrum`. Defaults to `self.baseline['adjusted_baseline']`.
        nm_search_range : float, default=0.5
            Half-width search window size in $m/z$ units ($[NM - Δ, NM + Δ]$).
        subtract_isotopes : bool, default=True
            If ``True``, reconstructs and subtracts secondary isotopic signals originating from lower-mass parent peaks prior to fitting.
        plot_flag : bool, default=True
            If ``True``, renders a 2-panel diagnostic figure displaying the spectral fit and residual error.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        save_plot_flag : bool, default=False
            If ``True``, exports generated diagnostic figures to disk.
        output_dir : str or pathlib.Path or None, default=None
            Directory path to save exported plot figures. Defaults to `self.plot_dir` or default plot directory if ``None``.
        plot_filename : str or None, default=None
            Filename for saved diagnostic plot images. Defaults to ``constrained_fit_nm<NM>.png`` if ``None``.
        **kwargs : dict
            Additional keyword arguments passed to internal solvers or shape selectors.

        Returns
        -------
        result : dict or None
            A diagnostic results dictionary stored on the instance at `self.diagnostic_fits['constrained_<NM>']`:
                * ``'nominal_mass'`` : Sliced nominal mass center value.
                * ``'mz_segment'`` : 1D array of mass coordinates in the sliced window.
                * ``'int_segment'`` : 1D array of baseline-subtracted intensities.
                * ``'total_fit'`` : 1D array of cumulative fitted model intensity.
                * ``'peaks'`` : List of dictionaries containing individual fitted peak parameters.
                * ``'fit_params'`` : 1D array of optimized NNLS peak height amplitudes.
        """
        from scipy.optimize import nnls
        from opentof.mass_calibration import get_mass_axis_for_ms_i
        from opentof.peak_fitting import peak_function_selector, fit_unconstrained_peaks
        from opentof.utils import return_mass, get_nm_segment_data, ensure_dir, get_default_plot_dir
        from opentof.isotopes import isotope_signal_on_axis

        # --- STEP 1: RESOLVE TARGET SPECTRUM & MASS AXIS ---
        # Case A: Inspect a specific spectrum index (ms_i) along the dataset time-series
        if ms_i is not None:
            spectra = self.tofdata_subtracted[ms_i, :] if getattr(self, 'tofdata_subtracted', None) is not None else self.tofdata[ms_i, :]
            if hasattr(spectra, 'compute'):
                spectra = spectra.compute()
            spectra_mass_axis = get_mass_axis_for_ms_i(ms_i, self.calibration, self.sample_index_axis)

        # Case B: Default to the global reference spectrum if target_spectrum is omitted
        elif target_spectrum is None:
            if getattr(self, 'reference', None) is None:
                raise ValueError("Deployment.reference is not populated. Call `define_reference_spectrum()` first or pass `target_spectrum` / `ms_i`.")
            spectra = self.reference['reference_spectrum']
            spectra_mass_axis = self.reference['rs_mass_axis']
            if target_baseline is None and getattr(self, 'baseline', None) is not None:
                spectra = spectra - self.baseline['adjusted_baseline']

        # Case C: Use user-provided target_spectrum and first-guess mass axis fallback
        elif target_mass_axis is None:
            spectra_mass_axis = self.first_guess_mass_axis

        # Subtract baseline if explicitly supplied alongside custom target_spectrum
        if target_baseline is not None and ms_i is None:
            spectra = spectra - target_baseline

        # --- STEP 2: ON-THE-FLY SECONDARY ISOTOPE RECONSTRUCTION & SUBTRACTION ---
        total_isotope_signal = np.zeros_like(spectra_mass_axis)
        custom_shape = kwargs.get('custom_shape', getattr(self, 'custom_peak_shape', None))

        if subtract_isotopes and getattr(self, 'peak_list', None) is not None:
            for peak in self.peak_list['peaks']:
                c_mass = return_mass(peak)
                
                # Evaluate lower-mass parent peaks capable of generating isotopic overlap in the target window
                if c_mass is not None and int(np.round(c_mass)) < nominal_mass:
                    parent_amp = 0.0

                    # 1. Attempt to retrieve pre-fitted parent peak amplitude from peak_data if ms_i is provided
                    if ms_i is not None and getattr(self, 'peak_data', None) is not None:
                        try:
                            fitted_row = self.peak_data.loc[self.peak_data['MS_index'] == ms_i].iloc[0]
                            amp_col = f"{peak}_amplitude"
                            if amp_col in fitted_row and not pd.isna(fitted_row[amp_col]):
                                parent_amp = fitted_row[amp_col]
                        except (IndexError, AttributeError):
                            parent_amp = 0.0

                    # 2. Fallback: Fit parent peak on the fly directly from target_spectrum
                    if parent_amp <= 0:
                        fwhm_g = self.peak_width_function(c_mass) if getattr(self, 'peak_width_function', None) else 0.03
                        p_mask = (spectra_mass_axis >= c_mass - fwhm_g * 2.0) & (spectra_mass_axis <= c_mass + fwhm_g * 2.0)
                        
                        if np.any(p_mask):
                            p_mz = spectra_mass_axis[p_mask]
                            p_sig = spectra[p_mask]
                            
                            if np.max(p_sig) > 0:
                                try:
                                    p_popt = fit_unconstrained_peaks(
                                        x_axis=p_mz, signal=p_sig,
                                        centers_guess=[c_mass], fwhms_guess=[fwhm_g],
                                        amplitudes_guess=[np.max(p_sig)],
                                        peak_type=peak_type, custom_shape=custom_shape,
                                        center_wiggle=0.1
                                    )
                                    parent_amp = max(0.0, p_popt[0])
                                except Exception:
                                    parent_amp = max(0.0, np.max(p_sig))

                    # Reconstruct continuous line shape profile for secondary isotopic variants
                    if parent_amp > 0 and isinstance(peak, str):
                        iso_sig = isotope_signal_on_axis(
                            formula=peak,
                            mass_axis=spectra_mass_axis,
                            parent_amplitude=parent_amp,
                            peak_width_function=self.peak_width_function,
                            peak_type=peak_type,
                            custom_shape=custom_shape
                        )
                        total_isotope_signal += iso_sig

        # Subtract isotopic interference background and enforce non-negative floor
        subtracted_spectra = np.maximum(spectra - total_isotope_signal, 0)

        # --- STEP 3: SLICE NOMINAL MASS WINDOW ---
        int_segment, mz_segment, _ = get_nm_segment_data(
            nominal_mass, subtracted_spectra, spectra_mass_axis, self.tof_axis, nm_search_range=nm_search_range
        )

        # --- STEP 4: RESOLVE CANDIDATE TARGET PEAKS ---
        peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)

        target_peaks = []
        if initial_masses is not None:
            for item in initial_masses:
                c_mass = return_mass(item)
                fwhm = self.peak_width_function(c_mass) if getattr(self, 'peak_width_function', None) else 0.03
                if peak_type == 'custom' and hasattr(peak_func, 'gauss_to_ps'):
                    fwhm *= peak_func.gauss_to_ps
                target_peaks.append({'name': str(item), 'mass': c_mass, 'fwhm': fwhm})
        elif getattr(self, 'peak_list', None) is not None:
            for peak in self.peak_list['peaks']:
                c_mass = return_mass(peak)
                if c_mass is not None and (nominal_mass - nm_search_range) <= c_mass <= (nominal_mass + nm_search_range):
                    fwhm = self.peak_width_function(c_mass) if getattr(self, 'peak_width_function', None) else 0.03
                    if peak_type == 'custom' and hasattr(peak_func, 'gauss_to_ps'):
                        fwhm *= peak_func.gauss_to_ps
                    target_peaks.append({'name': str(peak), 'mass': c_mass, 'fwhm': fwhm})

        # --- STEP 5: BUILD BASIS MATRIX & SOLVE NNLS CONSTRAINED SYSTEM ---
        heights = []
        total_fit = np.zeros_like(mz_segment)
        fitted_peaks = []
        if target_peaks:
            M = np.zeros((len(mz_segment), len(target_peaks)))
            for q, p_info in enumerate(target_peaks):
                if peak_type == 'pseudo_voigt':
                    M[:, q] = peak_func(mz_segment, 1.0, p_info['mass'], p_info['fwhm'], 0.5)
                else:
                    M[:, q] = peak_func(mz_segment, 1.0, p_info['mass'], p_info['fwhm'])
            
            # Solve Non-Negative Least Squares (NNLS) for peak heights
            heights, _ = nnls(M, int_segment)
            total_fit = M @ heights

            for q, p_info in enumerate(target_peaks):
                curve = M[:, q] * heights[q]
                fitted_peaks.append({
                    'name': p_info['name'],
                    'amplitude': heights[q],
                    'center_mass': p_info['mass'],
                    'fwhm_mass': p_info['fwhm'],
                    'curve': curve
                })

        # --- STEP 6: REDESIGNED DIAGNOSTIC VISUALIZATION (3:1 DUAL PANEL) ---
        if plot_flag:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, 
                figsize=(10, 6), 
                sharex=True, 
                gridspec_kw={'height_ratios': [3, 1]}, 
                dpi=150
            )

            raw_seg, _, _ = get_nm_segment_data(nominal_mass, spectra, spectra_mass_axis, self.tof_axis, nm_search_range=nm_search_range)
            iso_seg, _, _ = get_nm_segment_data(nominal_mass, total_isotope_signal, spectra_mass_axis, self.tof_axis, nm_search_range=nm_search_range)

            # --- TOP PANEL: Main Spectral Fit & Component Curves ---
            has_iso = subtract_isotopes and np.any(iso_seg > 0)
            if has_iso:
                ax1.plot(mz_segment, raw_seg, label='Raw Signal', color='gray', lw=1.2, alpha=0.5, linestyle=':')
                ax1.plot(mz_segment, int_segment, label='Target Signal (Deisotoped)', color='black', lw=1.8)
            else:
                ax1.plot(mz_segment, int_segment, label='Target Signal', color='black', lw=1.8)

            ax1.plot(mz_segment, total_fit, label='Total NNLS Fit', color='forestgreen', lw=2.0, linestyle='--')
            
            cmap = plt.get_cmap('tab10')
            for q, p_info in enumerate(fitted_peaks):
                col = cmap(q % 10)
                ax1.plot(mz_segment, p_info['curve'], color=col, linestyle='-', lw=1.5, alpha=0.8, 
                         label=f"{p_info['name']} (H: {p_info['amplitude']:.1f})")
                ax1.axvline(p_info['center_mass'], color=col, linestyle=':', alpha=0.6)

            ax1.set_title(f"Constrained Diagnostic Fit (m/z ~ {nominal_mass})", fontsize=11)
            ax1.set_ylabel("Intensity (ions/s)")
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.legend(loc='upper right', fontsize='small')

            # --- BOTTOM PANEL: Fit Residuals & Subtracted Isotope Profile ---
            residuals = int_segment - total_fit
            ax2.plot(mz_segment, residuals, color='crimson', lw=1.2, label='Fit Residuals')
            ax2.axhline(0, color='black', lw=0.8, linestyle='--')

            if has_iso:
                ax2.plot(mz_segment, iso_seg, color='darkorange', lw=1.5, label='Subtracted Isotope Signal')
                ax2.fill_between(mz_segment, 0, iso_seg, color='darkorange', alpha=0.2)
                ax2.set_ylabel("Residuals / Isotopes")
            else:
                ax2.set_ylabel("Residual Error")

            ax2.set_xlabel("m/z")
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='upper right', fontsize='x-small')

            plt.tight_layout()

            if save_plot_flag:
                if output_dir is None:
                    output_dir = getattr(self, 'plot_dir', None) or get_default_plot_dir()
                ensure_dir(output_dir)
                if plot_filename is None:
                    plot_filename = f"constrained_fit_nm{nominal_mass}.png"
                plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')

            if show_plot_flag:
                plt.show()
            plt.close()

        # --- STEP 7: PACKAGE RESULTS & PERSIST TO DEPLOYMENT ---
        result = {
            'nominal_mass': nominal_mass,
            'mz_segment': mz_segment,
            'int_segment': int_segment,
            'total_fit': total_fit,
            'peaks': fitted_peaks,
            'fit_params': np.array(heights)
        }

        self.diagnostic_fits[f"constrained_{nominal_mass}"] = result
        return result


    def fit_nm_unconstrained(self, nominal_mass, ms_i=None, initial_masses=None, 
                             peak_type='custom', target_spectrum=None, 
                             target_mass_axis=None, target_baseline=None,
                             nm_search_range=0.5, subtract_isotopes=True,
                             plot_flag=True, show_plot_flag=True, save_plot_flag=False, 
                             output_dir=None, plot_filename=None, 
                             smooth_factor=-1, smooth_function='inverse linear',
                             noise_level=None, noise_std_mult=10.0,
                             deriv_threshold=0.0, rel_filter=None, abs_filter=None,
                             max_peaks=12, tol=1e-8, **kwargs):
        """
        Perform diagnostic unconstrained non-linear least squares peak fitting on a sliced nominal mass segment.

        Slices a target nominal mass window (:math:`m/z ~NM +- nm_search_range`) from a target spectrum,
        reconstructs and subtracts secondary isotopic interference signals on the fly, discovers peak candidate positions via 2nd 
        derivative curvature analysis (or accepts explicit user mass guesses), and optimizes peak height, center, and width parameters 
        simultaneously using non-linear least squares.

        Parameters
        ----------
        nominal_mass : int or float
            Target nominal mass integer or center value (:math:`m/z`) to inspect.
        ms_i : int or None, default=None
            0-based writebuf/spectrum index along the dataset time-series axis to inspect. If ``None``, defaults to `target_spectrum` or `self.reference['reference_spectrum']`.
        initial_masses : sequence of (str or float) or None, default=None
            Sequence of chemical formula strings or exact numeric $m/z$ values to serve as initial center position guesses. If ``None``, peak candidates are discovered automatically.
        peak_type : {'custom', 'pseudo_voigt', 'gaussian', 'lorentzian'}, default='custom'
            Model line shape identifier passed to the optimization backend.
        target_spectrum : numpy.ndarray or None, default=None
            1D array of spectral intensity values to evaluate. Defaults to `self.reference['reference_spectrum']` if ``None``.
        target_mass_axis : numpy.ndarray or None, default=None
            1D array of calibrated mass-to-charge ($m/z$) coordinates matching `target_spectrum`. Defaults to reference or first-guess mass axis if ``None``.
        target_baseline : numpy.ndarray or None, default=None
            1D array representing baseline continuum intensity to subtract from `target_spectrum`. Defaults to `self.baseline['adjusted_baseline']`.
        nm_search_range : float, default=0.5
            Half-width search window size in $m/z$ units ($[NM - Δ, NM + Δ]$).
        subtract_isotopes : bool, default=True
            If ``True``, reconstructs and subtracts secondary isotopic signals originating from lower-mass parent peaks prior to fitting.
        plot_flag : bool, default=True
            If ``True``, renders a 2-panel diagnostic figure displaying the spectral fit and residual error.
        show_plot_flag : bool, default=True
            If ``True``, displays generated diagnostic plots interactively.
        save_plot_flag : bool, default=False
            If ``True``, exports generated diagnostic figures to disk.
        output_dir : str or pathlib.Path or None, default=None
            Directory path to save exported plot figures. Defaults to `self.plot_dir` or default plot directory if ``None``.
        plot_filename : str or None, default=None
            Filename for saved diagnostic plot images. Defaults to ``unconstrained_fit_nm<NM>.png`` if ``None``.
        smooth_factor : int, default=-1
            Window length for Savitzky-Golay derivative filtering. If ``-1``, calculated dynamically from local Signal-to-Noise Ratio (SNR).
        smooth_function : {'inverse linear', 'logistic', 'hyperbolic decay', 'double logarithmic', 'logarithmic'}, default='inverse linear'
            Dynamic window selection equation used when `smooth_factor=-1`.
        noise_level : float or None, default=None
            Background noise standard deviation ($σ$). Estimated automatically from non-peak channels if ``None``.
        noise_std_mult : float, default=10.0
            Multiplier applied to `noise_level` to establish the absolute peak detection threshold floor ($N * σ$).
        deriv_threshold : float, default=0.0
            Minimum 2nd derivative prominence threshold expressed as a percentage (0–100%) of the maximum observed curvature.
        rel_filter : float or None, default=None
            Minimum relative intensity threshold in [0.0, 1.0] required to qualify candidate peaks.
        abs_filter : float or None, default=None
            Minimum absolute intensity threshold (ions/s) required to qualify candidate peaks.
        max_peaks : int, default=12
            Maximum number of candidate peaks allowed per nominal mass window to prevent solver overload.
        tol : float, default=1e-8
            Convergence tolerance passed to the Trust Region Reflective (TRF) non-linear solver.
        **kwargs : dict
            Additional keyword arguments passed to :func:`fit_unconstrained_peaks`.

        Returns
        -------
        result : dict or None
            A diagnostic results dictionary stored on the instance at `self.diagnostic_fits['unconstrained_<NM>']`:
                * ``'nominal_mass'`` : Sliced nominal mass center value.
                * ``'mz_segment'`` : 1D array of mass coordinates in the sliced window.
                * ``'int_segment'`` : 1D array of baseline-subtracted intensities.
                * ``'total_fit'`` : 1D array of cumulative fitted model intensity.
                * ``'peaks'`` : List of dictionaries containing individual fitted peak parameters.
                * ``'fit_params'`` : Raw 1D parameter vector returned by the optimization solver.
            Returns ``None`` if no spectral channels or valid peak candidates are identified in the segment.
        """
        from scipy.signal import find_peaks, savgol_filter
        from opentof.utils import get_nm_segment_data, ensure_dir, get_default_plot_dir, return_mass
        from opentof.peak_fitting import fit_unconstrained_peaks, peak_function_selector, calculate_detection_threshold
        from opentof.mass_calibration import get_mass_axis_for_ms_i
        from opentof.isotopes import isotope_signal_on_axis

        # --- STEP 1: RESOLVE TARGET SPECTRUM & MASS AXIS ---
        # Case A: Inspect a specific spectrum index (ms_i) along the dataset time-series
        if ms_i is not None:
            target_spectrum = self.tofdata_subtracted[ms_i, :] if getattr(self, 'tofdata_subtracted', None) is not None else self.tofdata[ms_i, :]
            if hasattr(target_spectrum, 'compute'):
                target_spectrum = target_spectrum.compute()
            target_mass_axis = get_mass_axis_for_ms_i(ms_i, self.calibration, self.sample_index_axis)

        # Case B: Default to the global reference spectrum if target_spectrum is omitted
        elif target_spectrum is None:
            if getattr(self, 'reference', None) is None:
                raise ValueError("Deployment.reference is not populated. Call `define_reference_spectrum()` first or pass `target_spectrum` / `ms_i`.")
            target_spectrum = self.reference['reference_spectrum']
            target_mass_axis = self.reference['rs_mass_axis']
            if target_baseline is None and getattr(self, 'baseline', None) is not None:
                target_spectrum = target_spectrum - self.baseline['adjusted_baseline']

        # Case C: Use user-provided target_spectrum and first-guess mass axis fallback
        elif target_mass_axis is None:
            target_mass_axis = self.first_guess_mass_axis

        # Subtract baseline if explicitly supplied alongside custom target_spectrum
        if target_baseline is not None and ms_i is None:
            target_spectrum = target_spectrum - target_baseline

        # --- STEP 2: ON-THE-FLY SECONDARY ISOTOPE RECONSTRUCTION & SUBTRACTION ---
        total_isotope_signal = np.zeros_like(target_mass_axis)
        custom_shape = kwargs.get('custom_shape', getattr(self, 'custom_peak_shape', None))

        if subtract_isotopes and getattr(self, 'peak_list', None) is not None:
            for peak in self.peak_list['peaks']:
                c_mass = return_mass(peak)
                
                # Evaluate lower-mass parent peaks capable of generating isotopic overlap in the target window
                if c_mass is not None and int(np.round(c_mass)) < nominal_mass:
                    parent_amp = 0.0

                    # 1. Attempt to retrieve pre-fitted parent peak amplitude from peak_data if ms_i is provided
                    if ms_i is not None and getattr(self, 'peak_data', None) is not None:
                        try:
                            fitted_row = self.peak_data.loc[self.peak_data['MS_index'] == ms_i].iloc[0]
                            amp_col = f"{peak}_amplitude"
                            if amp_col in fitted_row and not pd.isna(fitted_row[amp_col]):
                                parent_amp = fitted_row[amp_col]
                        except (IndexError, AttributeError):
                            parent_amp = 0.0

                    # 2. Fallback: Fit parent peak on the fly directly from target_spectrum
                    if parent_amp <= 0:
                        fwhm_g = self.peak_width_function(c_mass) if getattr(self, 'peak_width_function', None) else 0.03
                        p_mask = (target_mass_axis >= c_mass - fwhm_g * 2.0) & (target_mass_axis <= c_mass + fwhm_g * 2.0)
                        
                        if np.any(p_mask):
                            p_mz = target_mass_axis[p_mask]
                            p_sig = target_spectrum[p_mask]
                            
                            if np.max(p_sig) > 0:
                                try:
                                    p_popt = fit_unconstrained_peaks(
                                        x_axis=p_mz, signal=p_sig,
                                        centers_guess=[c_mass], fwhms_guess=[fwhm_g],
                                        amplitudes_guess=[np.max(p_sig)],
                                        peak_type=peak_type, custom_shape=custom_shape,
                                        center_wiggle=0.1
                                    )
                                    parent_amp = max(0.0, p_popt[0])
                                except Exception:
                                    parent_amp = max(0.0, np.max(p_sig))

                    # Reconstruct continuous line shape profile for secondary isotopic variants
                    if parent_amp > 0 and isinstance(peak, str):
                        iso_sig = isotope_signal_on_axis(
                            formula=peak,
                            mass_axis=target_mass_axis,
                            parent_amplitude=parent_amp,
                            peak_width_function=self.peak_width_function,
                            peak_type=peak_type,
                            custom_shape=custom_shape
                        )
                        total_isotope_signal += iso_sig

        # Subtract isotopic interference background and enforce non-negative floor
        subtracted_spectrum = np.maximum(target_spectrum - total_isotope_signal, 0)

        # --- STEP 3: SLICE NOMINAL MASS WINDOW ---
        int_segment, mz_segment, tof_segment = get_nm_segment_data(
            nominal_mass, subtracted_spectrum, target_mass_axis, self.tof_axis, nm_search_range=nm_search_range
        )

        if len(int_segment) == 0:
            print(f"No spectral data found around nominal mass {nominal_mass}.")
            return None

        # --- STEP 4: NOISE LEVEL & ADAPTIVE SMOOTHING ---
        noise_dict = calculate_detection_threshold(intensity_axis=int_segment, noise_level=noise_level, noise_std_mult=noise_std_mult)

        # Calculate dynamic Savitzky-Golay filter window size based on SNR if smooth_factor=-1
        if smooth_factor == -1:
            snr = noise_dict['snr']
            smoothing_models = {
                'logistic': 3 + 8 / (1 + np.exp(snr - 5)),
                'hyperbolic decay': 3 + 20 / (snr + 1),
                'double logarithmic': 3 + np.log1p(np.log1p(20 / (snr + 1))),
                'logarithmic': 9 - 0.5 * np.log1p(snr),
                'inverse linear': 3 + 8 / (1 + snr)
            }
            smf = int(np.clip(smoothing_models.get(smooth_function, 3 + 8 / (1 + snr)), 3, 15))
            smooth_factor = smf if smf % 2 == 1 else smf + 1

        # Enforce odd integer window bounds within array slicing limits
        if smooth_factor >= len(int_segment):
            smooth_factor = max(3, len(int_segment) - (1 if len(int_segment) % 2 == 0 else 2))

        # --- STEP 5: PEAK DISCOVERY & THRESHOLD FILTERING ---
        if initial_masses is None:
            # Evaluate 2nd derivative curvature on normalized intensity segment
            norm_intensity = int_segment / (np.max(int_segment) + 1e-8)
            deriv = savgol_filter(norm_intensity, window_length=smooth_factor, polyorder=2, deriv=2)
            neg_deriv = -deriv  # Negate so concave-down peaks become positive maxima

            # Determine spatial separation constraint based on expected peak resolution
            expected_fwhm = self.peak_width_function(nominal_mass) if getattr(self, 'peak_width_function', None) else 0.03
            mz_step = np.mean(np.diff(mz_segment)) if len(mz_segment) > 1 else 0.001
            min_index_separation = max(1, int(np.ceil((expected_fwhm * 0.5) / mz_step)))

            raw_candidate_idxs, _ = find_peaks(neg_deriv, distance=min_index_separation)

            if len(raw_candidate_idxs) == 0:
                print(f"No peak candidates found in 2nd derivative for m/z ~ {nominal_mass}.")
                return None

            candidate_idxs = np.array(raw_candidate_idxs)
            mask = np.ones_like(candidate_idxs, dtype=bool)

            # Apply noise detection threshold gate
            mask &= int_segment[candidate_idxs] >= noise_dict['detection_threshold']

            # Apply relative and absolute intensity threshold filters
            if abs_filter is not None:
                mask &= int_segment[candidate_idxs] >= abs_filter
            if rel_filter is not None:
                mask &= norm_intensity[candidate_idxs] >= rel_filter

            # Filter candidates by 2nd derivative prominence percentage
            if deriv_threshold > 0.0 and len(candidate_idxs) > 0:
                max_deriv_val = np.max(neg_deriv[candidate_idxs])
                mask &= neg_deriv[candidate_idxs] >= (deriv_threshold / 100.0) * max_deriv_val

            valid_peaks = candidate_idxs[mask]

            if len(valid_peaks) == 0:
                print(f"No peak candidates survived noise thresholding ({noise_std_mult}σ) at m/z ~ {nominal_mass}.")
                return None

            # Sort candidate peaks by intensity descending and cap total candidates
            sorted_peaks = valid_peaks[np.argsort(int_segment[valid_peaks])[::-1]]
            if max_peaks is not None:
                sorted_peaks = sorted_peaks[:max_peaks]

            initial_masses = [mz_segment[ind] for ind in np.sort(sorted_peaks)]

        # --- STEP 6: FORMULATE INITIAL GUESSES & UNCONSTRAINED FIT ---
        c_guess = []
        for m in initial_masses:
            if isinstance(m, str) and getattr(self, 'peak_list', None) and self.peak_list and 'peaks' in self.peak_list and m in self.peak_list['peaks']:
                idx = self.peak_list['peaks'].index(m)
                c_guess.append(float(self.peak_list['centers'][idx]))
            else:
                c_guess.append(float(return_mass(m)))

        f_guess = [float(self.peak_width_function(m)) if getattr(self, 'peak_width_function', None) else 0.03 for m in c_guess]
        a_guess = [max(float(np.interp(m, mz_segment, int_segment)), 1e-3) for m in c_guess]

        if peak_type == 'custom' and custom_shape is None:
            peak_type = 'gaussian'

        peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)

        # Solve non-linear least squares optimization using TRF algorithm
        fit_params = fit_unconstrained_peaks(
            x_axis=mz_segment,
            signal=int_segment,
            centers_guess=c_guess,
            fwhms_guess=f_guess,
            amplitudes_guess=a_guess,
            peak_type=peak_type,
            custom_shape=custom_shape,
            tol=tol,
            **kwargs
        )

        # --- STEP 7: UNPACK FITTED PARAMETERS & RECONSTRUCT CURVES ---
        is_pv = (peak_type == 'pseudo_voigt')
        n_p = 4 if is_pv else 3
        num_fitted_peaks = len(fit_params) // n_p

        fitted_peaks = []
        total_fit = np.zeros_like(mz_segment, dtype=float)

        for i in range(num_fitted_peaks):
            p_i = fit_params[i * n_p : (i + 1) * n_p]
            amp, center, fwhm = p_i[0], p_i[1], p_i[2]
            
            curve = peak_func(mz_segment, *p_i)
            total_fit += curve

            peak_info = {
                'name': str(initial_masses[i]) if i < len(initial_masses) else f"Peak_{i+1}",
                'peak_index': i + 1,
                'amplitude': amp,
                'center_mass': center,
                'fwhm_mass': fwhm,
                'params': p_i,
                'curve': curve
            }
            if is_pv:
                peak_info['mixing'] = p_i[3]

            fitted_peaks.append(peak_info)

        # --- STEP 8: REDESIGNED DIAGNOSTIC VISUALIZATION (3:1 DUAL PANEL) ---
        if plot_flag:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, 
                figsize=(10, 6), 
                sharex=True, 
                gridspec_kw={'height_ratios': [3, 1]}, 
                dpi=150
            )

            raw_seg, _, _ = get_nm_segment_data(nominal_mass, target_spectrum, target_mass_axis, self.tof_axis, nm_search_range=nm_search_range)
            iso_seg, _, _ = get_nm_segment_data(nominal_mass, total_isotope_signal, target_mass_axis, self.tof_axis, nm_search_range=nm_search_range)

            # --- TOP PANEL: Main Spectral Fit & Component Curves ---
            has_iso = subtract_isotopes and np.any(iso_seg > 0)
            if has_iso:
                ax1.plot(mz_segment, raw_seg, label='Raw Signal', color='gray', lw=1.2, alpha=0.5, linestyle=':')
                ax1.plot(mz_segment, int_segment, label='Target Signal (Deisotoped)', color='black', lw=1.8)
            else:
                ax1.plot(mz_segment, int_segment, label='Target Signal', color='black', lw=1.8)

            ax1.plot(mz_segment, total_fit, color="forestgreen", lw=2.0, linestyle="--", label="Total Fit")

            cmap = plt.get_cmap('tab10')
            for i, p_info in enumerate(fitted_peaks):
                col = cmap(i % 10)
                ax1.plot(mz_segment, p_info['curve'], color=col, lw=1.5, alpha=0.8, label=f"{p_info['name']} ({p_info['center_mass']:.4f} m/z)")
                ax1.axvline(p_info['center_mass'], color=col, linestyle=":", alpha=0.6)

            # Draw detection threshold gatekeeper lines
            ax1.axhline(noise_dict['detection_threshold'], label=f'Detection Threshold ({noise_std_mult}σ)', linestyle="-.", color='red', alpha=0.3)
            ax1.axhline(noise_dict['signal_power'], label='Signal Power (p95)', linestyle="-.", color='purple', alpha=0.3)

            ax1.set_title(f"Unconstrained Diagnostic Fit (m/z ~ {nominal_mass})", fontsize=11)
            ax1.set_ylabel("Intensity (ions/s)")
            ax1.grid(True, alpha=0.3, linestyle="--")
            ax1.legend(loc="upper right", fontsize="small")

            # --- BOTTOM PANEL: Fit Residuals & Subtracted Isotope Profile ---
            residuals = int_segment - total_fit
            ax2.plot(mz_segment, residuals, color='crimson', lw=1.2, label='Fit Residuals')
            ax2.axhline(0, color='black', lw=0.8, linestyle='--')

            if has_iso:
                ax2.plot(mz_segment, iso_seg, color='darkorange', lw=1.5, label='Subtracted Isotope Signal')
                ax2.fill_between(mz_segment, 0, iso_seg, color='darkorange', alpha=0.2)
                ax2.set_ylabel("Residuals / Isotopes")
            else:
                ax2.set_ylabel("Residual Error")

            ax2.set_xlabel("m/z")
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='upper right', fontsize='x-small')

            plt.tight_layout()

            if save_plot_flag:
                if output_dir is None:
                    output_dir = getattr(self, 'plot_dir', None) or get_default_plot_dir()
                ensure_dir(output_dir)
                if plot_filename is None:
                    plot_filename = f"unconstrained_fit_nm{nominal_mass}.png"
                plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')

            if show_plot_flag:
                plt.show()
            plt.close()

        # --- STEP 9: PACKAGE RESULTS & PERSIST TO DEPLOYMENT ---
        result = {
            'nominal_mass': nominal_mass,
            'mz_segment': mz_segment,
            'int_segment': int_segment,
            'total_fit': total_fit,
            'peaks': fitted_peaks,
            'fit_params': fit_params
        }

        self.diagnostic_fits[f"unconstrained_{nominal_mass}"] = result
        return result

    # TODO: Maybe make a function that populates all nominal masses with this/a similar method? 
    # -> This would take a while currently so maybe its best to dissuade users from this
    # but keep the method in case they want to make something custom
    def populate_nm_data_for(self, nominal_mass, force_non_subtracted=False, **kwargs) -> Deployment:
        """
        Extract intensity segment data for a specific nominal mass across all spectra.

        Slices spectral intensity channels corresponding to the target nominal mass window 
        across all spectrum writebufs. Extracts 1D/2D intensity segments, inverted negative 
        2D arrays, and per-spectrum calibrated mass and TOF axes into `self.nm_data[nominal_mass]`.

        Parameters
        ----------
        nominal_mass : int or float
            Target nominal mass-to-charge (m/z) integer or center value.
        force_non_subtracted : bool, default=False
            If ``True``, forces data extraction from raw `tofdata` instead of `tofdata_subtracted`.
        **kwargs : dict
            Additional keyword arguments passed to :func:`get_raw_nm_data`.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.nm_data[nominal_mass]` populated containing:
                * ``'intensity'`` : 2D matrix of sliced intensity segments across spectra.
                * ``'neg2d_intensity'`` : Inverted 2D intensity matrix.
                * ``'mass_axes'`` : 2D matrix of calibrated mass axes per spectrum.
                * ``'tof_axes'`` : 2D matrix of TOF axes per spectrum.

        Raises
        ------
        ValueError
            If :meth:`mass_calibration` has not been executed (`self.calibration is None`).
        """
        # Local import to prevent circular module dependencies
        from opentof.utils import get_raw_nm_data

        # Guard clause: Ensure mass calibration parameters are populated
        if getattr(self, 'calibration', None) is None:
            raise ValueError("Mass calibration not run. Call `calibrate_mass()` first.")
        if getattr(self, 'baseline', None) is None:
            print("[Warning!] Deployment.baseline is not yet calculated so baseline subtraction cannot occur! Ensure this is correct or uneeded.")
        
        # --- Determine Data Source ---
        # Fall back to raw un-subtracted tofdata if forced or if baseline-subtracted data is uncalculated
        if force_non_subtracted or getattr(self, 'tofdata_subtracted', None) is None:
            if getattr(self, 'tofdata_subtracted', None) is None and not force_non_subtracted:
                print("Deployment.tofdata_subtracted not calculated. Using Deployment.tofdata instead.")
            
            data_source = self.tofdata
        else:
            data_source = self.tofdata_subtracted

        # Extract raw nominal mass segment arrays via utility engine
        results = get_raw_nm_data(
            nm=nominal_mass,
            tofdata=data_source,
            tof_axis=self.tof_axis,
            calibration_results=self.calibration,
            **kwargs
        )

        # Structure extracted output segment matrices into a dictionary
        nm_data = {}
        nm_data['intensity'] = results[0]
        nm_data['neg2d_intensity'] = results[1]
        nm_data['mass_axes'] = results[2]
        nm_data['tof_axes'] = results[3]
        
        # Cache extracted dictionary under target nominal mass key
        self.nm_data[nominal_mass] = nm_data
        
        # Return self instance to enable method chaining
        return self

    def automated_peak_discovery(self, spectra=None, spectra_mass_axis=None, 
                                 baseline_subtracted=False, overwrite=False, 
                                 use_averaged_dataset=False, show_plot_flag=False,
                                 save_plot_flag=False, output_dir=None, plot_subdir=None,
                                 **kwargs) -> Deployment:
        """
        Discover peak positions in mass space using multi-overlap peak fitting.

        Identifies candidate peak centers across a single provided spectra or accross the GAD. 
        When processing `use_averaged_dataset=True`, clusters discovered peaks using 
        resolution-agnostic DBSCAN, flags high-variance peak clusters, and optionally updates 
        `self.peak_list` with consensus cluster centers.

        Parameters
        ----------
        spectra : numpy.ndarray or None, default=None
            1D or 2D intensity array. Defaults to `self.reference['reference_spectrum']` if ``None``.
        spectra_mass_axis : numpy.ndarray or None, default=None
            1D or 2D mass axis array matching `spectra`. Defaults to reference or first-guess mass axis.
        baseline_subtracted : bool, default=False
            If ``False``, subtracts `self.baseline['adjusted_baseline']` from input `spectra`.
        overwrite : bool, default=False
            If ``True``, populates `self.peak_list` with discovered consensus peaks and returns `self`.
        use_averaged_dataset : bool, default=False
            If ``True``, uses `self._averaged_dataset[0]` matrix as input spectra.
        show_plot_flag : bool, default=False
            If ``True``, displays peak discovery segment plots interactively.
        save_plot_flag : bool, default=False
            If ``True``, exports peak discovery segment plots to disk.
        output_dir : str or pathlib.Path or None, default=None
            Export directory path for saved plots. Defaults to `self.plot_dir` or OpenTof default if ``None``.
        plot_subdir : str or None, default=None
            Sub-directory folder name for plots. Defaults to `self.default_plot_subdir`.
        n_jobs : int, default=1
            Number of parallel execution worker processes.
        **kwargs : dict
            Additional keyword arguments passed to :func:`multi_overlap_peak_fit` or 
            :func:`cluster_discovered_peaks` (e.g., `min_spectra_fraction`).

        Returns
        -------
        peak_df or self : pandas.DataFrame or Deployment
            DataFrame of discovered peak candidates with DBSCAN labels and variance flags, 
            or `self` instance if `overwrite=True`.

        Raises
        ------
        ValueError
            If :meth:`determine_peak_width` has not been run or if required dataset inputs are missing.
        """
        # Local imports to prevent circular module dependencies
        from opentof.peak_fitting import multi_overlap_peak_fit
        from opentof.utils import (
            get_nm_segment_data, 
            cluster_discovered_peaks, 
            get_default_plot_dir
        )

        # Guard clause: Ensure peak width function is parameterized
        if getattr(self, 'peak_width_function', None) is None:
            raise ValueError("Peak width function not populated. Call `determine_peak_width()` first.")

        using_multiple_spectra = False

        # --- Output Directory Resolution ---
        if output_dir is None:
            output_dir = getattr(self, 'plot_dir', None) or get_default_plot_dir()
            
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir
            
        output_dir = os.path.join(output_dir, plot_subdir)
        if save_plot_flag and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # --- Data Source Resolution ---
        if use_averaged_dataset:
            # Use cached Global Averaged Dataset matrix if requested
            if getattr(self, '_averaged_dataset', None) is not None:
                print("Using Deployment._averaged_dataset for multi-spectra peak discovery.")
                spectra = self._averaged_dataset[0]
                using_multiple_spectra = True
            else:
                raise ValueError("`use_averaged_dataset=True` but _averaged_dataset not found.")
            used_custom_spectra_flag = False
        else:
            # Fall back to single reference spectrum if custom spectra is unspecified
            if spectra is None:
                print("'spectra' not specified. Falling back to reference spectrum.")
                if getattr(self, 'reference', None) is None:
                    raise ValueError("Deployment.reference not populated.")
                spectra = self.reference['reference_spectrum']
                if spectra_mass_axis is None:
                    spectra_mass_axis = self.reference['rs_mass_axis']
            used_custom_spectra_flag = True

        # Ensure spectra input is at least a 2D array
        spectra = np.atleast_2d(spectra)

        if not using_multiple_spectra and spectra_mass_axis is None:
            spectra_mass_axis = self.first_guess_mass_axis

        # Apply baseline subtraction if input spectrum is un-subtracted
        if not baseline_subtracted:
            if getattr(self, 'baseline', None) is None:
                raise ValueError("Deployment.baseline not populated.")
            spectra = spectra - self.baseline['adjusted_baseline']

        all_found_peaks = []
        master_plot_flag = show_plot_flag or save_plot_flag

        print(f"Running automated peak discovery on {spectra.shape[0]} spectra...")

        # --- Peak Discovery Processing Loop ---
        for idx, spec in enumerate(spectra):
            # Select mass axis corresponding to current spectrum
            if using_multiple_spectra:
                current_mass_axis = self.calibration['batch_massaxes'][idx]
            else:
                current_mass_axis = spectra_mass_axis
            
            # Identify unique nominal mass integer steps spanning spectrum
            nm_list = np.unique(np.round(current_mass_axis)).astype(int)
            
            # Fit overlapping peak models across each nominal mass segment
            for nm in nm_list:
                seg_i, seg_m, seg_t = get_nm_segment_data(
                    nm=nm, spectra=spec, mass_axis=current_mass_axis, tof_axis=self.tof_axis
                )

                min_sep = self.peak_width_function(nm)
                unique_filename = f"mopf_spec{idx}_nm{int(nm)}.png"

                fit_results = multi_overlap_peak_fit(
                    intensity_axis=seg_i, 
                    mass_axis=seg_m, 
                    tof_axis=seg_t,
                    min_separation_fwhm=min_sep,
                    plot_flag=master_plot_flag,
                    show_plot_flag=show_plot_flag,
                    save_plot_flag=save_plot_flag,
                    output_dir=output_dir,
                    plot_filename=unique_filename,
                    **kwargs
                )

                # Close figure canvases to prevent Matplotlib memory leaks across loop iterations
                if master_plot_flag and not show_plot_flag:
                    plt.close('all')

                spec_idx = -1 if used_custom_spectra_flag else idx

                # Collect discovered peak candidate metadata
                if fit_results and 'peaks' in fit_results:
                    for peak in fit_results['peaks']:
                        all_found_peaks.append({
                            'spectra_index': spec_idx,
                            'NM': nm,
                            'center_mass': peak.get('center_mass'),
                            'area': peak.get('area', np.nan),
                            'A': peak.get('A', peak.get('amplitude', np.nan)),
                            'fwhm_mass': peak.get('fwhm_mass', peak.get('fwhm', np.nan))
                        })

        peak_df = pd.DataFrame(all_found_peaks)

        if peak_df.empty:
            print("No peaks were discovered.")
            return peak_df

        # --- Clustering & High-Variance Analysis ---
        if using_multiple_spectra:
            print("\nClustering peaks across the averaged dataset...")
            # Perform resolution-agnostic z-space DBSCAN clustering
            consensus_peaks, peak_df = cluster_discovered_peaks(
                peak_df=peak_df,
                peak_width_func=self.peak_width_function,
                min_spectra_fraction=kwargs.get('min_spectra_fraction', 0.1)
            )
            
            # Identify high-variance clusters (mass span > 2 * FWHM)
            peak_df['is_high_variance'] = False
            clustered_mask = peak_df['dbscan_labels'] != -1

            if clustered_mask.any():
                stats = peak_df[clustered_mask].groupby(['NM', 'dbscan_labels'])['center_mass'].agg(['min', 'max']).reset_index()
                stats['mass_span'] = stats['max'] - stats['min']
                
                # Evaluate expected 2 * FWHM threshold per nominal mass group
                stats['threshold'] = stats['NM'].apply(lambda x: self.peak_width_function(x) * 2.0)
                stats['is_high_variance'] = stats['mass_span'] > stats['threshold']

                # Merge high-variance boolean flags back into master DataFrame
                peak_df = peak_df.drop(columns=['is_high_variance'], errors='ignore')
                peak_df = peak_df.merge(
                    stats[['NM', 'dbscan_labels', 'is_high_variance']],
                    on=['NM', 'dbscan_labels'],
                    how='left'
                )
                peak_df['is_high_variance'] = peak_df['is_high_variance'].fillna(False)

            # Update self.peak_list with consensus peak centers if requested
            if overwrite:
                print(f"\nOverwriting Deployment.peak_list with {len(consensus_peaks)} clustered peaks...")
                self.populate_peak_list_and_isotopes(consensus_peaks)
                return self

            self.apd_df = peak_df
            print("Resulting dataframe stored in Deployment.apd_df")

            return self
            
        else:
            # peak_df['is_high_variance'] = False
            if overwrite:
                found_masses_list = peak_df['center_mass'].dropna().tolist()
                self.populate_peak_list_and_isotopes(found_masses_list)
                return self


            self.apd_df = peak_df
            print("Resulting dataframe stored in Deployment.apd_df")

            return self

    # TODO: Revisit/Restructure old interactive GUI code...
    def launch_wizard(self, spectra=None, spectra_mass_axis=None, 
                      initial_mz_zoom=None):
        """
        Spins up a standalone GUI window to process data interactively.
        Blocks the notebook cell execution until the window is closed.
        """
        # Lazy import keeps interactive dependencies away from headless operations
        from opentof.interactive import SpectrumWizardGUI

        if spectra is None:
            if getattr(self, 'reference', None) is None:
                print("reference_spectra not calculated, using global average spectra")
                spectra = np.average(self.tofdata, axis=0)
                spectra = spectra.compute()
            else:
                print("Target spectra not provided! Using reference spectra.")
                spectra = self.reference['reference_spectrum']
                spectra_mass_axis = self.reference['rs_mass_axis']
        
        # Initialize the GUI and pass 'self' (this exact deployment instance) into it
        app = SpectrumWizardGUI(deployment_obj=self, spectra=spectra,
                                spectra_mass_axis=spectra_mass_axis,
                                initial_mz_zoom=initial_mz_zoom)
        
        # This loop blocks the Python process while the window is active
        app.mainloop() 
        
        # Once closed, execution resumes here and returns control to the notebook
        print("ℹ️ Wizard closed. Resuming programmatic script execution.")

    @staticmethod
    def _get_scalar_attr(group_or_h5, name):
        """
        Safely extract a scalar value from HDF5 object attributes.

        Handles both scalar values and 1-element NumPy arrays stored inside HDF5 metadata 
        attributes, returning Python native types or raw string/bytes objects.

        Parameters
        ----------
        group_or_h5 : h5py.Group or h5py.File
            Open HDF5 file handle or group object containing attributes.
        name : str
            Attribute name key string.

        Returns
        -------
        val : scalar, str, bytes, or None
            Extracted scalar value, or ``None`` if attribute `name` is missing.
        """
        # Return None if requested attribute key is missing
        if name not in group_or_h5.attrs:
            return None
            
        val = group_or_h5.attrs[name]
        
        # Use .item() to extract scalar value if object is a 1-element NumPy container
        if hasattr(val, 'item') and not isinstance(val, (str, bytes)):
            return val.item() if getattr(val, 'size', 1) == 1 else val
            
        return val

    def _load_default_formula_db(self):
        """
        Load the packaged default molecular formula parquet database file.

        Resolves the absolute path to `compound_list.parquet` relative to the package installation 
        directory, reads the table into a pandas DataFrame, and sorts entries by exact mass.

        Returns
        -------
        pandas.DataFrame
            DataFrame of compound formulas pre-sorted by `'ExactMass'` ascending.
        """
        # Resolve absolute directory path where deployment.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute file path to packaged compound database
        parquet_path = os.path.join(current_dir, "compound_list.parquet")
        
        # Read parquet database into pandas DataFrame
        df = pd.read_parquet(parquet_path)
        
        # Pre-sort database by exact mass for fast binary search candidate lookups
        return df.sort_values("ExactMass").reset_index(drop=True)


    def fix_external_calibration(self, true_masses, search_range=None, 
                                 mode=0, peak_type='gaussian', plot_flag=True,
                                 update_fgma=True, verbose=True):
        """
        Recalculate a corrupted or mis-aligned instrument mass calibration.

        Samples 100 random spectra across the deployment, localizes target anchor peak centers 
        in raw sample index space, fits unconstrained line shapes, regresses new mass calibration 
        polynomial parameters against `true_masses`, and optionally renders a two-panel validation figure.

        Parameters
        ----------
        true_masses : list of (float or None)
            Ordered list of expected exact m/z values corresponding to the most intense peaks 
            in sample index space. Use ``None`` to skip unassigned intermediate peaks.
        search_range : int or None, default=None
            Sample channel index window size used to isolate each anchor peak. Defaults to 
            1/1000th of total sample length if ``None``.
        mode : int, default=0
            Mass calibration mathematical equation mode identifier (e.g., `0`, `1`, `2`).
        peak_type : {'gaussian', 'pseudo_voigt', 'lorentzian', 'custom'}, default='gaussian'
            Peak model used to localize anchor peak centers in sample index space.
        plot_flag : bool, default=True
            If ``True``, renders a two-panel comparison plot displaying 'BEFORE' and 'AFTER' mass alignment.
        update_fgma : bool, default=True
            If ``True``, overwrites `self.first_guess_mass_axis` with the newly corrected mass axis.
        verbose : bool, default=True
            If ``True``, prints peak localization and regression progress logs to stdout.

        Returns
        -------
        new_params : list of float
            Solved mass calibration polynomial parameters.
        corrected_axis : numpy.ndarray
            1D array representing the corrected mass-to-charge (m/z) axis.

        Raises
        ------
        ValueError
            If fewer than 2 valid (non-``None``) anchor peak truths are matched.
        """
        from opentof.mass_calibration import calibrate_mass, apply_mass_calibration
        from opentof.peak_fitting import fit_unconstrained_peaks

        # --- Step 0: Out-of-Core Random Spectrum Sampling ---
        total_spectra = self.tofdata.shape[0]
        n_samples = min(100, total_spectra)
        
        # Select 100 sorted random spectrum indices for optimized chunk loading from disk
        random_indices = np.sort(np.random.choice(total_spectra, size=n_samples, replace=False))
        
        if verbose: 
            print(f"Sampling {n_samples} random spectra to isolate anchor peaks...")
        
        # Load and compute mean spectrum across sampled writebufs
        sampled_data = self.tofdata[random_indices, :].compute()
        sample_spectrum = np.mean(sampled_data, axis=0)
        si_axis = np.arange(len(sample_spectrum))

        # Cache previous un-corrected first guess mass axis for validation plotting
        old_mass_axis = self.first_guess_mass_axis.copy() if self.first_guess_mass_axis is not None else None

        # Default search range window to 1/1000th of spectrum channel length if unassigned
        if search_range is None:
            search_range = int(len(old_mass_axis) / 1000) if old_mass_axis is not None else 50

        # --- Steps 1-3: Iterative Peak Extraction in Sample Index Space ---
        working_spectrum = sample_spectrum.copy()
        discovered_sips = []
        target_count = len(true_masses)
        
        if verbose: 
            print(f"Searching for the top {target_count} peak profiles in index space...")

        # Extract anchor peaks sequentially by localizing maximum intensity channels and zeroing fitted regions
        for idx in range(target_count):
            max_idx = np.argmax(working_spectrum)
            
            idx_min = max(0, max_idx - search_range // 2)
            idx_max = min(len(si_axis), max_idx + search_range // 2)
            
            seg_si = si_axis[idx_min:idx_max]
            seg_intensity = sample_spectrum[idx_min:idx_max]
            
            seg_max = np.max(seg_intensity)
            if seg_max > 0:
                norm_seg = seg_intensity / seg_max
                xc_guess = seg_si[np.argmax(norm_seg)]
                fwhm_guess = max(np.sum(norm_seg > 0.5), 2.0)
                
                try:
                    # Fit unconstrained peak shape to extract precise peak center in sample index space
                    popt = fit_unconstrained_peaks(
                        x_axis=seg_si, signal=norm_seg,
                        centers_guess=[xc_guess], fwhms_guess=[fwhm_guess], amplitudes_guess=[1.0],
                        peak_type=peak_type, center_wiggle=5.0
                    )
                    fitted_sip = popt[1]
                    discovered_sips.append(fitted_sip)
                    if verbose: 
                        print(f"  -> Peak {idx+1} localized at Sample Index: {fitted_sip:.2f} (Raw Intensity: {seg_max:.2e})")
                except Exception:
                    # Fall back to raw argmax channel if curve fitting fails
                    discovered_sips.append(float(max_idx))
                    if verbose: 
                        print(f"  -> Peak {idx+1} fallback to raw argmax at Sample Index: {max_idx}")
            else:
                discovered_sips.append(float(max_idx))
                
            # Zero out localized segment region to prevent re-extracting the same peak in next iteration
            working_spectrum[idx_min:idx_max] = 0.0

        # --- Steps 4-5: Match Sample Indices to Truths & Solve New Parameters ---
        final_sips = []
        final_truths = []
        
        # Filter out None entries (ignored unknown peaks)
        for sip, truth in zip(discovered_sips, true_masses):
            if truth is None:
                if verbose: 
                    print(f"Skipping unassigned/ignored peak discovered at sample index {sip:.2f}")
                continue
            final_sips.append(sip)
            final_truths.append(truth)
            
        # Require at least 2 anchor peaks to solve mass calibration equations
        if len(final_sips) < 2:
            raise ValueError("You must match at least 2 peaks to resolve calibration function coefficients.")

        if verbose:
            print(f"\nMatching sample indices to user expectations:")
            for sip, truth in zip(final_sips, final_truths):
                print(f"  Index {sip:7.2f}  ===>  True m/z {truth:.4f}")

            print(f"\nRegressing new calibration parameters for Mode {mode}...")
            
        # Solve non-linear mass calibration parameters
        new_params = calibrate_mass(np.array(final_sips), np.array(final_truths), mode=mode)
        
        if verbose: 
            print(f"Success! Solved Parameters: {[round(p, 6) for p in new_params]}")

        # Compute corrected mass-to-charge axis
        corrected_axis = apply_mass_calibration(si_axis, new_params, mode=mode)

        # --- Step 6: Render Validation Comparison Plot ---
        if plot_flag and old_mass_axis is not None:
            fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True, dpi=150)
            
            # Subplot 1: Previous broken mass alignment
            ax_before.plot(old_mass_axis, sample_spectrum, color='crimson', lw=1.2, label='Previous External Mass Axis')
            ax_before.set_title("BEFORE: Mis-aligned external calibration", fontsize=11,)
            ax_before.set_ylabel("Intensity (ions/s)")
            ax_before.grid(True, linestyle='--', alpha=0.5)
            ax_before.legend(loc='upper left')
            
            # Subplot 2: Corrected mass alignment
            ax_after.plot(corrected_axis, sample_spectrum, color='forestgreen', lw=1.2, label='Corrected Mass Axis')
            ax_after.set_title("AFTER: Mass Axis after 'manual' recalibration", fontsize=11,)
            ax_after.set_xlabel("Mass-to-charge (m/z)")
            ax_after.set_ylabel("Intensity (ions/s)")
            ax_after.grid(True, linestyle='--', alpha=0.5)
            ax_after.legend(loc='upper left')

            # Set plot axis display boundaries across both old and new axes
            global_xmin = min(np.min(old_mass_axis), np.min(corrected_axis)) - 10
            global_xmax = max(np.max(old_mass_axis), np.max(corrected_axis)) + 10
            ax_after.set_xlim(global_xmin, global_xmax)

            # Draw target truth mass vertical indicator lines across both subplots
            for t_mass in final_truths:
                ax_before.axvline(t_mass, color='blue', linestyle=':', alpha=0.6, lw=1.5)
                ax_before.text(t_mass, np.max(sample_spectrum) * 0.75, f' Target {t_mass:.2f}', color='blue', rotation=90, fontsize=8)
                
                ax_after.axvline(t_mass, color='blue', linestyle='--', alpha=0.8, lw=1.5)
                ax_after.text(t_mass, np.max(sample_spectrum) * 0.75, f' Aligned {t_mass:.2f}', color='blue', rotation=90, fontsize=8)

            plt.tight_layout()
            plt.show()

        # Update Deployment object first_guess_mass_axis if requested
        if update_fgma:
            if verbose: 
                print("Updated Deployment.first_guess_mass_axis!")
            self.first_guess_mass_axis = corrected_axis
        
        return new_params, corrected_axis

    def auto_fix_external_calibration(self, base_truths, target_max, mz_tol=1.0, max_nones=5, plot_flag=True):
        """
        Automatically repair a failed external mass calibration through iterative truth searching.

        Evaluates truth mass candidates (`base_truths`), testing both original and reversed 
        peak index orderings while dynamically injecting up to `max_nones` intermediate ``None`` 
        placeholders for unassigned unknown peaks until the resulting upper axis bound matches 
        `target_max` within `mz_tol`.

        Parameters
        ----------
        base_truths : sequence of (float or str)
            Core list of expected exact masses or formula strings (e.g., ``["I-", "IH2O-"]``) 
            known to represent the most intense peaks in the spectrum.
        target_max : float
            Expected maximum mass-to-charge (m/z) value for the calibrated mass axis.
        mz_tol : float, default=1.0
            Acceptable tolerance window (+-m/z) around `target_max`.
        max_nones : int, default=5
            Maximum number of unassigned unknown peak placeholders (``None``) to inject during iterations.
        plot_flag : bool, default=True
            If ``True``, renders the final validation plot upon successful recalibration.

        Returns
        -------
        bool
            ``True`` if a valid calibration configuration matching `target_max` was successfully 
            resolved; ``False`` otherwise.

        Raises
        ------
        ValueError
            If `self.first_guess_mass_axis` is ``None``.
        """
        # Local import to prevent circular dependencies
        from opentof.utils import return_mass

        # Translate chemical formula strings into float theoretical masses
        processed_truths = []
        for item in base_truths:
            if isinstance(item, str):
                processed_truths.append(return_mass(item))
            else:
                processed_truths.append(item)
        
        base_truths = processed_truths

        min_val = target_max - mz_tol
        max_val = target_max + mz_tol

        # Check initial external calibration state
        if self.first_guess_mass_axis is None:
            raise ValueError("first_guess_mass_axis is None. Cannot evaluate external calibration.")
            
        external_cal_max = self.first_guess_mass_axis.max()
        print(f"Initial external cal max: {external_cal_max:.2f}")

        # Exit early if existing axis upper bound already falls within target tolerance window
        if min_val <= external_cal_max <= max_val:
            print("Calibration is already within the expected range. No action needed.")
            return True

        print("Bad external calibration detected, attempting to correct...")

        # Outer Loop: Incrementally inject up to max_nones unknown peak placeholders
        for i in range(max_nones + 1):
            
            # Inner Loop: Test original truth ordering, followed by reversed truth ordering
            for is_reversed in (False, True):
                
                working_truths = base_truths[::-1] if is_reversed else base_truths
                
                # Construct truth list injecting 'i' None placeholders between first and remaining truths
                current_truths = [working_truths[0]] + [None] * i + working_truths[1:]
                
                order_str = "REVERSED" if is_reversed else "ORIGINAL"
                if i == 0:
                    print(f"\n--- Trying {order_str} base_truths with no injected unknown peak(s) ---")
                else:
                    print(f"\n--- Trying {order_str} base_truths with {i} injected unknown peak(s) in between truth peaks ---")
                
                try:
                    # Attempt trial calibration run suppressing warnings and verbose console output
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        new_params, test_axis = self.fix_external_calibration(
                            true_masses=current_truths, 
                            plot_flag=False, 
                            update_fgma=False,
                            verbose=False
                        )
                except ValueError as e:
                    print(f"  Calibration failed on this attempt: {e}")
                    continue
                    
                # Evaluate upper mass limit of tested trial axis
                external_cal_max = test_axis.max()
                print(f"  New external cal max evaluated at: {external_cal_max:.2f}")
                
                # Check if trial axis upper bound meets target acceptance criteria
                if min_val <= external_cal_max <= max_val:
                    print(f"\nSuccess! Calibration reached expected max range ({min_val:.1f} - {max_val:.1f}).")
                    
                    # Re-run final calibration pass to generate validation plot and update object instance
                    if plot_flag:
                        print("Re-running final configuration to generate validation plot & update object...")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.fix_external_calibration(
                                true_masses=current_truths, 
                                plot_flag=True, 
                                update_fgma=True,
                                verbose=True
                            )
                    else:
                        print("Updated Deployment.first_guess_mass_axis!")
                        self.first_guess_mass_axis = test_axis
                        
                    return True

        print(f"\nFailed to find a valid calibration after testing both original and reversed arrays, injecting up to {max_nones} None values.")
        return False

    # TODO revisit SOM helpers and wrapper functions...
    def nominal_mass_som(self, nm, 
                         som_size=None,
                         output_dir=None, 
                         plot_subdir=None, 
                         **kwargs):
        from opentof.som_helpers import (
            RowMinMaxScaler, build_nm_som, organize_som_data,
            process_som_weight_peaks
        )

        if output_dir is None:
            if getattr(self, 'plot_dir', None) is not None:
                output_dir = self.plot_dir
            else:
                from opentof.utils import get_default_plot_dir
                output_dir = get_default_plot_dir()
            
        if plot_subdir is None:
            plot_subdir = self.default_plot_subdir

        self.populate_nm_data_for(nm)

        nm_dict = self.nm_data[nm]
        
        # Define the variables we want to include
        variable_names = ['intensity', 'neg2d_intensity', 'mass_axes', 'tof_axes']
        
        scaled_vars = []
        variable_scalers = []
        
        # Process each variable: Scale it and store the scalers
        for var in variable_names:
            data = nm_dict[var]
            scaler = RowMinMaxScaler()
            
            scaled_data, scalers_arr = scaler.fit_transform(data)
            
            scaled_vars.append(scaled_data)
            variable_scalers.append(scalers_arr)
        
        # Create the "Vectorized" input_samples
        # Concatenate along axis 1 (columns)
        input_samples = np.concatenate(scaled_vars, axis=1)
        
        # Determine the length of a single variable
        variable_length = nm_dict['intensity'].shape[1]
        
        print(f"Number of samples: {len(input_samples)}")
        print(f"Vectorized feature length: {input_samples.shape[1]}")

        if som_size is None:
            print("'som_size' not provided, using some default size: (5,5)")
            som_size = (5,5)

        # Build/Train the SOM
        som = build_nm_som(nm, input_samples, som_size=som_size)
        
        # Organize the data
        nm_node_data, som_err_metrics = organize_som_data(
            som=som,
            input_samples=input_samples,
            variable_names=variable_names,
            variable_scalers=variable_scalers,
            variable_length=variable_length,
            topology='hexagonal' # or 'rectangular'
        )

        if getattr(self, 'custom_peak_shape', None) is not None:
            process_som_weight_peaks(nm_node_data, nm, self.peak_width_function, 
                                    peak_type='custom', custom_shape=self.custom_peak_shape,
                                    output_dir=output_dir, plot_subdir=plot_subdir)
        else:
            process_som_weight_peaks(nm_node_data, nm, self.peak_width_function,
                                     output_dir=output_dir, plot_subdir=plot_subdir)
        
        return nm_node_data, som_err_metrics, som


    def export_to_h5(self, output_dir=None, driver="PIF") -> Deployment:
        """
        Export workspace state, mass calibrations, baselines, and peak shapes to HDF5 files.

        Supports exporting to native OpenTof workspace state files (``driver='OT'``) or 
        Tofware-compatible file schemas: Instrument Function (``driver='IF'``), Processed 
        results (``driver='P'``), or both simultaneously (``driver='PIF'``).

        Parameters
        ----------
        output_dir : str or pathlib.Path or None, default=None
            Target directory path where exported `.h5` files will be saved. Defaults to 
            `self.plot_dir` or default output directory if ``None``.
        driver : {'PIF', 'OT', 'IF', 'P'}, default='PIF'
            File schema export driver:
            
            * ``'OT'``: Native OpenTof full workspace state serialization.
            * ``'IF'``: Tofware Instrument Function schema (mass calibration, peak width, peak shape).
            * ``'P'``: Tofware Processed schema (baseline, fitted peak data).
            * ``'PIF'``: Combined export driving both ``'IF'`` and ``'P'`` schema generators.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance after completing file serialization operations.

        Raises
        ------
        ValueError
            If `driver` is not one of ``'OT'``, ``'IF'``, ``'P'``, or ``'PIF'``.
        """
        # Validate driver selection and convert to uppercase
        driver = driver.upper()
        if driver not in ("OT", "IF", "P", "PIF"):
            raise ValueError("File export driver must be one of: 'OT', 'IF', 'P', 'PIF'")

        # Route export request to corresponding private serializer backend
        if driver == "OT":
            return self._export_ot(output_dir)
        elif driver == "P":
            return self._export_p(output_dir)
        elif driver == "IF":
            return self._export_if(output_dir)
        elif driver == "PIF":
            # Execute both Instrument Function (IF) and Processed (P) export routines
            self._export_if(output_dir)
            self._export_p(output_dir)
            return self

    def import_from_h5(self, filepath, driver="PIF") -> Deployment:
        """
        Import configurations, mass calibrations, and peak shapes from HDF5 state files.

        Reads calibration parameters, baseline fits, and metadata from a target file or 
        directory containing Tofware sub-folders (``IF/``, ``Processed/``) or native OpenTof 
        workspace files.

        Parameters
        ----------
        filepath : str or pathlib.Path
            File path or directory path containing target `.h5` import files.
        driver : {'PIF', 'OT', 'IF', 'P'}, default='PIF'
            File schema driver matching the target import file structure:
            
            * ``'OT'``: Native OpenTof full workspace state deserializer.
            * ``'IF'``: Tofware Instrument Function deserializer.
            * ``'P'``: Tofware Processed results deserializer.
            * ``'PIF'``: Combined import executing both ``'IF'`` and ``'P'`` schema readers.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance populated with imported attributes and metadata.

        Raises
        ------
        ValueError
            If `driver` is not one of ``'OT'``, ``'IF'``, ``'P'``, or ``'PIF'``.
        FileNotFoundError
            If `filepath` does not exist on disk.
        """
        # Validate driver selection and convert to uppercase
        driver = driver.upper()
        if driver not in ("OT", "IF", "P", "PIF"):
            raise ValueError("File import driver must be one of: 'OT', 'IF', 'P', 'PIF'")

        # Verify target file or directory exists on disk
        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"The explicit deployment file target '{path_obj}' does not exist.")

        # Route import request to corresponding private deserializer backend
        if driver == "OT":
            return self._import_ot(path_obj)
        elif driver == "P":
            return self._import_p(path_obj)
        elif driver == "IF":
            return self._import_if(path_obj)
        elif driver == "PIF":
            # Execute both Instrument Function (IF) and Processed (P) import routines
            self._import_if(path_obj)
            self._import_p(path_obj)
            return self


    # =========================================================================
    # EXPORT HELPERS
    # =========================================================================
    def _export_ot(self, output_dir):
        """
        Private helper to serialize complete Deployment workspace state into a native OpenTof HDF5 file.

        Saves metadata attributes, primary axis arrays, time-series calibration caches, empirical 
        peak shapes, dictionary structures (`calibration`, `reference`, `baseline`, etc.), wide-format 
        DataFrames (`peak_data`, `time_series`), and writes the 2D out-of-core Dask spectral dataset 
        to the `/tofdata` HDF5 group.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None
            Explicit directory or file path destination for the exported `.h5` file.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance after completing serialization.

        Raises
        ------
        ValueError
            If `output_dir` is ``None``.
        """
        # Guard clause: Enforce requirement for an explicit destination path
        if output_dir is None:
            raise ValueError("An explicit destination path or folder must be provided for 'OT' mode.")
        
        # Resolve target output file path (defaulting to 'opentof_deployment.h5' if directory path is passed)
        path_obj = Path(output_dir)
        if path_obj.is_dir() or path_obj.suffix != ".h5":
            out_filepath = path_obj / "opentof_deployment.h5"
        else:
            out_filepath = path_obj
            
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving OpenTof deployment state to: {out_filepath}")

        # --- Step 1: Serialize Metadata Attributes & Base Arrays ---
        with h5py.File(out_filepath, "w") as f:
            f.attrs["OpenTofFileType"] = "DeploymentState"
            f.attrs["_user_entered_cycling_flag"] = bool(self._user_entered_cycling_flag)
            f.attrs["chunk_size"] = int(self.chunk_size)
            
            if self.target_nbr_samples is not None:
                f.attrs["target_nbr_samples"] = int(self.target_nbr_samples)
            if self.instrument_type is not None:
                f.attrs["instrument_type"] = str(self.instrument_type)

            if getattr(self, '_averaged_dataset_interval', None) is not None:
                f.attrs["averaged_dataset_interval"] = self._averaged_dataset_interval

            # Serialize 1D primary axes
            if self.tof_axis is not None:
                f.create_dataset("tof_axis", data=np.asarray(self.tof_axis, dtype="float64"))
            if self.first_guess_mass_axis is not None:
                f.create_dataset("first_guess_mass_axis", data=np.asarray(self.first_guess_mass_axis, dtype="float64"))
            if self.peak_width_coeffs is not None:
                f.create_dataset("peak_width_coeffs", data=np.asarray(self.peak_width_coeffs, dtype="float64"))

            # Serialize global time-sorting masks and cached timestamps
            if getattr(self, '_global_sort_idx', None) is not None:
                f.create_dataset("global_sort_idx", data=np.asarray(self._global_sort_idx, dtype="int64"))
            if getattr(self, '_timestamps_cache', None) is not None:
                f.create_dataset("timestamps_cache", data=self._timestamps_cache.astype("int64"))

            if self.reagent_ion is not None:
                f.attrs["reagent_ion"] = str(self.reagent_ion)
            if self.plot_dir is not None:
                f.attrs["plot_dir"] = str(self.plot_dir)
            if self.segment_profiles is not None:
                f.attrs["segment_profiles"] = json.dumps(self.segment_profiles)

            # Serialize raw peak width floor coefficients and corruption tracking flags
            if getattr(self, 'raw_peak_width_coeffs', None) is not None:
                f.create_dataset("raw_peak_width_coeffs", data=np.asarray(self.raw_peak_width_coeffs, dtype="float64"))
            if getattr(self, '_corrupt_flags', None) is not None:
                f.create_dataset("_corrupt_flags", data=np.asarray(self._corrupt_flags, dtype=bool))

            # Serialize empirical peak shape grid and area ratio
            if getattr(self, 'peak_shape_interp', None) is not None:
                ps_grp = f.create_group("peak_shape_data")
                ps_grp.create_dataset("x", data=self.peak_shape_interp.x)
                ps_grp.create_dataset("y", data=self.peak_shape_interp.y)
                if getattr(self, 'custom_shape_area_ratio', None) is not None:
                    ps_grp.attrs["custom_shape_area_ratio"] = float(self.custom_shape_area_ratio)

            # --- Step 2: Recursive Dictionary Group Serializer ---
            def save_dict_to_group(hdf5_group, dict_obj):
                for k, v in dict_obj.items():
                    key_str = str(k)
                    if v is None:
                        continue
                    elif isinstance(v, dict):
                        sub_grp = hdf5_group.create_group(key_str)
                        save_dict_to_group(sub_grp, v)
                    elif isinstance(v, (np.ndarray, list, tuple)):
                        arr = np.asarray(v)
                        # Handle datetime arrays by converting to int64 nanoseconds
                        if arr.dtype.kind == 'M' or isinstance(v, pd.DatetimeIndex):
                            hdf5_group.create_dataset(key_str, data=pd.to_datetime(arr).astype("int64"))
                            hdf5_group[key_str].attrs["is_datetime"] = True
                        elif arr.dtype.kind in ('U', 'S', 'O'):
                            str_dt = arr.astype(str).tolist()
                            hdf5_group.create_dataset(key_str, data=str_dt)
                        else:
                            hdf5_group.create_dataset(key_str, data=arr)
                    elif isinstance(v, (int, float, bool, str)):
                        hdf5_group.attrs[key_str] = v
                    elif isinstance(v, (np.datetime64, datetime)):
                        hdf5_group.attrs[key_str] = str(v)
                    else:
                        try:
                            hdf5_group.attrs[key_str] = str(v)
                        except Exception:
                            pass

            # Map primary object dictionary attributes to output HDF5 groups
            dict_mappings = {
                "calibration": "calibration",
                "reference": "reference",
                "baseline": "baseline",
                "peak_list": "peak_list",
                "nm_data": "nm_data",
                "_cycling_status": "cycling_status",
                "isotopes": "isotopes"
            }

            for attr_name, h5_path in dict_mappings.items():
                val = getattr(self, attr_name, None)
                if val is not None and isinstance(val, dict):
                    grp = f.create_group(h5_path)
                    save_dict_to_group(grp, val)

            # Serialize cached Global Averaged Dataset (GAD) tuple
            if getattr(self, '_averaged_dataset', None) is not None:
                avg_tuple = self._averaged_dataset
                if len(avg_tuple) == 3:
                    avg_grp = f.create_group("averaged_dataset")
                    avg_grp.create_dataset("averaged_data", data=np.asarray(avg_tuple[0], dtype="float32"))
                    avg_grp.create_dataset("averaged_timestamps", data=pd.to_datetime(avg_tuple[1]).astype("int64"))
                    avg_grp.create_dataset("interval_indices", data=np.asarray(avg_tuple[2], dtype="int32"))

            # --- Step 3: Serialize Wide-Format DataFrames ---
            for df_attr in ["peak_data", "time_series"]:
                df = getattr(self, df_attr, None)
                if df is not None and isinstance(df, pd.DataFrame):
                    grp = f.create_group(df_attr)
                    grp.create_dataset("_index", data=df.index.to_numpy())
                    for col in df.columns:
                        col_data = df[col].to_numpy()
                        if col_data.dtype == object:
                            col_data = col_data.astype(str)
                        grp.create_dataset(str(col), data=col_data)

        # --- Step 4: Write Out-of-Core Dask TOF Dataset ---
        val = getattr(self, "_tofdata", None)
        if val is not None and isinstance(val, da.Array):
            print("Saving Deployment.tofdata to /tofdata...")
            target_chunks = (self.chunk_size, *val.chunksize[1:])
            da.to_hdf5(str(out_filepath), "/tofdata", val, chunks=target_chunks)

        print("OpenTof Deployment object saved to .h5 file successfully.")
        return self

    def _export_p(self, output_dir):
        """
        Private helper to export batch Tofware Processed (_p.h5) files.

        Iterates through active file handles, constructs Tofware-compliant HDF5 groups 
        (`FullSpectra`, `PeakData`, `TimingData`), populates `PeakTable` structured arrays, 
        maps fitted peak amplitudes from `self.peak_data` into 4D matrix arrays 
        (`writes`, `bufs`, `segments`, `peaks`), and saves processed files to a `Processed/` subfolder.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None
            Target export directory path. Defaults to `src_path.parent / "Processed"` if ``None``.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance after completing file export batch operations.
        """
        # Guard clause: Ensure active file handles exist
        if not self._file_handles:
            print("❌ Export canceled: No active HDF5 file handles found in this Deployment.")
            return self

        print(f"Starting batch export of {len(self._file_handles)} Processed (_P) files...")

        # --- Step 1: Construct PeakTable Compound Data Type ---
        peak_names = self.peak_list['peaks'] if self.peak_list else []
        nbr_peaks = len(peak_names)
        
        dt_peak = np.dtype([
            ('label', 'S64'),
            ('mass', 'f4'),
            ('lower integration limit', 'f4'),
            ('upper integration limit', 'f4')
        ])
        peak_table_data = np.zeros(nbr_peaks, dtype=dt_peak)
        
        # Populate structured PeakTable array entries
        if nbr_peaks > 0:
            for i, peak_name in enumerate(peak_names):
                mass = self.peak_list['centers'][i]
                fwhm = self.peak_list['fwhms'][i] if 'fwhms' in self.peak_list else 0.5
                peak_table_data[i]['label'] = str(peak_name).encode('utf-8')
                peak_table_data[i]['mass'] = mass
                peak_table_data[i]['lower integration limit'] = mass - (fwhm * 1.5)
                peak_table_data[i]['upper integration limit'] = mass + (fwhm * 1.5)

        # --- Step 2: Iterate and Export Per-File Processed HDF5 Output Files ---
        for k, h5_src in sorted(self._file_handles.items()):
            src_path = Path(h5_src.filename)
            
            # Resolve target export folder path
            if output_dir is None:
                target_dir = src_path.parent / "Processed"
            else:
                target_dir = Path(output_dir) / "Processed"
                
            target_dir.mkdir(parents=True, exist_ok=True)
            out_filepath = target_dir / f"{src_path.stem}_p.h5"

            with h5py.File(out_filepath, "w") as h5_out:
                # Set root Tofware schema file attributes
                h5_out.attrs["TofwareFileType"] = np.float32(3.0)
                h5_out.attrs["TofwareDataFileRef"] = src_path.name.encode('utf-8')
                h5_out.attrs["NbrPeaks"] = np.int32(nbr_peaks)

                # Copy standard root metadata attributes from raw source HDF5 file
                root_copy_attrs = [
                    "FileSignature", "DAQ Hardware", "DAQ Serial",
                    "HDF5 File Creation Time", "HDF5 Library Version",
                    "TofDAQ Build Date", "IonMode", "NbrBlocks",
                    "NbrBufs", "NbrCubes", "NbrMemories",
                    "NbrRawSamples", "NbrRuns", "NbrSamples",
                    "NbrSegments", "NbrWaveforms", "NbrWrites", "CurrentRun"
                ]
                for attr in root_copy_attrs:
                    if attr in h5_src.attrs:
                        h5_out.attrs[attr] = h5_src.attrs[attr]

                # Determine true write/buf matrix dimensions from raw dataset shapes
                if "TimingData" in h5_src and "BufTimes" in h5_src["TimingData"]:
                    actual_writes, actual_bufs = h5_src["TimingData"]["BufTimes"].shape
                elif "FullSpectra" in h5_src and "TofData" in h5_src["FullSpectra"]:
                    actual_writes, actual_bufs = h5_src["FullSpectra"]["TofData"].shape[:2]
                else:
                    writes_val = self._get_scalar_attr(h5_src, "NbrWrites")
                    actual_writes = int(writes_val) if writes_val is not None else 1
                    bufs_val = self._get_scalar_attr(h5_src, "NbrBufs")
                    actual_bufs = int(bufs_val) if bufs_val is not None else 1

                # Overwrite root attributes with validated physical dimensions
                h5_out.attrs["NbrWrites"] = np.int32(actual_writes)
                h5_out.attrs["NbrBufs"] = np.int32(actual_bufs)

                segments_val = self._get_scalar_attr(h5_src, "NbrSegments")
                nbr_segments = int(segments_val) if segments_val is not None else 1

                samples_val = self._get_scalar_attr(h5_src, "NbrSamples")
                nbr_samples = int(samples_val) if samples_val is not None else (self.target_nbr_samples if self.target_nbr_samples else 100000)

                # Construct FullSpectra HDF5 group
                fs_grp = h5_out.create_group("FullSpectra")
                fs_attrs = [
                    "MassCalibMode", "MassCalibration a", "MassCalibration b",
                    "MassCalibration nbrParameters", "MassCalibration nbrPoints",
                    "MassCalibration R0", "MassCalibration dm", "MassCalibration m0",
                    "SampleInterval", "ClockPeriod", "Single Ion Signal"
                ]
                if "FullSpectra" in h5_src:
                    for attr in fs_attrs:
                        if attr in h5_src["FullSpectra"].attrs:
                            fs_grp.attrs[attr] = h5_src["FullSpectra"].attrs[attr]

                # Write MassAxis, SumSpectrum, and placeholder TofData datasets
                mass_axis = self.first_guess_mass_axis if self.first_guess_mass_axis is not None else np.linspace(1.0, 400.0, nbr_samples)
                dset_ma = fs_grp.create_dataset("MassAxis", data=mass_axis.astype(np.float32))
                dset_ma.attrs["IGORWaveType"] = np.int32(2)

                dset_ss = fs_grp.create_dataset("SumSpectrum", data=np.zeros(len(mass_axis), dtype=np.float64))
                dset_ss.attrs["IGORWaveType"] = np.int32(4)

                dset_td = fs_grp.create_dataset("TofData", data=np.zeros((1, 1, 1, 1), dtype=np.float32))
                dset_td.attrs["IGORWaveType"] = np.int32(2)

                # Construct PeakData HDF5 group and populate attributes
                pd_grp = h5_out.create_group("PeakData")
                pd_grp.attrs["hS_DateTimeInitialized"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')
                pd_grp.attrs["hS_DateTimeFinished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')
                pd_grp.attrs["hS_bsl_flag"] = np.int32(1 if self.baseline else 0)
                pd_grp.attrs["hS_isot_flag"] = np.int32(0)
                pd_grp.attrs["hS_negPeaks_flag"] = np.int32(0)
                pd_grp.attrs["hS_LLS_flag"] = np.int32(1)
                pd_grp.attrs["hS_jitter"] = np.int32(0)
                pd_grp.attrs["hS_freePW_flag"] = np.int32(0)
                pd_grp.attrs["hS_segAv_flag"] = np.int32(0)
                pd_grp.attrs["hS_bufAv_flag"] = np.int32(0)
                pd_grp.attrs["hS_loSeg"] = np.int32(1)
                pd_grp.attrs["hS_hiSeg"] = np.int32(nbr_segments)
                pd_grp.attrs["hS_TOFduty_flag"] = np.int32(0)
                pd_grp.attrs["hS_n_peakTable"] = np.int32(nbr_peaks)
                pd_grp.attrs["hS_kVersion"] = np.float32(1.0)
                pd_grp.attrs["hS_kSubVersion"] = np.float32(0.0)
                pd_grp.attrs["hS_kBeta"] = np.float32(0.0)
                pd_grp.attrs["hS_instrumentIndex"] = np.float32(0.0)

                pd_grp.create_dataset("PeakTable", data=peak_table_data)

                # Pre-allocate 4D PeakData intensity matrix (writes, bufs, segments, peaks)
                peak_data_matrix = np.zeros((actual_writes, actual_bufs, nbr_segments, nbr_peaks), dtype=np.float32)

                # Populate 4D PeakData matrix from self.peak_data DataFrame fitting results
                if getattr(self, 'peak_data', None) is not None and nbr_peaks > 0:
                    file_sort_indices = self._sort_indices.get(k, np.arange(actual_writes * actual_bufs))
                    
                    start_idx = sum(len(self._sort_indices[idx]) for idx in range(k))
                    end_idx = start_idx + len(file_sort_indices)
                    
                    file_df = self.peak_data.iloc[start_idx:end_idx]

                    for p_idx, p_name in enumerate(peak_names):
                        col_name = f"{p_name}_area"
                        if col_name in file_df.columns:
                            intensities = file_df[col_name].values
                            flat_array = np.zeros(actual_writes * actual_bufs, dtype=np.float32)
                            flat_array[file_sort_indices] = intensities
                            peak_data_matrix[:, :, 0, p_idx] = flat_array.reshape((actual_writes, actual_bufs))
                
                pd_grp.create_dataset("PeakData", data=peak_data_matrix)

                # Construct TimingData HDF5 group
                td_grp = h5_out.create_group("TimingData")
                if "TimingData" in h5_src:
                    for attr, val in h5_src["TimingData"].attrs.items():
                        td_grp.attrs[attr] = val
                    if "BufTimes" in h5_src["TimingData"]:
                        dset_bt = td_grp.create_dataset("BufTimes", data=h5_src["TimingData"]["BufTimes"][:])
                        dset_bt.attrs["IGORWaveType"] = np.int32(4)

            print(f" -> Exported 'Processed' file: {out_filepath.name}")

        print("Global Processed batch export finished successfully.")
        return self

    def _export_if(self, output_dir):
        """
        Private helper to export batch Tofware Instrument Function (_IF.h5) files.

        Slices global time-series calibration parameters (`batch_params`) per file timestamp window, 
        evaluates empirical peak shape matrices, builds Tofware-compliant attribute keys 
        (e.g., `l1`, `m1`, `t1`), and populates `Peaks`, `FullSpectra`, and `MassCalib` groups 
        inside exported `_IF.h5` files saved to an `IF/` subfolder.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None
            Target export directory path. Defaults to `src_path.parent / "IF"` if ``None``.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance after completing file export batch operations.
        """
        # Guard clause: Ensure active file handles exist
        if not self._file_handles:
            print("❌ Export canceled: No active HDF5 file handles found in this Deployment.")
            return self

        print(f"Starting batch export of {len(self._file_handles)} Instrument Function (_IF) files...")

        # --- Step 1: Extract Calibration Metadata & Peak Shape Grid ---
        cal_res = self.calibration if self.calibration is not None else {}
        mode_val = cal_res.get("mass_cal_mode", 2)
        eq_strings = {0: "p1*sqrt(m)+p2", 2: "p1*m^p3+p2"}
        fit_type = "CustomShape" if self.custom_peak_shape is not None else "Gaussian"
        
        global_times_raw = cal_res.get("averaged_timestamps", np.array([]))
        if len(global_times_raw) > 0:
            global_times = pd.to_datetime(global_times_raw).to_numpy().astype('datetime64[ns]')
        else:
            global_times = np.array([], dtype='datetime64[ns]')

        global_pars = np.array(cal_res.get("batch_params", []))
        global_flags = np.array(cal_res.get("flags", [])) if "flags" in cal_res else None

        # Build 2D peak shape domain grid across -20 to +20 sigma
        x_ps = np.linspace(-20, 20, 801)
        y_ps = np.zeros_like(x_ps)
        
        if self.peak_shape_interp is not None:
            y_ps = self.peak_shape_interp(x_ps)
        else:
            y_ps = np.exp(-0.5 * (x_ps ** 2))
        peak_shape_matrix = np.column_stack([x_ps, y_ps])

        # Extract unique calibrant m/z values
        global_cals_raw = np.atleast_1d(np.asarray(cal_res.get("batch_calibrant_mz_values", [[]])[0]))
        cals = sorted(list(set([c for c in global_cals_raw if not np.isnan(c)])))

        cal_formulas = []
        orig_indices_aligned = []

        defined_calibrants = {}
        if self.calibration is not None and "calibrants" in self.calibration:
            defined_calibrants = self.calibration["calibrants"]

        # Match calibrant masses back to formula label strings
        for m_val in cals:
            raw_idx_matches = np.where(global_cals_raw == m_val)[0]
            orig_indices_aligned.append(raw_idx_matches[0])
            
            matched_formula = None
            best_ppm = 500.0
            
            if isinstance(defined_calibrants, dict):
                for name, theoretical_mass in defined_calibrants.items():
                    if theoretical_mass is not None and theoretical_mass > 0:
                        ppm = np.abs((theoretical_mass - m_val) / m_val) * 1e6
                        if ppm < best_ppm:
                            best_ppm = ppm
                            matched_formula = name
            
            if matched_formula is not None:
                cal_formulas.append(matched_formula)
            else:
                cal_formulas.append(f"Cal_{m_val:.2f}")

        # Local helper to clean formula label strings for Tofware compatibility
        def clean_formula_for_tofware(formula_item):
            if isinstance(formula_item, (int, float)):
                return f"Mass{float(formula_item):.2f}_"
            cleaned = str(formula_item).replace('+', '_').replace('-', '_')
            if not cleaned.endswith('_'):
                cleaned += '_'
            return cleaned

        # --- Step 2: Iterate and Export Per-File IF HDF5 Output Files ---
        for k, h5_src in sorted(self._file_handles.items()):
            src_path = Path(h5_src.filename)
            
            # Resolve target export folder path
            if output_dir is None:
                target_dir = src_path.parent / "IF"
            else:
                target_dir = Path(output_dir) / "IF"
                
            target_dir.mkdir(parents=True, exist_ok=True)
            out_filepath = target_dir / f"{src_path.stem}_IF.h5"
            
            # Slice global calibration parameter time tracks corresponding to current file timestamp bounds
            file_timestamps = self._file_timestamps[k]
            t_min = pd.Timestamp(np.min(file_timestamps)).to_datetime64()
            t_max = pd.Timestamp(np.max(file_timestamps)).to_datetime64()

            if len(global_times) > 0 and global_times.dtype.kind == 'M':
                file_mask = (global_times >= t_min) & (global_times <= t_max)
                if not np.any(file_mask):
                    file_midpoint = t_min + (t_max - t_min) / 2
                    closest_idx = np.argmin(np.abs(global_times - file_midpoint))
                    file_mask = np.zeros(len(global_times), dtype=bool)
                    file_mask[closest_idx] = True
            else:
                file_mask = np.ones(max(1, len(global_pars)), dtype=bool)

            sliced_pars = global_pars[file_mask] if len(global_pars) > 0 else np.array([[1000.0, -2000.0, 0.5]])
            sliced_times = global_times[file_mask] if len(global_times) > 0 else np.array([0.0])
            
            with h5py.File(out_filepath, "w") as h5_out:
                h5_out.attrs["TofwareFileType"] = np.float32(2.0)
                
                # Construct Peaks HDF5 group
                peaks_grp = h5_out.create_group("Peaks")
                peaks_grp.attrs["PeakListBslSigmaHi"] = np.float64(8.0)
                peaks_grp.attrs["PeakListBslSigmaLo"] = np.float64(-8.0)
                peaks_grp.attrs["bslBufMask"] = "0"
                peaks_grp.attrs["bslForceSmooth"] = np.uint8(1)
                peaks_grp.attrs["bslLoPass"] = np.uint8(1)
                peaks_grp.attrs["bslLoPass_N"] = np.uint32(20)
                peaks_grp.attrs["bslLoPass_f1"] = np.float32(0.0)
                peaks_grp.attrs["bslLoPass_f2"] = np.float32(0.05)
                peaks_grp.attrs["bslNbrParams"] = "2"
                peaks_grp.attrs["bslNoiseScaleFactor"] = np.float32(1.0)

                if self.peak_width_coeffs is not None:
                    peaks_grp.attrs["PeakWidthCoeffs"] = np.array(self.peak_width_coeffs, dtype="float64")
                
                if self.baseline is not None and "adjusted_baseline" in self.baseline:
                    peaks_grp.attrs["bslParams"] = ";".join(f"{v:.8f}" for v in self.baseline["adjusted_baseline"][:5]) + "..."
                else:
                    peaks_grp.attrs["bslParams"] = "0.00000000;0.00000000..."
                peaks_grp.attrs["bslUpdated"] = datetime.now().strftime("%Y-%m-%d %H:%M #00000")
                peaks_grp.attrs["bslWidth"] = np.uint8(10)
                
                peaks_grp.create_dataset("PeakShape", data=peak_shape_matrix, dtype="float64")
                peaks_grp.create_dataset("PeakWidthFit", data=np.array([0.0], dtype="float32"))
                peaks_grp.create_dataset("PeakwidthPoints", data=np.array([0.0], dtype="float32"))

                # Construct FullSpectra HDF5 group and set mass calibration attributes
                fs_grp = h5_out.create_group("FullSpectra")
                fs_grp.attrs["MassCalibMode"] = np.int8(mode_val)
                fs_grp.attrs["MassCalibration FitConst"] = ""
                fs_grp.attrs["MassCalibration Function"] = eq_strings.get(mode_val, "p1*m^p3+p2")
                fs_grp.attrs["MassCalibration PeakFitType"] = fit_type
                fs_grp.attrs["MassCalibration PkSR"] = np.float64(20.0)
                fs_grp.attrs["MassCalibration bufAv_flag"] = np.float32(0.0)
                fs_grp.attrs["MassCalibration calcMethod"] = np.float32(0.0)
                fs_grp.attrs["MassCalibration eBuf"] = np.float32(-1.0)
                fs_grp.attrs["MassCalibration n_smooth"] = np.float32(0.0)
                fs_grp.attrs["MassCalibration n_wrAv"] = np.float32(0.0)
                fs_grp.attrs["MassCalibration preAv_flag"] = np.float32(1.0)
                fs_grp.attrs["MassCalibration sBuf"] = np.float32(-1.0)
                fs_grp.attrs["MassCalibration timeAv_flag"] = np.float32(1.0)
                # fs_grp.attrs["MassCalibration timeAv_mins"] = np.float32(2.0)
                fs_grp.attrs["MassCalibration timeAv_mins"] = int(self._averaged_dataset_interval / 60)
                    
                fs_grp.attrs["MassCalibration nbrPoints"] = np.int8(len(cals))
                avg_file_pars = np.mean(sliced_pars, axis=0)
                fs_grp.attrs["MassCalibration nbrParameters"] = np.int8(len(avg_file_pars))
                
                for idx_p, p_val in enumerate(avg_file_pars):
                    fs_grp.attrs[f"MassCalibration p{idx_p+1}"] = np.float64(p_val)
                    
                for idx_c, m_val in enumerate(cals, 1):
                    raw_formula = cal_formulas[idx_c - 1]
                    cleaned_label = clean_formula_for_tofware(raw_formula)
                    orig_raw_idx = orig_indices_aligned[idx_c - 1]
                    
                    fs_grp.attrs[f"MassCalibration f{idx_c}"] = "null"
                    fs_grp.attrs[f"MassCalibration l{idx_c}"] = cleaned_label
                    fs_grp.attrs[f"MassCalibration m{idx_c}"] = np.float64(m_val)
                    fs_grp.attrs[f"MassCalibration thr{idx_c}"] = np.float64(0.0)
                    fs_grp.attrs[f"MassCalibration w{idx_c}"] = np.float64(1.0)
                    
                    if "batch_measured_sip" in cal_res and len(cal_res["batch_measured_sip"]) > 0:
                        t_val = cal_res["batch_measured_sip"][0][orig_raw_idx]
                        if np.isnan(t_val):
                            t_val = (m_val ** 0.5) * 5000.0
                    else:
                        t_val = (m_val ** 0.5) * 5000.0
                    fs_grp.attrs[f"MassCalibration t{idx_c}"] = np.float64(t_val)

                # Write MassAxis dataset
                m_axis = self.reference.get("rs_mass_axis", self.first_guess_mass_axis) if self.reference else self.first_guess_mass_axis
                
                if m_axis is None:
                    if self.tofdata is not None:
                        tof_len = self.tofdata.shape[-1]
                    elif hasattr(self, "sample_index_axis") and self.sample_index_axis is not None:
                        tof_len = len(self.sample_index_axis)
                    else:
                        tof_len = 100000
                    print("Please check that a reference spectra has been defined or that the"
                          "first_guess_mass_axis is defined. Currently writing a dummy entry for the 'MassAxis' dataset.")
                    m_axis = np.linspace(1.0, 400.0, tof_len, dtype="float64")

                m_axis = m_axis.astype(np.float64)
                fs_ma_group = fs_grp.create_dataset("MassAxis", data=m_axis, dtype="float64")
                fs_ma_group.attrs['IGORWaveType'] = 4
                fs_ma_group.attrs['IGORWaveScaling'] = [[0.00000000e+00, 0.00000000e+00], [9.99999972e-10, 1.00000000e-07]]
                fs_ma_group.attrs['TofwareDoNotIndex'] = 1.0
                fs_grp.create_dataset("masscalibration peakshape", data=y_ps, dtype="float32")

                # Construct MassCalib HDF5 group for time-series parameter tracks
                mc_grp = h5_out.create_group("MassCalib")
                n_file_intervals = len(sliced_pars)
                
                mc_grp.create_dataset("coAdds", data=np.ones(n_file_intervals, dtype="float32"))
                
                if global_flags is not None:
                    sliced_flags = global_flags[file_mask]
                else:
                    sliced_flags = np.zeros(n_file_intervals, dtype="int32")
                    
                mc_grp.create_dataset("flag", data=sliced_flags, dtype="int32")
                
                pars_4d = sliced_pars[:, np.newaxis, np.newaxis, :]
                mc_grp.create_dataset("pars", data=pars_4d, dtype="float64")
                
                if sliced_times.dtype.kind == 'M':
                    sliced_times_numeric = (sliced_times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D')
                else:
                    sliced_times_numeric = np.asarray(sliced_times, dtype="float64")
                mc_grp.create_dataset("time", data=sliced_times_numeric, dtype="float64")
                
                n_cals = len(cals) if len(cals) > 0 else 1
                mc_grp.create_dataset("masses", data=np.tile(cals, (n_file_intervals, 1)), dtype="float32")
                mc_grp.create_dataset("ppm", data=np.zeros(n_file_intervals, dtype="uint16"))
                mc_grp.create_dataset("ppmMasses", data=np.zeros((n_file_intervals, 1, 1, n_cals), dtype="float64"))
                mc_grp.create_dataset("tofs", data=np.zeros((n_file_intervals, 1, 1, n_cals), dtype="float32"))
                mc_grp.create_dataset("writes", data=np.arange(n_file_intervals, dtype="uint32"))

            print(f" -> Exported 'Instrument Function' file: {out_filepath.name}")

        print("Global IF batch export finished successfully.")
        return self


    # =========================================================================
    # IMPORT HELPERS
    # =========================================================================
    def _import_ot(self, path_obj):
        """
        Private helper to deserialize a complete native OpenTof state file (.h5) into memory.

        Restores all saved scalar attributes, primary axis arrays, time-series calibration maps, 
        empirical peak shape interpolators, 2D out-of-core Dask spectral datasets (`tofdata`), 
        and wide-format pandas DataFrames (`peak_data`, `time_series`).

        Parameters
        ----------
        path_obj : pathlib.Path
            Path pointing to the native OpenTof workspace state `.h5` file.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance populated with restored attributes and state caches.
        """
        print(f"Loading OpenTof Deployment state from: {path_obj}")
        
        with h5py.File(path_obj, "r") as h5:
            # --- Step 1: Restore Root Scalar Attributes & Flags ---
            self._user_entered_cycling_flag = bool(h5.attrs.get("_user_entered_cycling_flag", False))
            self.chunk_size = int(h5.attrs.get("chunk_size", 1000))
            
            if "target_nbr_samples" in h5.attrs:
                self.target_nbr_samples = int(h5.attrs["target_nbr_samples"])
            if "instrument_type" in h5.attrs:
                self.instrument_type = str(h5.attrs["instrument_type"])
            if "averaged_dataset_interval" in h5.attrs:
                self._averaged_dataset_interval = h5.attrs["averaged_dataset_interval"]

            # --- Step 2: Restore 1D Primary Axes & Polynomial Functions ---
            if "tof_axis" in h5:
                self.tof_axis = h5["tof_axis"][:]
            if "first_guess_mass_axis" in h5:
                self.first_guess_mass_axis = h5["first_guess_mass_axis"][:]
            
            if "peak_width_coeffs" in h5:
                self.peak_width_coeffs = h5["peak_width_coeffs"][:]
                self.peak_width_function = np.poly1d(self.peak_width_coeffs)

            if "global_sort_idx" in h5:
                self._global_sort_idx = h5["global_sort_idx"][:]
            if "timestamps_cache" in h5:
                self._timestamps_cache = h5["timestamps_cache"][:].astype("datetime64[ns]")

            if "reagent_ion" in h5.attrs:
                self.reagent_ion = str(h5.attrs["reagent_ion"])
            if "plot_dir" in h5.attrs:
                self.plot_dir = str(h5.attrs["plot_dir"])
            if "segment_profiles" in h5.attrs:
                self.segment_profiles = json.loads(h5.attrs["segment_profiles"])

            # Restore raw peak width polynomial floor and corruption tracking masks
            if "raw_peak_width_coeffs" in h5:
                self.raw_peak_width_coeffs = h5["raw_peak_width_coeffs"][:]
                self.raw_peak_width_function = np.poly1d(self.raw_peak_width_coeffs)
            if "_corrupt_flags" in h5:
                self._corrupt_flags = h5["_corrupt_flags"][:]

            # --- Step 3: Re-chunk Out-of-Core TOF Dataset ---
            if "tofdata" in h5:
                self._tofdata = da.from_array(h5["tofdata"], chunks=(self.chunk_size, -1))
                print("Deployment.tofdata reconstructed!")

            # --- Step 4: Recursive Group-to-Dictionary Restorer ---
            def load_group_to_dict(hdf5_group):
                res_dict = {}
                for attr_k, attr_v in hdf5_group.attrs.items():
                    res_dict[attr_k] = attr_v
                for k in hdf5_group.keys():
                    node = hdf5_group[k]
                    if isinstance(node, h5py.Group):
                        res_dict[k] = load_group_to_dict(node)
                    elif isinstance(node, h5py.Dataset):
                        data_arr = node[:]
                        if data_arr.dtype.kind in ('S', 'O'):
                            data_arr = data_arr.astype(str)
                        if node.attrs.get("is_datetime", False):
                            data_arr = data_arr.astype("datetime64[ns]")
                        
                        if data_arr.ndim == 0:
                            res_dict[k] = data_arr.item()
                        else:
                            res_dict[k] = data_arr if data_arr.dtype.kind != 'U' else data_arr.tolist()
                return res_dict

            # Map restored HDF5 group dictionaries to instance property attributes
            dict_restore_mappings = {
                "calibration": "calibration",
                "reference": "reference",
                "baseline": "baseline",
                "peak_list": "peak_list",
                "nm_data": "nm_data",
                "cycling_status": "_cycling_status",
                "isotopes": "isotopes"
            }
            for h5_path, attr_name in dict_restore_mappings.items():
                if h5_path in h5:
                    setattr(self, attr_name, load_group_to_dict(h5[h5_path]))

            # Restore Global Averaged Dataset (GAD) tuple
            if "averaged_dataset" in h5:
                avg_grp = h5["averaged_dataset"]
                avg_data = avg_grp["averaged_data"][:]
                avg_ts = avg_grp["averaged_timestamps"][:].astype("datetime64[ns]")
                avg_idx = avg_grp["interval_indices"][:]
                self._averaged_dataset = (avg_data, avg_ts, avg_idx)

            # Reconstruct baseline-subtracted TOF array via 2D broadcasting
            if self.baseline and "adjusted_baseline" in self.baseline and self._tofdata is not None:
                adj_bsl = np.array(self.baseline["adjusted_baseline"])
                self.tofdata_subtracted = self._tofdata - adj_bsl.reshape(1, -1)

            # Re-bind custom peak shape function if interpolator grid is present
            if self.calibration and self.baseline and "batch_params" in self.calibration:
                if getattr(self, "peak_shape_interp", None) is not None:
                    def restored_shape(x, A, x_c, FWHM):
                        return A * self.peak_shape_interp((x - x_c) * 2.3548 / FWHM)
                    
                    if hasattr(self, "peak_shape_interp") and hasattr(self.peak_shape_interp, 'x'):
                        x_comm = self.peak_shape_interp.x
                        y_m = self.peak_shape_interp.y
                        restored_shape.ps_sigma = np.sum(y_m) * (x_comm[1] - x_comm[0])
                    else:
                        restored_shape.ps_sigma = 1.0
                        
                    restored_shape.gauss_to_ps = 1.0
                    self.custom_peak_shape = restored_shape

            # Enforce 1D NumPy array types for peak list metadata arrays
            if self.peak_list and isinstance(self.peak_list, dict):
                for arr_key in ["centers", "fwhms"]:
                    if arr_key in self.peak_list:
                        self.peak_list[arr_key] = np.array(self.peak_list[arr_key])

            # --- Step 5: Restore Wide-Format DataFrames ---
            for df_attr in ["peak_data", "time_series"]:
                if df_attr in h5:
                    grp = h5[df_attr]
                    df_dict = {col: grp[col][:] for col in grp.keys() if col != "_index"}
                    restored_df = pd.DataFrame(df_dict, index=grp["_index"][:])
                    setattr(self, df_attr, restored_df)
            
            # Re-construct empirical 1D cubic spline interpolator from saved shape grid
            if "peak_shape_data" in h5:
                ps_grp = h5["peak_shape_data"]
                x_grid = ps_grp["x"][:]
                y_grid = ps_grp["y"][:]
                self.peak_shape_interp = interp1d(x_grid, y_grid, kind="cubic", fill_value=0, bounds_error=False)
                
                if "custom_shape_area_ratio" in ps_grp.attrs:
                    self.custom_shape_area_ratio = float(ps_grp.attrs["custom_shape_area_ratio"])
                    
                def restored_shape(x, A, x_c, FWHM):
                    return A * self.peak_shape_interp((x - x_c) * 2.3548 / FWHM)
                restored_shape.ps_sigma = np.sum(y_grid) * (x_grid[1] - x_grid[0])
                restored_shape.gauss_to_ps = 1.0
                self.custom_peak_shape = restored_shape

        print("OpenTof Deployment file import completed successfully.")
        return self

    def _import_p(self, path_obj):
        """
        Private helper to import Tofware Processed (_p.h5) result files.

        Discovers `_p.h5` files in target paths or `Processed/` subfolders, extracts compound 
        labels and integration limits from `PeakTable` arrays, flattens 4D `PeakData` matrices, 
        re-creates absolute datetime timestamps from FILETIME headers, sorts writebufs chronologically, 
        and merges results into `self.peak_data`.

        Parameters
        ----------
        path_obj : pathlib.Path
            File path or directory path targeting processed HDF5 files.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with `self.peak_data` DataFrame populated.
        """
        # Discover matching Processed (_p.h5) files in directory or subfolder
        if path_obj.is_dir():
            if (path_obj / "Processed").exists():
                files_to_load = sorted(list((path_obj / "Processed").glob("*_p.h5")))
            else:
                files_to_load = sorted(list(path_obj.glob("*_p.h5")))
        else:
            files_to_load = [path_obj]

        # Guard clause: Exit gracefully if no matching files are found
        if not files_to_load:
            print(f"⚠️ No matching _p.h5 files discovered at target: {path_obj}")
            return self
            
        print(f"Importing peak data from {len(files_to_load)} Processed (_P) files...")

        all_peak_data_frames = []
        peak_list_populated = False

        # Loop through discovered Processed files to extract PeakTable and PeakData
        for idx, file in enumerate(files_to_load):
            with h5py.File(file, "r") as h5:
                
                # Extract target compound labels, centers, and integration FWHMs from PeakTable
                if "PeakData" in h5 and "PeakTable" in h5["PeakData"]:
                    ptable = h5["PeakData"]["PeakTable"][:]
                    if not peak_list_populated:
                        peaks = [row['label'].decode('utf-8') for row in ptable]
                        masses = [row['mass'] for row in ptable]
                        
                        self.peak_list = {'peaks': peaks, 'centers': np.array(masses)}
                        
                        fwhms = [(row['upper integration limit'] - row['lower integration limit']) / 1.5 for row in ptable]
                        self.peak_list['fwhms'] = np.array(fwhms)
                        peak_list_populated = True

                    # Extract 4D PeakData matrix (writes, bufs, segments, peaks)
                    if "PeakData" in h5["PeakData"]:
                        pd_matrix = h5["PeakData"]["PeakData"][:]
                        writes, bufs, segs, npeaks = pd_matrix.shape
                        flat_intensities = pd_matrix[:, :, 0, :].reshape(writes * bufs, npeaks)
                        
                        # Process acquisition timing arrays and calculate pandas DatetimeIndex
                        if "TimingData" in h5 and "BufTimes" in h5["TimingData"] and "AcquisitionTimeZero" in h5["TimingData"].attrs:
                            time_zero_raw = self._get_scalar_attr(h5["TimingData"], "AcquisitionTimeZero")
                            time_zero = self._convert_filetime_to_pandas_datetime(time_zero_raw)
                            
                            if hasattr(time_zero, '__len__') and not isinstance(time_zero, pd.Timestamp):
                                time_zero = time_zero[0]

                            buf_times = h5["TimingData"]["BufTimes"][:].flatten()
                            
                            safe_len = min(len(buf_times), len(flat_intensities))
                            buf_times = buf_times[:safe_len]

                            valid_mask = (buf_times > 0) | (np.arange(len(buf_times)) == 0)
                            valid_indices = np.where(valid_mask)[0]
                            
                            flat_ts = np.array([time_zero + pd.to_timedelta(s, unit='s') for s in buf_times]).flatten()
                            
                            # Sort timestamps chronologically within file
                            sorted_relative_idx = np.argsort(flat_ts[valid_indices])
                            sort_idx = valid_indices[sorted_relative_idx]
                            
                            sorted_ts = flat_ts[sort_idx]
                            sorted_intensities = flat_intensities[sort_idx]
                            
                            # Build pandas DataFrame for current file
                            df_dict = {'time': sorted_ts}
                            for i, p_name in enumerate(self.peak_list['peaks']):
                                df_dict[f"{p_name}_amplitude"] = sorted_intensities[:, i]
                            
                            all_peak_data_frames.append(pd.DataFrame(df_dict))

                # Extract first guess mass axis from first file
                if idx == 0 and "FullSpectra" in h5 and "MassAxis" in h5["FullSpectra"]:
                    self.first_guess_mass_axis = h5["FullSpectra"]["MassAxis"][:]

        # Concatenate file-level DataFrames into master time-series peak_data
        if all_peak_data_frames:
            combined_df = pd.concat(all_peak_data_frames, ignore_index=True)
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            combined_df['MS_index'] = combined_df.index
            self.peak_data = combined_df
            print(" -> Successfully imported and merged PeakData arrays.")
            
        print("Processed (_p) files imported successfully.")
        return self

    def _import_if(self, path_obj):
        """
        Private helper to import Tofware Instrument Function (_IF.h5) configuration files.

        Discovers `_IF.h5` files in target paths or `IF/` subfolders, extracts peak width 
        polynomial coefficients, reconstructs 1D empirical peak shape interpolators, parses 
        calibration parameter time-tracks from `MassCalib/pars`, concatenates calibration 
        history across files, and automatically triggers :meth:`generate_averaged_dataset`.

        Parameters
        ----------
        path_obj : pathlib.Path
            File path or directory path targeting instrument function HDF5 files.

        Returns
        -------
        self : Deployment
            Updated `Deployment` instance with calibration and peak shape attributes populated.
        """
        # Discover matching Instrument Function (_IF.h5) files in directory or subfolder
        if path_obj.is_dir():
            if (path_obj / "IF").exists():
                files_to_load = sorted(list((path_obj / "IF").glob("*_IF.h5")))
            else:
                files_to_load = sorted(list(path_obj.glob("*_IF.h5")))
        else:
            files_to_load = [path_obj]

        # Guard clause: Exit gracefully if no matching files are found
        if not files_to_load:
            print(f"⚠️ No matching _IF.h5 files discovered at target: {path_obj}")
            return self

        print(f"Importing and merging configuration parameters from {len(files_to_load)} files...")

        all_params = []
        all_times = []
        all_flags = []
        cal_masses = []
        mode_val = 2

        timeAv_mins_list = []

        # Loop through discovered IF files to extract peak shapes, width coeffs, and mass cal tracks
        for idx, file in enumerate(files_to_load):
            with h5py.File(file, "r") as h5:
                if idx == 0:
                    # Extract peak width polynomial coefficients and empirical peak shape grid
                    if "Peaks" in h5:
                        p_grp = h5["Peaks"]
                        
                        if "PeakWidthCoeffs" in p_grp.attrs:
                            coeffs = self._get_scalar_attr(p_grp, "PeakWidthCoeffs")
                            if coeffs is not None:
                                self.peak_width_coeffs = np.atleast_1d(coeffs)
                                self.peak_width_function = np.poly1d(self.peak_width_coeffs)
                                print(f" -> Reconstructed peak_width_function via poly1d: {self.peak_width_coeffs.tolist()}")
                        
                        if "PeakShape" in p_grp:
                            ps_data = p_grp["PeakShape"][:]
                            x_common = ps_data[:, 0]
                            y_mean = ps_data[:, 1]
                            
                            self.peak_shape_interp = interp1d(x_common, y_mean, kind="cubic", fill_value=0, bounds_error=False)
                            
                            def restored_shape(x, A, x_c, FWHM):
                                return A * self.peak_shape_interp((x - x_c) * 2.3548 / FWHM)
                            restored_shape.ps_sigma = np.sum(y_mean) * (x_common[1] - x_common[0])
                            restored_shape.gauss_to_ps = 1.0
                            self.custom_peak_shape = restored_shape
                        
                    # Extract reference mass axis and calibrant masses
                    if "FullSpectra" in h5:
                        fs_grp = h5["FullSpectra"]
                        if "MassAxis" in fs_grp:
                            self.first_guess_mass_axis = fs_grp["MassAxis"][:]
                            if self.reference is None: 
                                self.reference = {}
                            self.reference["rs_mass_axis"] = self.first_guess_mass_axis
                        mode_val = int(self._get_scalar_attr(fs_grp, "MassCalibMode")) if "MassCalibMode" in fs_grp.attrs else 2
                        
                        nbr_pts = int(self._get_scalar_attr(fs_grp, "MassCalibration nbrPoints")) if "MassCalibration nbrPoints" in fs_grp.attrs else 0
                        for m_idx in range(1, nbr_pts + 1):
                            attr_m = f"MassCalibration m{m_idx}"
                            if attr_m in fs_grp.attrs:
                                cal_masses.append(float(self._get_scalar_attr(fs_grp, attr_m)))

                # Record timeAv_mins averaging interval attribute
                if "FullSpectra" in h5:
                    fs_grp = h5["FullSpectra"]
                    if "MassCalibration timeAv_mins" in fs_grp.attrs:
                        t_val = self._get_scalar_attr(fs_grp, "MassCalibration timeAv_mins")
                        if t_val is not None:
                            timeAv_mins_list.append(float(t_val))

                # Extract time-series mass calibration parameter matrices
                if "MassCalib" in h5:
                    mc_grp = h5["MassCalib"]
                    if "pars" in mc_grp:
                        pars_squeezed = mc_grp["pars"][:].squeeze(axis=(1, 2))
                        if pars_squeezed.ndim == 1:
                            pars_squeezed = pars_squeezed[np.newaxis, :]
                        all_params.append(pars_squeezed)
                        
                        file_times = mc_grp["time"][:] if "time" in mc_grp else np.array([0.0])
                        all_times.append(file_times)
                        
                        file_flags = mc_grp["flag"][:] if "flag" in mc_grp else np.zeros(len(pars_squeezed), dtype="int32")
                        all_flags.append(file_flags)

        # Merge and sort calibration parameters chronologically across files
        if all_params:
            merged_pars = np.concatenate(all_params, axis=0)
            merged_times = np.concatenate(all_times, axis=0)
            merged_flags = np.concatenate(all_flags, axis=0)

            sort_order = np.argsort(merged_times)
            
            self.calibration = {
                "batch_params": merged_pars[sort_order].tolist(),
                "mass_cal_mode": mode_val,
                "averaged_timestamps": merged_times[sort_order],
                "batch_calibrant_mz_values": [cal_masses] * len(merged_pars) if cal_masses else [],
                "flags": merged_flags[sort_order].tolist()
            }
            
            if cal_masses:
                if self.peak_width_function is not None:
                    self.populate_peak_list_and_isotopes(cal_masses)
                else:
                    print("\n[WARNING] Calibrant masses found, but 'peak_width_function' is not yet initialized.")
                    print("          deployment.peak_list matrix assignment skipped to prevent TypeErrors.")
                    print("          Please execute d.determine_peak_width() manually to build resolution bounds.\n")
                
            print(f" -> Successfully unified processing states across {len(merged_pars)} historical calibration blocks.")

        # Evaluate timeAv_mins consistency and trigger automated Global Averaged Dataset generation
        if timeAv_mins_list:
            consensus_mins = timeAv_mins_list[0]
            
            if not all(t == consensus_mins for t in timeAv_mins_list):
                print(f"\n[WARNING] Inconsistent 'timeAv_mins' values discovered across files: {set(timeAv_mins_list)}")
                print(f"          Falling back to the consensus value: {consensus_mins} minutes.\n")
                
            averaging_interval_seconds = int(consensus_mins * 60)
            print(f" -> Automatically generating averaged dataset with an interval of {averaging_interval_seconds} seconds...")
            self.generate_averaged_dataset(averaging_interval=averaging_interval_seconds)

        print("IF files imported successfully.")
        return self


    def export_peak_list_to_tofware(self, output_filepath=None):
        """
        Export `self.peak_list` to a tab-delimited text file matching Tofware's peak table schema.

        Parses target peak formulas or numeric m/z values, calculates +-1.5 * FWHM 
        integration limits, determines ionic charge values from formula sign indicators, extracts 
        neutral stoichiometry sum formulas, and exports a tab-separated text file compatible 
        with Tofware peak lists.

        Parameters
        ----------
        output_filepath : str or pathlib.Path or None, default=None
            Target file path for the exported `.txt` file. Defaults to ``"tofware_peak_list.txt"`` 
            inside `self.plot_dir` / `default_plot_subdir` if ``None``.

        Returns
        -------
        df_tofware : pandas.DataFrame
            DataFrame formatted according to Tofware's 28-column peak list schema.

        Raises
        ------
        ValueError
            If `self.peak_list` is unpopulated or missing the `'peaks'` key.
        """
        # Local imports to prevent circular dependencies
        import re
        from pathlib import Path
        from opentof.utils import get_default_plot_dir

        # Guard clause: Ensure peak list is populated
        if getattr(self, 'peak_list', None) is None or 'peaks' not in self.peak_list:
            raise ValueError("deployment.peak_list is not populated. Call `populate_peak_list_and_isotopes()` first.")

        peaks = self.peak_list['peaks']
        centers = self.peak_list['centers']
        fwhms = self.peak_list.get('fwhms', np.full(len(peaks), 0.5))

        # Standard 28-column schema required by Tofware peak table importers
        columns = [
            "def", "fit", "ion", "x0", "tag", "sumFormula", 
            "x_Lo", "x_Hi", "d2_Ctr", "d2_Lo", "d2_Hi", 
            "d1_Ctr", "d1_Lo", "d1_Hi", "d0_Ctr", "d0_Lo", "d0_Hi", 
            "calFac", "calUnit", "charge", "ionizFrac", "fragOf", 
            "isotopeOf", "comments", "standards", "families", "baseline", "intCalIn"
        ]

        rows = []
        unknown_counter = 1

        # Construct row entries for each peak target
        for i, peak_item in enumerate(peaks):
            x0 = float(centers[i])
            fwhm = float(fwhms[i])
            
            # Define integration boundaries as +/- 1.5 * FWHM around peak center
            x_lo = x0 - (fwhm * 1.5)
            x_hi = x0 + (fwhm * 1.5)

            # Handle numeric masses or generic unknown peak placeholders
            if isinstance(peak_item, (int, float)) or str(peak_item).startswith("<unknown"):
                ion_name = f"<unknown{unknown_counter:04d}>"
                sum_formula = ""
                charge_val = ""
                unknown_counter += 1
            else:
                # Handle chemical formula string
                ion_name = str(peak_item)
                
                # Determine integer charge value from '+' or '-' sign indicators
                pos_charges = len(re.findall(r'\+', ion_name))
                neg_charges = len(re.findall(r'-', ion_name))
                if pos_charges > 0:
                    charge_val = 1 if pos_charges == 1 else pos_charges
                elif neg_charges > 0:
                    charge_val = -1 if neg_charges == 1 else -neg_charges
                else:
                    charge_val = ""

                # Extract neutral stoichiometry string by stripping brackets, isotope numbers, and charges
                cleaned = re.sub(r'\[\d+\]', '', ion_name)
                cleaned = re.sub(r'\[\d+([A-Z][a-z]?)\]', r'\1', cleaned)
                sum_formula = re.sub(r'[\+-]', '', cleaned)

            # Populate structured row dictionary matching schema
            row = {col: "" for col in columns}
            row["def"] = 1
            row["fit"] = 1
            row["ion"] = ion_name
            row["x0"] = f"{x0:.6f}"
            row["sumFormula"] = sum_formula
            row["x_Lo"] = f"{x_lo:.4f}"
            row["x_Hi"] = f"{x_hi:.4f}"
            row["charge"] = charge_val

            rows.append(row)

        df_tofware = pd.DataFrame(rows, columns=columns)

        # Resolve output text file path
        if output_filepath is None:
            base_dir = self.plot_dir if getattr(self, 'plot_dir', None) is not None else get_default_plot_dir()
            out_path = Path(base_dir) / self.default_plot_subdir / "tofware_peak_list.txt"
        else:
            out_path = Path(output_filepath)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export tab-delimited text file
        df_tofware.to_csv(out_path, sep="\t", index=False)
        print(f"✅ Tofware peak list exported successfully ({len(df_tofware)} peaks) to:\n  -> {out_path}")

        return df_tofware

class HDF5ArrayWrapper:
    """
    Picklable array wrapper interfacing HDF5 dataset slicing operations with Dask.

    Performs on-demand loading of raw HDF5 write/buffer dataset slices, handles segment 
    extraction and multi-segment summation, applies scalar count-rate scaling multipliers 
    (converting raw counts to ions/second), truncates sample channels to `target_nbr_samples`, 
    and re-orders writebufs into chronological order using pre-computed sorting indices.

    Parameters
    ----------
    filepath : str or pathlib.Path
        File path to the raw HDF5 acquisition file.
    key : str
        HDF5 dataset key string (e.g., ``"FullSpectra/TofData"``).
    sort_idx : numpy.ndarray
        1D integer array mapping raw writebuf indices to chronological order.
    seg_list : list of int or None, default=None
        List of segment indices to extract and sum for multi-segment data.
    nbr_waveforms : float, default=1.0
        Total number of waveform accumulations per writebuf.
    SIS : float, default=1.0
        Single Ion Signal scaling factor from HDF5 metadata attributes.
    tof_period : float, default=25008.0
        TOF extraction pulse period in nanoseconds.
    target_nbr_samples : int or None, default=None
        Standardized sample channel truncation length.

    Attributes
    ----------
    scale_factor : numpy.float32
        Pre-computed scalar multiplier converting raw spectral values to ions/second.
    shape : tuple of int
        Virtual 2D array dimensions ``(num_spectra, num_samples)`` presented to Dask.
    dtype : numpy.dtype
        Output NumPy data type (``numpy.float32``).
    ndim : int
        Array dimension count (`2`).
    """

    def __init__(self, filepath, key, sort_idx, seg_list=None, 
                 nbr_waveforms=1.0, SIS=1.0, tof_period=25008.0, 
                 target_nbr_samples=None):
        self.filepath = str(filepath)
        self.key = key
        self.sort_idx = np.asarray(sort_idx, dtype=np.int64)
        self.seg_list = seg_list
        self.target_nbr_samples = target_nbr_samples

        # Pre-compute scalar scaling multiplier converting raw counts to ions/second:
        # scale_factor = (1e6 / tof_period * 1e3) / (nbr_waveforms * num_segs * SIS)
        multiplier = (1e6 / float(tof_period)) * 1e3
        num_segs = len(seg_list) if seg_list is not None else 1
        effective_waveforms = float(nbr_waveforms) * num_segs
        self.scale_factor = np.float32(multiplier / (effective_waveforms * float(SIS)))

        # Read raw dataset shape from open HDF5 file handle
        with h5py.File(self.filepath, 'r') as f:
            dset = f[self.key]
            self.raw_shape = dset.shape
            self.n_bufs = self.raw_shape[1]
            self.raw_samples = self.raw_shape[-1]

        # Determine effective sample channel length after truncation
        self.n_samples = (
            self.target_nbr_samples 
            if (self.target_nbr_samples and self.raw_samples > self.target_nbr_samples) 
            else self.raw_samples
        )

        # Virtual 2D shape presented to Dask: (num_sorted_spectra, num_samples)
        self.shape = (len(self.sort_idx), self.n_samples)
        self.dtype = np.float32
        self.ndim = 2

    def __getitem__(self, keys):
        """
        Slice and load requested spectrum blocks from the raw HDF5 dataset.

        Translates Dask virtual 2D slice keys into 3D/4D HDF5 write/buf indices, loads 
        minimal bounding box dataset windows from disk, sums segment slices, applies 
        the pre-computed count-rate scalar multiplier, and returns a 2D float32 array.

        Parameters
        ----------
        keys : tuple or slice or int
            Slice index tuple specifying target spectrum and sample channel boundaries.

        Returns
        -------
        extracted : numpy.ndarray
            2D float32 array of shape ``(num_requested_spectra, num_requested_samples)`` 
            containing scaled spectral intensity values in ions/second.
        """
        # Parse slice keys passed from Dask array graph evaluations
        if isinstance(keys, tuple):
            spec_slice = keys[0]
            sample_slice = keys[1] if len(keys) > 1 else slice(None)
        else:
            spec_slice = keys
            sample_slice = slice(None)

        # Convert spectrum slice to integer index array
        if isinstance(spec_slice, slice):
            spec_indices = np.arange(*spec_slice.indices(self.shape[0]))
        else:
            spec_indices = np.atleast_1d(spec_slice)

        # Return empty 2D array if zero spectra are requested
        if len(spec_indices) == 0:
            return np.empty((0, self.n_samples), dtype=np.float32)

        # Map virtual sorted spectrum indices to raw flat writebuf positions
        raw_flat_indices = self.sort_idx[spec_indices]
        writes = raw_flat_indices // self.n_bufs
        bufs = raw_flat_indices % self.n_bufs

        w_min, w_max = np.min(writes), np.max(writes)

        # Read minimal bounding box window from HDF5 file on disk
        with h5py.File(self.filepath, 'r') as f:
            dset = f[self.key]
            
            try:
                # Process 4D dataset layout (writes, bufs, segments, samples)
                if dset.ndim == 4:
                    if self.seg_list is not None:
                        if len(self.seg_list) == 1:
                            raw_data = dset[w_min:w_max+1, :, self.seg_list[0]:self.seg_list[0]+1, :]
                        else:
                            raw_data = dset[w_min:w_max+1, :, self.seg_list, :]
                            raw_data = np.sum(raw_data, axis=2, keepdims=True)
                    else:
                        raw_data = dset[w_min:w_max+1, :, :, :]
                        if raw_data.shape[2] > 1:
                            raw_data = np.sum(raw_data, axis=2, keepdims=True)
                # Process 3D dataset layout (writes, bufs, samples)
                else:
                    raw_data = dset[w_min:w_max+1, :, :]
                    raw_data = raw_data[:, :, np.newaxis, :]

                # Extract target writebuf slices relative to bounding window minimum
                rel_writes = writes - w_min
                extracted = raw_data[rel_writes, bufs, 0, :]

            except (OSError, IndexError):
                # Fallback recovery loop: Read write-by-write to recover valid chunks and zero-fill truncated ones
                extracted = np.zeros((len(writes), self.raw_samples), dtype=np.float32)
                for i, (w, b) in enumerate(zip(writes, bufs)):
                    try:
                        if dset.ndim == 4:
                            seg = self.seg_list[0] if (self.seg_list and len(self.seg_list) == 1) else slice(None)
                            val = dset[w, b, seg, :]
                            if val.ndim > 1:
                                val = np.sum(val, axis=0)
                        else:
                            val = dset[w, b, :]
                        extracted[i] = val
                    except (OSError, IndexError):
                        pass

        # Multiply raw integer counts by count-rate scaling factor in-place
        extracted = extracted.astype(np.float32, copy=False)
        extracted *= self.scale_factor

        # Truncate sample channels to target_nbr_samples if required
        if self.target_nbr_samples and extracted.shape[-1] > self.target_nbr_samples:
            extracted = extracted[:, :self.target_nbr_samples]

        # Apply sample channel slice bounds if specified
        if sample_slice != slice(None):
            extracted = extracted[:, sample_slice]

        return extracted
