# Installation

**OpenTof** is a Python module that (like most Python modules) relies on code that has already been written by others to run.

### **There are three steps to installing OpenTof in its current state (beta release)**:

1. Create a Python environment using your preferred method (.venv or conda).
2. Download the zip file and extract (or clone) the OpenTof repository to a known location.
3. Navigate to where the repository was extracted or cloned to and run: `pip install -e .` to install OpenTof as an editable Python package. The `.` (period) tells pip to install the current working directory as a Python package so it is critical that this command is run from the directory with the contents `README.md`, `pyproject.toml`, and `/opentof`! After OpenTof is installed in your current Python environment you do not need to keep working within this directory (in fact its better not to!).

Whether relying on OpenTof to install necessary code dependancies or installing them yourself the following packages are necessary in addition to OpenTof itself:
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- dask
- chemparse
- customtkinter
- pyarrow
- minisom (must be pip installed)

If you plan to use Jupyter notebooks also install:
- jupyter
- jupyterlab
- notebook
- ipykernel
- ipywidgets

After meeting these requirements OpenTof should be installed locally within the specified Python environment! Additionally, because the package was installed in "editable" mode any changes you make to the source code should be reflected in the code at next runtime.