# Installation

**OpenTof** is a Python module that (like most Python modules) relies on code that has already been written by others to run.

Luckily we can install opentof through pypi to take care of these dependancies for us!

Opentof can be installed by running: `pip install opentof` in a Python environment that has pip installed.

If the above processes doesn't work for you or you want to try a different way, below is the method I use when working on OpenTof's source code:

### **There are three steps to installing OpenTof in its current state if you expect to make changes to the code (beta release)**:

1. Create a Python environment using your preferred method (.venv or conda).
2. Download the zip file and extract (or clone) the OpenTof repository to a known location.
3. Navigate to where the repository was extracted or cloned to and run: `pip install -e .` to install OpenTof as an editable Python package. The `.` (period) tells pip to install the current working directory as a Python package, so it is critical that this command is run from the directory with the contents `README.md`, `pyproject.toml`, and `/opentof`! After OpenTof is installed in your current Python environment you do not need to keep working within this directory (in fact its better not to!).

Whether relying on OpenTof to install necessary code dependancies or installing them yourself, the following packages are needed when running OpenTof:
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- dask
- chemparse
- customtkinter
- pyarrow
- minisom

If you plan to use Jupyter notebooks also install:
- jupyter
- ipykernel
- jupyterlab
- notebook
- ipywidgets

After meeting these requirements OpenTof should be installed locally within the specified Python environment! Additionally, because the package was installed in "editable" mode, any changes you make to the source code should be reflected in the code upon next import. I also usually force refresh OpenTof upon import just to double check new changes are reflected in the most recent runtime.

```python
import importlib
import opentof as ot
importlib.reload(ot)  # force reloads OpenTof
```