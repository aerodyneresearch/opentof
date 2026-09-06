# OpenTof

A Python library for end-to-end processing of Time-of-Flight Mass Spectrometry (ToF-MS) data from Tofwerk instruments (`.h5` files).

## Installation

Install OpenTof and all its dependencies directly via `pip`:

```bash
pip install opentof
```

## Installation (Local / Current Developers)
If you previously used the old script setup, your environment is still completely valid. However, this procedure installs opentof as an editable local package. This allows you to edit the source code and see your changes instantly without rebuilding anything.

### Step 0: Clone the Repository
```bash
git clone https://github.com/aerodyneresearch/opentof.git 
cd opentof
```

### Step 1: Create and Activate your environment
Open PowerShell in your repository root directory and run:

```bash
python -m venv .venv
```
and
```bash
.\.venv\Scripts\activate
```

### Step 2: Clean up old installation (Existing Developers Only)
If your environment has old package metadata floating around, clean it out first:
```bash
pip uninstall opentof
```

### Step 3: Install in Editable Dev Mode
From the repository root directory (where pyproject.toml lives), run:
```bash
pip install -e .
```

(Note: The -e flag stands for "editable". The . tells pip to look at the current directory's pyproject.toml. Pip will verify all dependencies and link your environment directly to your live opentof/ folder).
