# Releasing OpenTof

Maintainer guide for publishing `opentof` to TestPyPI (rehearsal) and PyPI
(production). Commands are written for Windows / PowerShell; adjust paths to
your local checkout.

Publishing uses **PyPI Trusted Publishing** (OpenID Connect via GitHub
Actions), so no API tokens or passwords are stored in the repository.

## Key concepts (read once)

- **The version number is set manually** in the `version` field of
  `pyproject.toml` (e.g. `version = "0.1.0"`). The build stamps that exact
  string into the artifact filenames and metadata. The git tag does **not** set
  the version — it only triggers the publish. Bump `version` for every release
  and keep it in sync with the tag.
- **Published versions are immutable.** PyPI and TestPyPI reject re-uploading an
  existing version; you can only publish a new version (and "yank" a bad one).
  This is why the local smoke test below matters.
- **Two distributions per build.** `python -m build` produces an sdist
  (`opentof-<version>.tar.gz`) and a wheel
  (`opentof-<version>-py3-none-any.whl`). pip prefers the wheel.
- **Tag scheme.** Releases are triggered by distinctive, deliberate tags:
  - TestPyPI: `release_testv*` (e.g. `release_testv0.1.0`)
  - PyPI: `release_v*` (e.g. `release_v0.1.0`)
  An ordinary `vX.Y.Z` tag does **not** trigger a publish.

---

## Part A — Local smoke test (do this before every release)

Build the package as it will be published, install it into a **clean**
environment, and confirm it imports. This tests the built artifact, not your
working source tree.

### 1. Build the package (inside the repo)

```powershell
cd <path-to-your-opentof-checkout>
python -m pip install --upgrade build
python -m build
```

Success ends with:
`Successfully built opentof-<version>.tar.gz and opentof-<version>-py3-none-any.whl`

The wheel and sdist land in `dist\`. (`build\` and `opentof.egg-info\` are
build scratch and are git-ignored.)

### 2. Create a fresh virtual environment OUTSIDE the repo

Doing this from a different folder prevents Python from importing the local
`opentof\` source instead of the installed package (which would give a false
pass).

```powershell
cd $env:TEMP
python -m venv opentof-smoketest
.\opentof-smoketest\Scripts\Activate.ps1
```

Your prompt should now start with `(opentof-smoketest)`. If script execution is
blocked, run `Set-ExecutionPolicy -Scope Process -Bypass` once and retry.

### 3. Install the built wheel

```powershell
pip install "<path-to-your-opentof-checkout>\dist\opentof-<version>-py3-none-any.whl"
```

Dependencies are pulled from real PyPI — a genuine end-user install. Success
ends with `Successfully installed opentof-<version> ...`.

### 4. Confirm it imports the installed copy

```powershell
python -c "import opentof; print(opentof.__file__)"
```

Check there is no traceback and the printed path points **into the venv's
site-packages** (e.g. `...\opentof-smoketest\Lib\site-packages\opentof\__init__.py`),
not back to your source checkout.

### 5. Clean up

```powershell
deactivate
Remove-Item -Recurse -Force $env:TEMP\opentof-smoketest
```

---

## Part B — Publish to TestPyPI (rehearsal)

Prerequisites (one-time, already configured for this repo):

- A TestPyPI account with a **pending/active Trusted Publisher** for project
  `opentof`, owner `aerodyneresearch`, repo `OpenTof`, workflow
  `testrelease.yml`, environment `testpypi`.
- A GitHub environment named `testpypi` with a deployment **tag** rule of
  `release_testv*`.
- `.github/workflows/testrelease.yml` present on the default branch.

Steps:

1. Confirm the smoke test (Part A) passed and `version` in `pyproject.toml` is
   the version you intend to publish.
2. Commit any pending changes to the default branch.
3. Create and push a matching tag. Either from the command line:

   ```powershell
   git tag release_testv0.1.0
   git push origin release_testv0.1.0
   ```

   Or with **TortoiseGit** (Windows GUI): right-click the repo folder →
   *TortoiseGit → Create Tag...*, set the tag name to `release_testv0.1.0`,
   base it on `HEAD (master)`, tick the **Push** option, and click OK. (If you
   leave Push unticked, create the tag then push it separately via
   *TortoiseGit → Push...* with "Include Tags" enabled.) Creating the tag
   locally does nothing until it is pushed — the pushed tag is what triggers the
   workflow.

4. Watch the run in the repo's **Actions** tab. The `Publish to TestPyPI` job
   builds and uploads via OIDC.
5. On success, the project appears at `https://test.pypi.org/project/opentof/`
   and the pending publisher becomes active.
6. Verify the install. TestPyPI does not host your dependencies, so point pip at
   real PyPI for them:

   ```powershell
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ opentof
   ```

If something is wrong, fix it and publish a **new** version (bump `version` and
use a new tag) — you cannot overwrite an existing version.

---

## Part C — Publish to PyPI (production)

The production setup mirrors TestPyPI, with the parallel names below. Set this
up once against the repository you publish from.

1. **Register a Trusted Publisher on PyPI** (pypi.org, not test) for:
   - Project name: `opentof`
   - Owner: the GitHub owner of the public repo
   - Repository: the public repo name
   - Workflow: `release.yml`
   - Environment: `pypi`
2. **Create a `pypi` GitHub environment** with a deployment **tag** rule of
   `release_v*` (and required reviewers, if the plan supports them).
3. **Add `.github/workflows/release.yml`** — identical to `testrelease.yml`
   except:
   - `environment: pypi` (and its `url` pointing at the real project page)
   - the publish step omits `repository-url` (defaults to real PyPI)
   - trigger on `release_v*` tags
4. **Release** by bumping `version`, running the Part A smoke test, then:

   ```powershell
   git tag release_v0.1.0
   git push origin release_v0.1.0
   ```

5. Verify: `pip install opentof` from a clean environment.

---

## Checklist per release

- [ ] Bump `version` in `pyproject.toml`.
- [ ] Update `CHANGELOG` / release notes (if kept).
- [ ] Run the Part A smoke test — build + clean-venv install + import.
- [ ] Commit and push to the default branch.
- [ ] Push the release tag (`release_testv*` for TestPyPI, `release_v*` for PyPI).
- [ ] Watch the Actions run to green.
- [ ] Verify the install from the target index.
