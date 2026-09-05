# Contributing to OpenTof

Thanks for your interest in improving OpenTof! This project is developed and
maintained by a small team at Aerodyne Research, and we welcome bug reports,
feature suggestions, and code contributions from the community.

This guide explains how to get involved.

## Ways to contribute

- **Report a bug** — open an issue using the *Bug report* template.
- **Suggest a feature** — open an issue using the *Feature request* template.
- **Contribute code** — open a pull request (see below). Good first
  contributions include fixing a documented bug, improving documentation, or
  adding tests.

If you are planning a larger change, please open an issue to discuss it first
so we can agree on the approach before you invest time in it.

## Reporting bugs

Before opening a bug report, please search existing issues to check it hasn't
already been reported. A good report includes:

- What you did and what you expected to happen.
- What actually happened (including the full error message / traceback).
- A minimal example that reproduces the problem, if possible.
- Your environment: OpenTof version, Python version, and operating system.

## Development setup

1. Fork the repository on GitHub and clone your fork.
2. Create and activate a virtual environment.
3. Install OpenTof in editable mode with the development extras:

   ```bash
   pip install -e ".[dev]"
   ```

   This installs the package plus the tools used for testing (pytest, build).
   If you plan to work with the notebooks, also install the notebook extras:

   ```bash
   pip install -e ".[dev,notebook]"
   ```

## Making changes

1. Create a branch for your change (e.g. `fix/parser-edge-case`).
2. Make your change, keeping it focused — one logical change per pull request
   is easier to review.
3. Add or update tests where it makes sense.
4. Update documentation (docstrings, README) if behavior changes.
5. Make sure the test suite passes locally before opening the PR.

## Pull request process

1. Push your branch to your fork and open a pull request against `master`.
2. Fill out the pull request template so reviewers understand the change and
   how it was tested. Link any related issue.
3. Your PR will be reviewed by the OpenTof maintainers. At least one maintainer
   approval is required before a change can be merged. We may ask for changes
   or clarification — this is a normal part of the process.
4. Once approved and passing checks, a maintainer will merge your PR.

Please be patient: the maintainers review contributions alongside their other
work, so it may take some time to get to your PR.

## Code of conduct

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in
all interactions.

## License

OpenTof is released under the Apache License 2.0. By submitting a contribution,
you agree that your contribution is licensed under the same terms (see Section 5
of the Apache 2.0 license). Please only contribute code that you have the right
to submit.
