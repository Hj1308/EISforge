# Contributing to EISForge

Thank you for your interest in EISForge! Contributions of all kinds are welcome —
bug reports, feature requests, documentation improvements, and code.

This project is maintained by Hoda Jafari. Please be respectful and constructive in
all interactions.

---

## Reporting a bug

If you find a bug, please open an issue on GitHub:

**https://github.com/Hj1308/EISforge/issues**

To help us fix it quickly, please include:

1. **What you did** — the steps that led to the problem (e.g. "loaded a Gamry `.DTA`
   file in the EIS tab and clicked Fit").
2. **What you expected to happen.**
3. **What actually happened** — the full error message if there was one (copy the whole
   text, not a screenshot of part of it).
4. **Your setup** — operating system (Windows / macOS / Linux), Python version
   (`python --version`), and EISForge version (v0.3.0).
5. **A sample file** if possible — a small data file that reproduces the problem makes
   it far easier to diagnose. Please only share data you are free to distribute.

---

## Requesting a feature

Have an idea for a new analysis, parser, or improvement? Please open an issue and label
it (or just mention in the title) as a **feature request**. Describe:

- What you would like EISForge to do.
- The scientific use case — what electrochemical question it would help answer.
- If relevant, a reference to the method or paper it is based on.

---

## Asking a question

If you are unsure how to use a feature, or have a question about the analysis methods,
you can also open an issue. There are no silly questions — if something is unclear,
others probably find it unclear too, and your question helps us improve the docs.

---

## Setting up a development environment

If you would like to contribute code, here is how to get set up.

**1. Clone the repository:**

```
git clone https://github.com/Hj1308/EISforge.git
cd EISforge
```

**2. (Recommended) create and activate a virtual environment:**

```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

**3. Install EISForge with development tools:**

```
pip install -e ".[dev]"
```

This installs the package in "editable" mode (so your changes take effect immediately)
along with the tools needed for testing and linting.

---

## Running the tests

Before submitting any code change, please make sure all tests pass:

```
pytest
```

If `pytest` is not found, try:

```
python -m pytest
```

All tests should pass before you open a pull request. If you add a new feature, please
add a test for it in the `tests/` directory.

---

## Submitting a code change (pull request)

1. **Fork** the repository on GitHub (click the "Fork" button).
2. **Create a branch** for your change:
   ```
   git checkout -b my-feature
   ```
3. **Make your change**, keeping it focused — one topic per pull request is easiest to
   review.
4. **Run the tests** (`pytest`) and make sure they pass.
5. **Commit** with a short, clear message describing what changed.
6. **Push** your branch to your fork and open a **pull request** against the `main`
   branch of this repository.

In the pull request description, please explain what your change does and why. If it
fixes an open issue, mention the issue number.

---

## Scientific correctness

EISForge is a research tool, so scientific accuracy matters as much as working code.
If your change affects any calculation (fitting, Tafel slopes, ECSA, band-edge
positions, unit conversions, reference potentials, etc.), please:

- Explain the equation or method you used, ideally with a literature reference.
- Keep physically distinct constants separate (for example, the vacuum reference and the
  NHE conversion are two different numbers and must not be merged).
- Add or update a test that checks the numerical result.

---

## License

By contributing to EISForge, you agree that your contributions will be licensed under
the same [MIT License](LICENSE) that covers the project.

---

## Citation

If you use EISForge in your research, citation is requested — see the
[Citation and Attribution](README.md#-citation-and-attribution) section of the README.

Thank you for helping make EISForge better!
