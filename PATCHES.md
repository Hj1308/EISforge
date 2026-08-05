# Patch Log

## patch01 — README/paper JOSS doc fixes, validator test coverage
**Date:** 2026-08-05
**Files changed:** README.md, paper.md, tests/test_validators.py (new), PATCHES.md (new)
**What it does:**
- README Quick Start CV example: fixed broken attribute names
  (`result.ecsa_cm2` -> `result.ecsa`, `result.onset_potential` ->
  `result.e_onset`) and added the missing non-default `ecsa=` constructor
  kwarg. Verified `electrolyte=ElectrolyteInfo(...)` and
  `catalyst_type="noble_metal"` are valid as written (isinstance branch in
  `CVAnalyzer.__init__`; literal string equals the module constant). The
  corrected example runs verbatim on a synthetic CV and prints
  `0.1225 cm²` / `0.44511209108550553 V vs. RHE`.
- README test-coverage table: corrected the `test_eis_fitting.py` row
  (analytic Randles/RC impedance math only, no eisforge imports) and added
  rows for the previously unlisted `test_validators.py` and
  `test_koutecky_levich.py` (C_mol_cm3 override).
- New `tests/test_validators.py` (5 tests): KramersKronigValidator on a
  synthetic causal Randles spectrum (passes at default threshold, residual
  arrays, summary), the <10-point non-blocking "Insufficient data points"
  guard, threshold enforcement, and the n=10 boundary. Covers the
  validator that is on the app.py user-facing path and writes to the Excel
  export.
- paper.md: decoded all 23 literal `\uXXXX` escape sequences to real
  Unicode (the title, em-dashes, χ², ⁻², µC, ⁻¹, ≥, ±, →, ∞ would have
  published literally); replaced both chi-squared claims with the actual
  reproducible figures from the shipped `tests/data/sample_eis.idf`:
  R0 = 51.6 Ω, R1 = 6.55×10⁴ Ω, CPE1_0 = 1.35×10⁻⁵, CPE1_1 = 0.896,
  reduced χ² = 0.00067 (OLS) / 0.00076 (Huber), converging identically
  from widely separated initial guesses.
**Tested on:**
- Synthetic clean Randles spectrum + real `tests/data/sample_eis.idf`
  (fit numbers reproduced by `repro_kk_fit.py`).
- Full suite: `python -m pytest -q` -> 178 passed (173 prior + 5 new);
  `python -m ruff check .` -> all checks passed.
**Known follow-ups (not fixed here):** `impedance` 1.7.1 `linKK()` crashes
inside its own `eval_linKK` (`NameError: name 'np' is not defined`) and
rejects the `mu=` kwarg, so the Voigt fallback is the effective K-K path
in production. The real `sample_eis.idf` fails K-K via that fallback
(43.8% residual) — expected for a pseudo-inductive spectrum, but worth
documenting in the app UI.
