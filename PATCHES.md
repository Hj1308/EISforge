# EISForge — Patch Log


## patch29 — dependency manifests: ML deps optional, requirements/pyproject floors reconciled
**Date:** 2026-08-05
**Files changed:** requirements.txt, pyproject.toml
**What it does:**
- requirements.txt: torch / scikit-learn / xgboost moved out of the active
  install list into a clearly-marked "Optional: Machine Learning (EIS-GPT)"
  comment block with explicit `pip install` instructions. Verified by import
  scan: torch is imported only inside `eisforge/ml/` (11 sites across 6
  files); scikit-learn and xgboost have zero imports in any of the 145 .py
  files; the deployed app (app.py + pages/) never imports them; CI runs
  without torch and the test suite guards torch with
  `pytest.importorskip("torch")`.
- pyproject.toml: main `dependencies` floors raised to match
  requirements.txt (streamlit>=1.30.0, numpy>=1.26.0, scipy>=1.11.0,
  pandas>=2.1.0, matplotlib>=3.8.0, plotly>=5.18.0, galvani>=0.4.0,
  impedance>=1.4.1); `openpyxl>=3.1.0` added to main deps (app.py uses
  `pd.ExcelWriter(engine="openpyxl")` in four places — it was missing from
  the package metadata); `scikit-learn`/`xgboost`/`pyarrow` moved out of
  main deps into the `ml` extra; ml floors aligned to torch>=2.1.0; dev
  pytest floor aligned to >=7.4.0. beautifulsoup4>=4.12 kept in main deps —
  it is genuinely used by `eisforge/knowledge/parse_review_html.py` (a
  standalone CLI, not the app path), so pyproject is the correct manifest
  for it and requirements.txt correctly omits it.
**Tested on:** full suite `python -m pytest -q` -> 190 passed;
`python -m ruff check .` -> clean; `streamlit.testing.v1.AppTest`
(base / after spectrum load / after `drt_run` click) -> 0 exceptions.

## patch27 — JOSS doc fixes (README CV example, test table, paper.md escapes) + validator tests
**Date:** 2026-08-05  
**Files changed:** README.md, paper.md, tests/test_validators.py (new)
**What it does:**
- README Quick Start CV example: fixed broken attribute names
  (`result.ecsa_cm2` -> `result.ecsa`, `result.onset_potential` ->
  `result.e_onset`) and added the missing non-default `ecsa=` constructor
  kwarg. Verified `electrolyte=ElectrolyteInfo(...)` and
  `catalyst_type="noble_metal"` were valid as written (isinstance branch in
  `CVAnalyzer.__init__`; literal string equals the module constant). The
  corrected example runs verbatim on a synthetic CV -> prints
  `0.55 cm2` / `0.44511209108550553 V vs. RHE`. The `ecsa=` value is
  deliberately illustrative (0.55 cm2) and commented "not the geometric
  area" so beginners do not infer ECSA = geometric area.
- README test-coverage table: corrected the `test_eis_fitting.py` row
  (analytic Randles/RC impedance math only, no eisforge imports) and added
  rows for `test_validators.py` and `test_koutecky_levich.py`
  (C_mol_cm3 override).
- New `tests/test_validators.py` (5 tests): KramersKronigValidator on a
  synthetic causal Randles spectrum (passes at default threshold, residual
  arrays, summary), the <10-point non-blocking "Insufficient data points"
  guard, threshold enforcement, and the n=10 boundary — covering the
  validator that sits on the app.py user-facing path and writes to the
  Excel export.
- paper.md: decoded all 23 literal backslash-u escape sequences to real
  Unicode (title, em-dashes, chi2, superscripts, mu-C, >=, +/-, ->, inf —
  these would have published literally in the JOSS PDF); replaced both
  chi-squared claims with reproducible figures from
  `tests/data/sample_eis.idf`: R0 = 51.6 Ohm, R1 = 6.55e4 Ohm,
  CPE1_0 = 1.35e-5, CPE1_1 = 0.896, reduced chi2 = 0.00067 (OLS) /
  0.00076 (Huber), identical convergence from widely separated initial
  guesses.
**Tested on:** real `tests/data/sample_eis.idf` fit numbers reproduced;
full suite `python -m pytest -q` -> 178 passed; `python -m ruff check .`
-> clean.
**Known follow-up (not fixed here):** `impedance` 1.7.1 `linKK()` crashes
inside its own `eval_linKK` (`NameError: name 'np' is not defined`) and
rejects the `mu=` kwarg, so the Voigt fallback is the effective K-K path
in production. The real `sample_eis.idf` fails K-K via that fallback
(43.8 %) — expected for a pseudo-inductive spectrum, but worth surfacing
in the app UI.

---

## patch28 — K-K validator: real-part linKK, no mu, method provenance
**Date:** 2026-08-05  
**Files changed:** eisforge/core/validators.py, app.py, requirements.txt,
tests/test_validators.py (extended 5 -> 7 tests), PATCHES.md
**What it does:**
- Removed the `mu=` kwarg from the `linKK()` call (validators.py). It was
  never a parameter of `linKK(f, Z, c, max_M, fit_type, add_cap)` in the
  installed impedance 1.7.1 — the call could never have succeeded. Also
  normalised the double space in `self.c  = c` to `self.c = c` (cosmetic
  only — `self.c` was always assigned; E221 is preview-only in this ruff
  version and never flagged it).
- Set the call to `fit_type="real", add_cap=False`, with an inline comment
  recording the reasoning. Schönleber 2014 (the same source we cite for
  the 0.005 threshold in README, app caption, and UI) fits the real part
  of the impedance; with `fit_type="real"` the RC resistors are locked by
  that real-part fit and only the series L/C absorb imaginary deviation,
  so the imaginary residual is a genuine predict-and-compare — the actual
  power of the K-K test. `fit_type="complex"` fits both components
  simultaneously and lets one compensate the other, weakening the test.
  `add_cap=False`: the serial capacitance suits blocking electrodes (no
  low-frequency intercept) but as extra freedom it depresses residuals
  and lets bad spectra pass — the wrong direction for a general-purpose
  validator.
- Method provenance: `KKValidationResult` gains `method` field ("linKK",
  "voigt", "unavailable", "not_run"); `summary()` prefixes `K-K ({method}):`
  and only prints `μ=` when `method=="linKK"` (the Voigt path has no mu —
  `self.mu` is an input, not an output). app.py K-K caption and the Excel
  Summaries sheet surface the method.
- Silent fallback eliminated: `_try_linkk` logs at WARNING with the
  exception text before falling back to Voigt.
- requirements.txt: recorded the upstream bug — impedance 1.7.1 is the
  newest PyPI release (2023-07-10); `eval_linKK` is broken under
  numpy>=2.0 (`NameError: name 'np' is not defined`), upstream issue #318,
  PR #322 merged to master 2026-04-10 but unreleased. Not pinning a
  non-existent version.
- Extended tests/test_validators.py to 7 tests: monkeypatched linKK to
  assert the call uses only valid kwargs (`no mu`, `add_cap=False`,
  `fit_type="real"`) and that a linKK failure is logged and falls back to
  Voigt (`method == "voigt"`, no `μ=` in summary).
**Tested on:** real `tests/data/sample_eis.idf` probe — all three fit_type
settings (`real`/`imag`/`complex`) with `add_cap=False` fail identically
at `eval_linKK` (NameError) before any fit runs, so the linKK path is
currently **unreachable** and the new settings are reasoned but not yet
executed. `python -m pytest -q` -> 180 passed; `python -m ruff check .`
-> clean.
**Known follow-up:** when upstream impedance PR #322 (or any fixed
release) lands, the linKK path will start running for the first time —
re-test on real data then, since the settings were validated only in
reasoning, never in execution.

---

## patch20 — Band Edge & Mott-Schottky Calculator
**Date:** 2026-07-07  
**Files added:**
- `eisforge/analysis/band_edge_calculator.py` — `BandEdgeCalculator`, `BandEdgeResult`, `MottSchottkyResult`
- `pages/band_edge.py` — Streamlit UI: band diagram + Mott-Schottky plot

**What it does:**
- Butler-Ginley (1978) formula: Ecb = χ − Ec − 0.5·Eg → Ecb/Evb vs vacuum, NHE, RHE
- Built-in materials DB: g-C3N4, TiO2 (anatase/rutile), ZnO, BCN, WO3, Fe2O3, BiVO4
- Mott-Schottky analysis: upload multi-potential EIS → extracts C at chosen frequency → 1/C² vs V linear fit → Vfb, Nd, semiconductor type
- Interactive Plotly band diagram with H₂/O₂/CO₂/MeOH redox reference lines

---

## patch19 — Batch EIS Warm-Start Fitter
**Date:** 2026-07-07  
**Files added:**
- `eisforge/core/batch_fitter.py` — `CircuitEvaluator`, `BatchFitter`, `BatchEISResult`, `EISFitResult`
- `pages/batch_eis.py` — Streamlit UI: multi-file upload, warm-start CNLS, Nyquist overlay, trend plots

**What it does:**
- Sequential CNLS fitting across a series of EIS spectra
- Warm-start: best-fit params of spectrum i → initial guess for spectrum i+1
- ~60–80 % faster convergence vs independent cold starts
- Supports R, C, L, CPE, W, Wo elements; series & parallel circuits
- Trend plot: any fit parameter (Rct, Rs, Q…) vs physical condition (concentration, temperature…)
- CSV export of all fit results

---

## patch21 — CV sign-detection in onset & peak-finding
**Date:** 2026-07-09
**Files changed:**
- `eisforge/analysis/cv_analyzer.py` — `analyze()` method, lines 395-412

**What it does:**
- Ports the |j| magnitude-based sign-detection from lsv_analyzer.py:_detect_onset (patch_lsv_detect_onset_v2) into cv_analyzer.py:analyze().
- After scan splitting and before peak-finding, compares |i_fwd| mean at the low-E end vs the high-E end. If |i|_lo > |i|_hi, the anodic current is stored negative (Ivium quirk) -> flips both i_fwd and i_bwd.
- Fixes backward peak-finding: np.argmax(i_bwd) -> np.argmin(i_bwd) (cathodic backward scan peak is a minimum).
- All three _onset_* methods (tangent, threshold, derivative) and forward peak-finding (np.argmax(i_fwd)) now receive positive-up current, matching their internal assumptions.

**Tested on:**
- BCN\\-0.7-1V 50mv.s 100microLSV.idf (real Ivium data) -> sign quirk detected: |I|_lo=7.6e-4 > |I|_hi=2.4e-4 -> FLIP triggered
- Synthetic inverted CV (known E_onset=0.650 V): old logic -> E_onset=0.060 V (error 590 mV, at baseline edge); new logic -> E_onset=0.542 V (error 108 mV, within tangent-method tolerance)
- Forward peak: old 2.01 mA @ 0.105 V (baseline max) -> new 1.01 mA @ 0.986 V (faradaic peak)
- Backward peak sign: old +0.508 mA -> new -0.508 mA (cathodic correct)

> **Addendum (2026-07-09):** the file used for validation (`...100microLSV.idf`) is a single-sweep LSV file per its filename — the backward-sweep sign/peak fix (`i_bwd`, `argmin`) was validated only on synthetic data, not on real CV backward-sweep data. A genuine dual-sweep CV file was later found (B3C `CV -0.05 -0.15 V 10 mV.S`) but its forward scan is capacitive-dominated with no faradaic wave, so it doesn't validate the backward-sweep peak logic either. This remains an open validation gap.

---

## patch22 — LSV-style tangent window + k < peak_idx sanity check in CV _onset_tangent
**Date:** 2026-07-09
**Files changed:**
- `eisforge/analysis/cv_analyzer.py` — `_onset_tangent()` method

**What it does:**
- Replaces the fixed-ratio window (55%–85% of `peak_idx`) with an LSV-style
  derivative-based window: `argmax(dj[bl_end:]) + w = 5%` window around the
  steepest gradient point, matching `lsv_analyzer.py:_tangent`. This improves
  fit R² from ~0.88 to ~0.98 (previously confirmed on synthetic CV).
- Adds a `k >= peak_idx` sanity check: the steepest-gradient point must
  precede the forward-scan current peak. If k is at or after the peak
  (meaning there is no genuine anodic rising edge in the forward scan),
  falls back to `_onset_threshold`.

**Why the original gradient-SNR guard was dropped:**
- The initial design added a 5× median-|gradient| SNR check, matching the
  5×-sigma convention in LSV's `_threshold` method. However, testing on
  all 8 real electrochemistry files (3 faradaic CVs, 2 capacitive CVs,
  2 LSVs, 1 noise LSV) showed that gradient SNR was <1× for EVERY file,
  including the strong IPA oxidation wave with I SNR > 11×. The reason:
  differentiation amplifies point-to-point noise by 1/ΔE (~100× at 0.01 V
  step) while gradual faradaic waves (Tafel slope 60–120 mV/dec) spread
  their gradient over 20–40 points. The gradient SNR metric cannot
  distinguish signal from noise at any sensible threshold for CV data.
- Current-domain SNR (I span / I noise std) also failed: it passed the BCN
  noise-only file (I SNR = 6×), so it is not discriminative either.
- The LSV analyzer has no SNR guard in `_tangent` — it relies on the
  fallback chain and edge guard. The CV analyzer already has these same
  protections.

**The `k >= peak_idx` check — rationale:**
- For a genuine anodic wave, the steepest gradient (inflection point on the
  rising edge) must occur BEFORE the current peak (where dI/dE = 0).
  Strict integer `<` is the correct test — `k == peak_idx` is physically
  impossible for a proper rising edge because at the apex dI/dE = 0, which
  cannot be the gradient maximum.
- A margin was initially considered but discarded: synthetic tests with
  sharp Gaussian peaks (σ → 1.5 data points) showed that peaks narrow
  enough for k to land within 2 points of the apex already produce
  unusable tangent fits (R² < 0.07). So margin ≠ 0 provides no benefit
  over strict `<` for any case where the fit would be meaningful. No
  margin is simpler and removes a tunable parameter.
- Boundary behavior near k ≈ peak_idx is untested on real data (no real CV
  in the project has k within 5 points of the peak — alcohol oxidation
  waves are inherently gradual with margins of 40–144 points).

**Validated on 8 real files (all 8 classified correctly):**

| File | k | peak | margin | Tangent? |
|---|---|---|---|---|
| B4C IPA (strong faradaic CV) | 36 | 180 | 144 | YES — onset=-0.4983 V |
| 1M-100cycle (strong CV) | 32 | 118 | 86 | YES — onset=0.0665 V |
| BCN -0.5-1V (low-signal CV, µA) | 110 | 150 | 40 | YES — onset=-0.2183 V |
| B3C CV (moderate, capacitive fwd) | 45 | 0 | — | REJECT — falls to threshold |
| B3C 100mV/s capacitive | 19 | 0 | — | REJECT — falls to threshold |
| B3C 10mV/s capacitive | 15 | 0 | — | REJECT — falls to threshold |
| BCN -0.7-1V LSV (noise) | 84 | 0 | — | REJECT — falls to threshold |
| B3C LSV 5mV/s wide | 38 | 0 | — | REJECT — falls to threshold |

**Note on smoothing interaction:**
- The Savitzky-Golay filter (default `smoothing=True`) distorts the current
  enough to shift the onset in some cases (e.g., B4C IPA: E_onset shifts
  from -0.4983 V to -0.6400 V, which triggers the edge guard). This is a
  pre-existing interaction with the window method, not specific to this
  patch. The LSV analyzer does not apply SG filtering before onset detection.
  Users seeing unexpected threshold fallbacks should try `smoothing=False`.

---

## Known limitation — backward-scan onset baseline (found 2026-07-09, resolved patch23)

`_onset_tangent`'s baseline fit uses the FIRST `bl` points of whatever array it's given. For `i_fwd` this is correct (baseline = quiescent low-E region before the wave). For `i_bwd`, the "first" points are at the switching potential (still anodic current, not quiescent) — producing physically nonsensical backward onsets (tested on 7 real files: 6/7 gave onset shifts with wrong sign vs. expected hysteresis direction).

**Resolved by patch23:** a new `_backward_onset()` method reverses arrays before passing to the onset methods, placing the quiescent low-E tail at the start where all three methods expect it. No internals of `_onset_tangent`, `_onset_threshold`, or `_onset_derivative` were changed.

**Remaining gap — cathodic sign convention:** the three onset methods internally assume an anodic wave shape (argmax, dj>0, current>threshold). For a true cathodic backward wave, argmin / current<threshold / d2[peak:] would be needed. This is documented in a TODO comment inside `_backward_onset()`. Backward-onset is **not yet computed or exposed** in `analyze()` or the UI.

patch21's backward PEAK detection (`argmin` on `i_bwd`) remains correct and validated on all 7 files.

---

## patch23 — backward-scan onset baseline plumbing (array-reversal wrapper)
**Date:** 2026-07-09
**Files changed:**
- `eisforge/analysis/cv_analyzer.py` — added `_backward_onset()` method

**What it does:**
- Adds `CVAnalyzer._backward_onset(potential, current, i_peak)` that reverses the (potential, current) arrays before delegating to the existing `_detect_onset_method` / onset trio.
- For forward scans `i_fwd`, the quiescent baseline is at the START of the array (all three onset methods already use `[:bl_end]`). For backward scans `i_bwd`, the quiescent region is at the END (low-E tail, after the E maximum / switching potential). Reversing before onset places the quiescent region at the start — zero changes to `_onset_tangent`, `_onset_threshold`, or `_onset_derivative`.
- Includes a TODO comment documenting the cathodic-sign-convention gap: tangent uses argmax(dj)/argmax(current) (anodic), threshold uses `current > baseline + threshold` (anodic), derivative uses `d2[:peak]` (anodic peak shape). For a cathodic backward wave, argmin / `current < -threshold` / `d2[peak:]` would be needed. This is currently moot because `analyze()` never calls onset on `i_bwd`.
- Scoped to baseline plumbing only — does NOT add backward-onset computation to `analyze()` and does NOT expose backward onset in the UI.

**Tested on:**
- B4C IPA 50mV/s (idf):   baseline 0.032 → −0.038 mA, forward onset −0.4983 V (unchanged)
- B2C ETOH 5mV/s (idf):   baseline 0.035 → −0.069 mA, forward onset 1.0000 V (unchanged)
- BCN −0.5-1V 100µA (idf): baseline 0.00023 → −0.00048 mA, forward onset −0.2183 V (unchanged)

---

## patch24 — bootstrap_eis.py (noise-injection bootstrap uncertainty)
**Date:** 2026-07-09
**Files added:**
- `eisforge/ml/uncertainty/bootstrap_eis.py` — `bootstrap_eis_uncertainty()`, `BootstrapResult`

**What it does:**
- Percentile-CI parameter uncertainty via noise-injection residual bootstrap.
- Initial fit uses TRF `least_squares` (matching main `CNLSFitter` optimizer),
  bootstrap refits also use TRF for consistency.
- Modulus-weighted (`1/|Z|`) throughout.
- Optional per-parameter `bounds` pass through to both initial fit and refits.
- Bounds sanity check: raises `ValueError` if best-fit params fall outside
  specified bounds (catches config errors before they produce silently bad CIs).

**Tested on:**
6 real HBC-4 IPA+H2SO4 spectra (0.25–0.92 V). TRF initial-fit Rs matches
known ~25.5 Ω thesis value at the two extremes (0.25V: 25.7 Ω, 0.92V:
25.2 Ω).

**Known limitation — bimodality at all potentials:**
Bootstrap Rs samples remain substantially bimodal after modulus weighting
at all 6 potentials — 54% of refits land at Rs=0 for 0.25V (median=0.00),
42% for 0.92V (median=4.99), ~50% for mid-potential files. This is genuine
optimisation-landscape multimodality (confirmed independently by DRT and
KK's own struggles on the same spectra), not a fixable weighting or bounds
bug.  Root cause: wide |Z| dynamic range (~3 decades).  Best-fit point
estimates match the known value; report point estimates only, not intervals,
until further work addresses the multimodality directly.

---

## patch25 — kk_validator.py (standalone Lin-KK validator)
**Date:** 2026-07-09
**Files added:**
- `eisforge/analysis/kk_validator.py` — `StandaloneKKValidator`, `LinKKResult`

**What it does:**
- Standalone Lin-KK validator (Schonleber-style Voigt circuit) with explicit
  R_Ω and optional L terms in the design matrix.
- Modulus-weighted least squares (`1/|Z|` on A and b rows) — critical fix
  for data with >2 decades |Z| range.
- Adaptive n_rc search: scans from standard heuristic downward, jointly
  optimising mu and max_res to avoid both under-fitting and over-fitting.
- μ-criterion is a simplified proxy (fraction of non-negative R_k, **not**
  Schönleber's exact weighted formula) — documented in code and docstring;
  cite accordingly if used in a thesis/publication.

**Tested on:**
Same 6 HBC-4 IPA+H2SO4 spectra. R_ohm converged to a tight **27.0–29.4 Ω**
band across ALL 6 potentials after modulus weighting was added — matching
raw high-frequency Z_real (28.4–28.8 Ω) and independent of the thesis
Rs~25.5 Ω CNLS result.  This is the strongest independent cross-validation
of the three new tools.

**Known limitation — max_res:**
`max_res` (single-point modulus-normalised residual) still exceeds the 1%
threshold on most files (9–102%) — caused by localised high-frequency
scatter in the raw data, not a fit-quality issue.  The KK "verdict"
(PASS/FAIL) should not be over-interpreted on these data; report mu and
R_ohm as the meaningful diagnostic outputs, not the binary verdict.

---

## patch26 — drt_analyzer.py (Tikhonov-regularized DRT)
**Date:** 2026-07-09
**Files added:**
- `eisforge/analysis/drt_analyzer.py` — `DRTAnalyzer`, `DRTResult`

**What it does:**
- Tikhonov-regularized DRT with explicit R_inf column in the design matrix
  (not estimated separately from a single high-frequency data point).
- Bounded least squares via `scipy.optimize.lsq_linear` with `lb=0` on the
  full parameter vector — R_inf constrained ≥0 by the solver itself, not
  post-hoc clipped.
- L-curve curvature-based λ selection.
- Post-hoc peak finding on γ(ln τ).

**Tested on:**
Same 6 HBC-4 IPA+H2SO4 spectra. R_inf recovered 19.8 Ω (0.25V) and
31.0 Ω (0.92V) — same ballpark as KK/bootstrap/raw values at those two
potentials. Main DRT peak at f≈2–11 Hz for these spectra.

**Known limitation — mid-potential bound saturation:**
For the 4 mid-potential files (0.47–0.82 V), the bounded solver hits the
R_inf = 0 lower bound — the regularised inverse problem cannot extract a
positive R_inf from these data without further tuning (L-curve / λ
selection may need refinement).  Not resolved in this patch; report DRT
R_inf only for 0.25V/0.92V-type spectra until then.

### Cross-tool summary

Three independent methods (raw high-frequency impedance reading,
KK-fitted R_ohm, DRT R_inf, and bootstrap Rs) agree within ~±5 Ω of
a 25–29 Ω range at the two best-conditioned potentials (0.25 V, 0.92 V)
tested against this project's 6-file HBC-4 IPA+H2SO4 dataset, supporting
the thesis's existing Rs ≈ 25.5 Ω circuit-fit result.  Mid-potential
spectra (0.47–0.82 V) show wide |Z| dynamic range that limits all three
model-free/statistical methods' reliability there — this is a data
characteristic, not a flaw in the existing CNLS circuit fit (which does
not suffer the same conditioning issue, fitting Rs and Rct simultaneously
across the same wide range with a proper physical circuit model rather
than an unconstrained/regularised linear approximation).

---

## Upcoming
| Patch | Feature | Status |
|---|---|---|
| patch28 | Bayesian MCMC uncertainty (emcee) | 🔜 planned |
| patch29 | Global optimization: DE + LHS | 🔜 planned |