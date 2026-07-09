# EISForge — Patch Log

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

## Upcoming
| Patch | Feature | Status |
|---|---|---|
| patch23 | Bayesian MCMC uncertainty (emcee) | 🔜 planned |
| patch24 | Global optimization: DE + LHS | 🔜 planned |
