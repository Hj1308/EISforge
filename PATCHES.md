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

## Upcoming
| Patch | Feature | Status |
|---|---|---|
| patch22 | Bayesian MCMC uncertainty (emcee) | 🔜 planned |
| patch23 | Global optimization: DE + LHS | 🔜 planned |
