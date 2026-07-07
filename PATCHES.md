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

## Upcoming
| Patch | Feature | Status |
|---|---|---|
| patch21 | Bayesian MCMC uncertainty (emcee) | 🔜 planned |
| patch22 | Global optimization: DE + LHS | 🔜 planned |
