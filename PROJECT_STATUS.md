# EISForge — Project Status & Roadmap

**Author:** Hoda Jafari
**GitHub:** https://github.com/Hj1308/EISforge
**Live demo:** https://eisforge-app.streamlit.app/
**Current version:** v0.3.0
**License:** MIT
**Last Updated:** July 2026

---

## Project Vision

EISForge is an open-source, catalyst-aware electrochemistry analysis toolkit for the
Alcohol Oxidation Reaction (AOR). It combines:

- Classical EIS analysis (CNLS fitting, Kramers–Kronig validation, robust Huber-IRLS fitting)
- CV and LSV analysis with automatic, catalyst-aware interpretation
- Multi-model equivalent-circuit ranking (AICc)
- Automated ECSA calculation (H-UPD, CO stripping, Cdl)
- Scan-rate kinetics and chronoamperometry stability analysis
- A literature-guided knowledge layer
- A physics-informed ML module (EIS-GPT) — architecture complete, weights in development
- A modern Streamlit web interface

**Target users:** electrochemistry researchers working on AOR, batteries, fuel cells,
semiconductor photocatalysts, and corrosion.

---

## Release Status

| Item | Status |
| --- | --- |
| Version | v0.3.0 (released Jul 2026) |
| Live web demo (Streamlit Cloud) | ✅ eisforge-app.streamlit.app |
| Zenodo concept DOI (all versions) | ✅ 10.5281/zenodo.20649692 |
| Zenodo version DOI (v0.3.0) | ✅ 10.5281/zenodo.21209400 |
| CI (GitHub Actions, Python 3.10 + 3.11) | ✅ passing |
| Test suite | ✅ 164 tests passing |
| JOSS manuscript (paper.md) | 🔬 drafted, not yet submitted |

---

## Completed Work

### 1. Project infrastructure
- [x] GitHub repository configured, MIT License, README with citation info
- [x] `requirements.txt`, `pyproject.toml`, `setup.py`
- [x] `eisforge/` package structure, `.gitignore`
- [x] CI/CD pipeline (GitHub Actions — lint + Python 3.10 + 3.11)
- [x] Zenodo DOI registration (concept + version)
- [x] `CITATION.cff` for automatic citation parsing
- [x] Repository cleanup — one-off dev patch scripts removed from tracking

### 2. Data parsers (`eisforge/parsers/`)
- [x] `base_parser.py` — abstract base + dataset container
- [x] `gamry_parser.py` — Gamry `.DTA`
- [x] `generic_csv_parser.py` — CSV/TSV/TXT with auto-detection
- [x] `ivium_parser.py` — Ivium `.idf` (CV + EIS)  ⚠️ *renamed from `autolab_parser.py`; `.idf` is an Ivium format, not Autolab*
  - Auto-detects CV vs EIS from method header
  - Handles latin-1 / cp1252 / utf-8 encodings
  - Converts current A → mA automatically
  - Multi-scan support
  - `AutolabIDFParser` kept as a deprecated back-compat alias
- [x] `biologic_parser.py` — BioLogic `.mpt` / `.mpr` (via galvani)

### 3. Core engine (`eisforge/core/`)
- [x] `analyzer.py` — orchestration
- [x] `fitter.py` — CNLS fitter (bounds, modulus weighting, robust Huber-IRLS re-weighting)
- [x] `preprocessor.py`
- [x] `validators.py` — Kramers–Kronig (impedance.py linKK + Voigt fallback)

### 4. Analysis modules (`eisforge/analysis/`)
- [x] `cv_analyzer.py` — E_onset, peak position/height, ECSA-normalized current density, AOR interpretation
- [x] `lsv_analyzer.py` — Tafel slope, j₀, overpotentials, mass/specific activity, E_half
- [x] `eis_cv_correlator.py` — cross-technique correlation
- [x] `ecsa_calculator.py` — H-UPD / CO stripping / Cdl methods
- [x] `koutecky_levich.py` — electron-transfer number
- [x] `batch_analyzer.py` — mean ± SD over n ≥ 3, reproducibility scoring
- [x] Scan-rate kinetics (b-value, Randles–Ševčík linearity, mechanism assignment)
- [x] Chronoamperometry (i–t retention %, steady-state current, initial drop)
- [x] `band_edge_calculator.py` — E_cb / E_vb for semiconductor catalysts
  - Vacuum reference (`EC_REF_VAC = 4.50 eV`) and NHE conversion (`E_NHE_OFFSET = 4.44 eV`, Trasatti/IUPAC) are kept as two distinct constants, matching the empirical band-edge formula and the vs-NHE conversion respectively.

### 5. EIS interpretation & suggestion engine
- [x] Rule-based physical interpretation (Rs, per-arc effective capacitance via Brug, time constants, NDR / pseudo-inductive / Warburg fingerprints)
- [x] AICc multi-model circuit suggestion (Burnham & Anderson convention)
- [x] Low-frequency pseudo-inductive / NDR topologies for AOR
- [x] Excel export (multi-sheet .xlsx: Summary, Fit_Parameters, Data, Fit_Curve)
- [x] Smooth fit overlay on 400 log-spaced frequencies

### 6. EIS-GPT — physics-informed ML (`eisforge/ml/eis_gpt/`)
- [x] `tokenizer.py` — spectrum → token sequence (5D features + sinusoidal positional encoding)
- [x] `physics_loss.py` — Kramers–Kronig + passivity + HF-limit constraints in loss
- [x] `transformer.py` — 6-layer encoder, 8 heads, circuit classification + parameter regression
- [x] `aor_dataset_generator.py` — synthetic AOR spectra with realistic noise
- [ ] Pre-trained weights — **in development (v0.4)**; model currently returns untrained predictions

### 7. Knowledge base (`eisforge/knowledge/`)
- [x] Literature-guided interpretation layer (curated from peer-reviewed papers)
- [x] `literature_engine.py` — literature-driven initial parameter guessing

### 8. Web interface (`app.py`)
- [x] Streamlit UI with catalyst-family / electrolyte awareness
- [x] Tabs: CV | LSV | EIS | EIS-GPT | Correlation | ECSA | K-L | Scan-Rate | Chronoamperometry
- [x] Interactive Plotly visualizations
- [x] Multi-format upload (.idf, .dta, .mpt, .csv, .txt)
- [x] Automatic unit + RHE conversion

### 9. Tests
- [x] 164 tests passing (CI on every push)
- [x] Covers: EIS fitting & K-K, interpreter, suggestion engine, inductive/NDR circuits, scan-rate, chronoamperometry, ECSA, batch, parsers

---

## In Progress / Next Steps

### Phase B — JOSS submission requirements (current focus)
- [ ] `CONTRIBUTING.md` + issue/bug-report guidelines (JOSS community-guidelines requirement)
- [ ] API documentation (docstring coverage + a small docs site)
- [ ] Runnable examples in `examples/` with real sample data
- [ ] Finalize and submit `paper.md` to JOSS
- [ ] PyPI package release (`pip install eisforge`)

### Phase C — Known technical debt
- [ ] Validate Gamry `.DTA` and BioLogic `.mpt` parsers on **real** files (only `.idf` validated on real data so far)
- [ ] Make K-K validation fail gracefully on sparse/drifted data (currently can crash)
- [ ] Harden multi-cycle CV scan separation

### Phase D — Future features
- [ ] `litbase` — lightweight literature database engine (pandas/stdlib only, to stay within Streamlit Cloud free tier); design initiated
- [x] DRT (distribution of relaxation times) with Tikhonov regularization (EIS tab, DRT expander)
- [ ] Train EIS-GPT on synthetic data and ship pretrained weights (v0.4)
- [ ] Additional parsers (Zahner, PalmSens, CHI)

---

## Known Issues

1. **EIS-GPT not trained** — returns untrained predictions until v0.4 weights ship. Should be labelled "experimental — untrained" in the UI.
2. **K-K validation** — occasional failure on sparse or drifted data.
3. **Multi-cycle CV splitting** — may fail on CVs with many cycles.
4. **Parser validation** — Gamry and BioLogic parsers not yet tested on real files.

---

## Technical Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.10+ |
| EIS fitting | impedance.py (CNLS, K-K) |
| Numerics | NumPy, SciPy |
| Data | Pandas |
| ML | PyTorch (EIS-GPT) |
| Visualization | Plotly |
| Web UI | Streamlit |
| BioLogic | galvani |
| Testing | pytest |
| CI | GitHub Actions (Python 3.10 + 3.11) |

---

*Document status: Active development. This file is maintained by hand — update it when
major modules or releases land so it never drifts from the actual repository state.*
