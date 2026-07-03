# EISForge — Project Context

**Author:** Hoda Jaafari  
**Repo:** https://github.com/Hj1308/EISforge  
**Stack:** Python 3.10+ | Streamlit | Plotly | impedance.py | scipy | PyTorch  
**Status:** Active Development — June 2026

---

## Architecture Overview

```
EISforge/
├── app.py                          ← Streamlit UI (7 tabs, 64KB — main entry point)
├── cv_app.py                       ← Standalone CV app
├── analyze_cv.py                   ← CV batch analysis script
├── train_eis_gpt.py                ← ML training script
├── apply_v2_fixes.py               ← Migration/patch script
│
├── eisforge/                       ← Core Python package
│   ├── core/
│   │   ├── analyzer.py             ← EisAnalyzer: orchestration class
│   │   ├── fitter.py               ← CNLSFitter: main fitting engine
│   │   ├── preprocessor.py         ← Data preprocessing
│   │   └── validators.py           ← Kramers-Kronig validator
│   │
│   ├── parsers/
│   │   ├── base_parser.py          ← Abstract base + EISDataset dataclass
│   │   ├── autolab_parser.py       ← Metrohm Autolab .idf (CV + EIS)
│   │   ├── gamry_parser.py         ← Gamry .DTA files
│   │   ├── biologic_parser.py      ← BioLogic .mpt/.mpr (via galvani)
│   │   └── generic_csv_parser.py   ← CSV/TXT auto-detection
│   │
│   ├── analysis/
│   │   ├── cv_analyzer.py          ← CV: onset, peak, ECSA, AOR
│   │   ├── lsv_analyzer.py         ← LSV: Tafel, j0, overpotential
│   │   ├── eis_cv_correlator.py    ← Cross-technique correlation
│   │   └── ecsa_calculator.py      ← ECSA: H-UPD, CO Stripping, Cdl
│   │
│   ├── ml/
│   │   ├── aor_dataset_generator.py
│   │   └── eis_gpt/
│   │       ├── tokenizer.py        ← EIS→Transformer tokens (5D features)
│   │       ├── physics_loss.py     ← K-K penalty + Passivity + HF limit
│   │       └── transformer.py      ← 6-layer encoder, classification+regression
│   │
│   ├── knowledge/
│   │   ├── literature_engine.py    ← Literature-driven initial parameter guessing
│   │   └── electrochemistry_knowledge.json
│   │
│   ├── standards/
│   └── utils/
│
└── tests/
    ├── test_ecsa.py                ← 12 tests PASSING
    ├── test_transformer.py
    ├── test_tokenizer.py
    ├── test_idf.py
    ├── test_ir.py
    └── test_all.py
```

---

## Domain Glossary — Electrochemical Terms

### EIS (Electrochemical Impedance Spectroscopy)
- **EIS**: Frequency-domain technique measuring complex impedance Z(ω) = Z' - jZ''
- **Nyquist plot**: -Im(Z) vs Re(Z) — semicircle = one RC time constant
- **Bode plot**: |Z| and phase angle (°) vs log(frequency/Hz)
- **CNLS**: Complex Non-Linear Least Squares — fitting method used in `CNLSFitter`
- **Circuit string**: impedance.py notation, e.g., `"R0-p(R1,CPE1)-Wo1"`
- **Randles circuit**: `"R0-p(R1,C1)"` — Rs + (Rct ∥ Cdl) — baseline model
- **CPE**: Constant Phase Element — `Q` (admittance, S·s^n) and `n` (0<n<1, n=1 is ideal capacitor)
- **Warburg (Wo)**: Semi-infinite diffusion, 45° line on Nyquist
- **Rs / R0**: Solution (Ohmic) resistance — intercept on Nyquist real axis
- **Rct / R1**: Charge transfer resistance — diameter of semicircle
- **K-K / linKK**: Kramers-Kronig validation — checks data quality/causality
- **Voigt circuit**: Fallback K-K model used in `validators.py`
- **chi-squared (χ²)**: Goodness of fit metric in `FitResult.chi_squared`
- **modulus weighting**: `weight = 1/|Z|` — IUPAC standard, used by default in `CNLSFitter`
- **outlier_threshold**: `3.0` σ by default in MAD-based outlier detection
- **TRF**: Trust Region Reflective — primary scipy optimizer (supports bounds)
- **LM**: Levenberg-Marquardt — fallback optimizer (no bounds, most robust)

### CV/LSV (Voltammetry)
- **CV**: Cyclic Voltammetry — potential swept anodic then cathodic
- **LSV**: Linear Sweep Voltammetry — single direction sweep
- **E_onset**: Onset potential (V vs RHE) — where current begins to rise
- **Ep / Epa / Epc**: Peak potential / anodic peak / cathodic peak
- **ip / ipa / ipc**: Peak current / anodic / cathodic
- **I_f/I_b ratio**: Forward/backward peak ratio — AOR diagnostic
- **scan rate (ν)**: mV/s — sweep speed, affects diffusion layer
- **AOR**: Alcohol Oxidation Reaction — primary target application
- **RHE**: Reversible Hydrogen Electrode — reference scale used throughout
- **j**: Current density (mA/cm² or mA/cm²_ECSA)

### ECSA (Electrochemically Active Surface Area)
- **ECSA**: Active surface area in cm² — from `ecsa_calculator.py`
- **H-UPD**: Hydrogen underpotential deposition — for Pt/Pd, q_ref = 210/212 µC/cm²
- **CO Stripping**: For PtRu/PtSn/Pd, q_ref = 420/424 µC/cm²
- **Cdl method**: Double-layer capacitance — for carbon/metal-free catalysts
- **_split_scans()**: Internal method — separates cathodic/anodic via `argmax` (CRITICAL: do NOT use `len//2`)
- **fit_intercept**: Returned by ECSA calculator for accurate Streamlit trendline

### BET / Semiconductor
- **BET surface area**: m²/g from N₂ adsorption isotherm
- **BJH**: Barrett-Joyner-Halenda — pore size distribution
- **Band edge**: Ecb = X - Ec - 0.5·Eg; Evb = Ecb + Eg (NEXT MODULE to build)
- **Mott-Schottky**: Flat-band potential analysis for semiconductors
- **BCN / g-C3N4**: Boron carbon nitride / graphitic carbon nitride — target materials

### EIS-GPT (ML Module)
- **EIS-GPT**: 6-layer Transformer encoder for circuit classification + parameter regression
- **5D features**: [log(f), Z', Z'', |Z|, θ] per frequency point
- **physics_loss**: K-K penalty + passivity constraint + HF limit penalty
- **Status**: Model NOT trained yet — returns random predictions (known issue)

---

## Key Classes and Their Roles

| Class | File | Purpose |
|-------|------|---------|
| `EISDataset` | `parsers/base_parser.py` | Dataclass: holds `frequency`, `z_complex` numpy arrays |
| `CNLSFitter` | `core/fitter.py` | CNLS fitting with TRF→LM fallback, outlier removal |
| `FitResult` | `core/fitter.py` | Dataclass: `parameters`, `parameter_errors`, `chi_squared`, `z_fit`, `converged` |
| `EisAnalyzer` | `core/analyzer.py` | Orchestration: parse → validate → fit → plot |
| `CVAnalyzer` | `analysis/cv_analyzer.py` | E_onset (3 methods), peak extraction, AOR interpretation |
| `LSVAnalyzer` | `analysis/lsv_analyzer.py` | Tafel slope, j0, overpotential, mass activity |
| `ECSACalculator` | `analysis/ecsa_calculator.py` | H-UPD, CO Stripping, Cdl — 12 tests passing |

---

## Supported File Formats

| Format | Parser | Instrument |
|--------|--------|------------|
| `.idf` | `autolab_parser.py` | Metrohm Autolab (CV + EIS, multi-scan) |
| `.DTA` | `gamry_parser.py` | Gamry Instruments |
| `.mpt` / `.mpr` | `biologic_parser.py` | BioLogic (via galvani) |
| `.csv` / `.txt` | `generic_csv_parser.py` | Any instrument, auto-detects delimiter |

---

## Known Issues (Do Not Ignore These)

1. **EIS-GPT untrained** — `eisforge/ml/eis_gpt/` exists but returns random predictions
2. **K-K on sparse data** — `validators.py` linKK may fail; Voigt fallback kicks in
3. **CV multi-cycle splitting** — `_split_scans()` may fail on complex multi-cycle CVs
4. **Real data not yet validated** — ECSA calculator only tested on synthetic data
5. **impedance.py NaN bug** — `CNLSFitter` intentionally bypasses impedance.py optimizer; use `circuit.predict()` ONLY for Z evaluation, NEVER `circuit.fit()`

---

## Critical Conventions

- **Docstring style**: NumPy format (Parameters / Returns / Raises sections)
- **Test framework**: pytest — all tests in `tests/` directory
- **Commit style**: `feat(module): description` / `fix(module): description` / `test(module): description`
- **Never call `impedance.py` internal fit** — it returns NaN; use `CNLSFitter` only
- **`_split_scans()` uses `argmax`** — not `len//2` — this was a critical bug fix
- **Current unit in autolab_parser**: A → mA conversion is done automatically
- **Encoding in autolab_parser**: tries `latin-1` → `cp1252` → `utf-8` in that order
- **Z'' sign convention**: plot `-Z.imag` (not `+Z.imag`) on Nyquist y-axis

---

## Next Modules to Build (Priority Order)

1. **`band_edge_calculator.py`** — Ecb/Evb for BCN/g-C3N4/TiO2/ZnO ← NEXT
2. **DRT Analysis** — Distribution of Relaxation Times (Tikhonov regularization)
3. **Statistical Reproducibility** — batch multi-file mean ± std
4. **EIS-GPT Training** — 10,000+ synthetic spectra + physics-informed loss
5. **CI/CD Pipeline** — GitHub Actions
6. **Zenodo DOI** — for publication

---

## Dependencies (requirements.txt — do not add unlisted packages)

Core: `numpy`, `scipy`, `pandas`, `pyarrow`  
EIS: `impedance` (impedance.py — for circuit model evaluation ONLY)  
BioLogic: `galvani`  
ML: `torch`  
Viz: `plotly`  
UI: `streamlit`  
Signal: `scipy.signal` (medfilt used in CNLSFitter outlier detection)
