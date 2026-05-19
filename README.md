# EISForge 🔬⚡

> **Advanced open-source framework for Electrochemical Impedance Spectroscopy (EIS) analysis with Physics-Informed Machine Learning**
>
> *The first open-source EIS framework combining classical CNLS fitting, CV/LSV analysis, and a Physics-Informed Transformer (EIS-GPT) — a free alternative to ZView and EC-Lab.*

---

## Author & Citation

**Hoda Jafari** | 📧 hoda.jaafari@gmail.com | 🔗 https://github.com/Hj1308 | 📅 May 2026

If you use EISForge in your research, please cite:

```bibtex
@software{jafari2026eisforge,
  author    = {Jafari, Hoda},
  title     = {EISForge: Physics-Informed ML for Electrochemical Impedance Spectroscopy},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Hj1308/EISforge-},
  note      = {Open-source EIS/CV/LSV analysis with Physics-Informed Transformer}
}
```

---

## What Makes EISForge Different? 🚀

| Feature | ZView | EC-Lab | EISForge |
|---|---|---|---|
| CNLS circuit fitting | ✅ | ✅ | ✅ |
| Kramers-Kronig validation | ✅ | ✅ | ✅ |
| CV analysis (E_onset, I_f/I_b) | ❌ | ✅ | ✅ |
| LSV + Tafel analysis | ❌ | ✅ | ✅ |
| Automatic data preprocessing | ❌ | ❌ | ✅ **NEW** |
| Literature-guided initial guesses | ❌ | ❌ | ✅ **NEW** |
| Physics-Informed Transformer (EIS-GPT) | ❌ | ❌ | ✅ **NOVEL** |
| Uncertainty quantification | ❌ | ❌ | ✅ **NEW** |
| Autolab / Gamry / BioLogic support | partial | partial | ✅ |
| Free & Open Source | ❌ | ❌ | ✅ |

---

## Installation

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -r requirements.txt
pip install -e .
```

**Run the web interface:**

```bash
streamlit run app.py
```

---

## Supported File Formats

| Format | Instrument | Extension |
|---|---|---|
| Autolab NOVA | Metrohm Autolab | `.idf` |
| Gamry Framework | Gamry Instruments | `.dta` |
| EC-Lab | BioLogic | `.mpt`, `.mpr` |
| Generic CSV/TXT | Any | `.csv`, `.txt` |

> Autolab `.idf` parser auto-detects column order (Z', Z'', frequency) and file type (CV vs EIS vs LSV). Current is automatically converted from A to mA.

---

## Key Features

### 1. EIS Analysis
- **CNLS Fitting** — direct `scipy.optimize.least_squares` (LM + TRF strategies)
- **Kramers-Kronig Validation** — linKK with Voigt-circuit fallback
- **Supported circuits** — R-RC, R-RCPE, R-RCPE-W, R-RCPE-RCPE, and custom
- **Chi-squared** as low as 0.0008 on real experimental data

### 2. Data Preprocessing (ZView-style)

```python
from eisforge.core.preprocessor import DataPreprocessor

# Remove inductive artifacts (Z'' < 0 at high frequency)
clean = DataPreprocessor.remove_inductive_artifacts(dataset)

# Crop frequency range
clean = DataPreprocessor.crop_frequencies(dataset, f_min=0.01)

# Remove |Z| jumps in Z' OR Z'' (per-axis detection)
clean = DataPreprocessor.remove_z_jumps(dataset, threshold_pct=20.0)

# Drop specific noisy frequencies (50/60 Hz mains noise)
clean = DataPreprocessor.drop_specific_frequency(dataset, target_freq=50.0)

# Full pipeline in one call:
clean = DataPreprocessor.clean_pipeline(dataset)
```

### 3. CV Analysis

```python
from eisforge.analysis.cv_analyzer import CVAnalyzer

ana = CVAnalyzer(scan_rate=50.0, electrode_area=1.0, electrolyte="acidic")
r   = ana.analyze(potential, current)

print(f"E_onset  = {r.e_onset:.4f} V")
print(f"I_f/I_b  = {r.if_ib_ratio:.3f}")
print(f"j_f      = {r.j_forward_peak:.4f} mA/cm²")
```

### 4. LSV Analysis

```python
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

ana = LSVAnalyzer(scan_rate=5.0, electrode_area=1.0, electrolyte="acidic")
r   = ana.analyze(potential, current)

print(f"Tafel slope     = {r.tafel_slope:.1f} mV/dec")
print(f"η @ 10 mA/cm²  = {r.overpotential_10*1000:.1f} mV")
print(f"Mass activity   = {r.mass_activity:.3f} mA/mg")
```

### 5. Literature Knowledge Base

```python
from eisforge.knowledge.literature_engine import LiteratureEngine

guess = LiteratureEngine().query(
    system_type="AOR", catalyst="PtRu/C",
    electrolyte="acidic", alcohol="ethanol", potential=0.5,
)
print(guess.recommended_circuit)  # "R0-p(R1,CPE1)"
print(guess.initial_guess)        # {'R0': 15.0, 'R1': 250.0, ...}
```

Covers: AOR (Pt, Pd, PtRu, PtSn), Li-ion batteries, corrosion, PEMFC, biosensors.

### 6. EIS-GPT — Physics-Informed Transformer

Novel architecture treating EIS as a sequence (each frequency = one token):

```
L_total = L_reconstruction + λ₁·L_KK + λ₂·L_passivity + λ₃·L_HF_limit
```

Predicts circuit topology and parameters directly from the spectrum — no manual circuit selection required.

> Model weights are currently random. Train using `scripts/train_ml_models.py`.

---

## Full Python Example

```python
from eisforge.core.analyzer     import EisAnalyzer
from eisforge.core.preprocessor  import DataPreprocessor
from eisforge.core.fitter        import CNLSFitter

# Load (auto-detects format)
ana = EisAnalyzer()
raw = ana.load("EIS_0.82V.idf")

# Preprocess
clean = DataPreprocessor.clean_pipeline(raw)

# Fit
result = CNLSFitter("R0-p(R1,CPE1)", [30.0, 31000.0, 2e-7, 0.78]).fit(clean)
print(result.parameter_table())

# Parameter           Value          ±Error
# ────────────────────────────────────────
# R0            2.5230e+01      1.77e-01
# R1            3.3462e+04      1.56e-05
# CPE1_0        1.0487e-05      7.21e-08
# CPE1_1        7.4527e-01      1.01e-03
# Reduced χ² = 0.000800
```

---

## Project Structure

```
EISforge-/
├── app.py                          # Streamlit web interface (5 tabs)
├── requirements.txt
├── setup.py
│
└── eisforge/
    ├── core/
    │   ├── analyzer.py             # Orchestration
    │   ├── fitter.py               # CNLS via scipy direct
    │   ├── validators.py           # Kramers-Kronig
    │   └── preprocessor.py         # 4 cleaning methods
    │
    ├── parsers/
    │   ├── base_parser.py          # EISDataset container
    │   ├── autolab_parser.py       # Autolab .idf (CV + EIS)
    │   ├── gamry_parser.py         # Gamry .dta
    │   └── generic_csv_parser.py   # CSV/TXT
    │
    ├── analysis/
    │   ├── cv_analyzer.py          # CV full analysis
    │   ├── lsv_analyzer.py         # LSV + Tafel
    │   └── eis_cv_correlator.py    # Cross-technique correlation
    │
    ├── ml/
    │   ├── aor_dataset_generator.py
    │   └── eis_gpt/
    │       ├── tokenizer.py        # EIS → tokens
    │       ├── transformer.py      # Physics-Informed Transformer
    │       └── physics_loss.py     # K-K + passivity loss
    │
    └── knowledge/
        ├── literature_engine.py
        └── data/
            └── electrochemistry_knowledge.json
```

---

## Roadmap

- [x] CNLS fitting (scipy direct, χ² = 0.0008 on real data)
- [x] Kramers-Kronig validation
- [x] Autolab / Gamry / BioLogic parsers
- [x] CV analysis (E_onset, I_f/I_b, j_geometric, j_ECSA)
- [x] LSV analysis (Tafel, overpotential, mass/specific activity)
- [x] Data preprocessing pipeline (4 independent methods)
- [x] Literature knowledge base
- [x] Streamlit web interface (CV / LSV / EIS / EIS-GPT / Correlation)
- [x] Physics-Informed Transformer architecture
- [ ] Train EIS-GPT on synthetic spectra
- [ ] iR compensation (auto R_s from EIS)
- [ ] Automatic ECSA (H_upd, CO stripping, Cdl)
- [ ] DRT analysis (Distribution of Relaxation Times)
- [ ] Statistical reproducibility (n=3 batch)
- [ ] Faradaic efficiency calculator
- [ ] Zenodo DOI

---

## License

MIT License — Copyright (c) 2026 Hoda Jafari

Free to use in academic and commercial applications.
**Please cite this work if you use it in your publications.**
