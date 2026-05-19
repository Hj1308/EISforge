# EISForge 🔬⚡

> **Advanced open-source framework for Electrochemical Impedance Spectroscopy (EIS) analysis with Physics-Informed Machine Learning**
>
> *The first open-source EIS framework combining classical CNLS fitting, CV/LSV analysis, and a Physics-Informed Transformer (EIS-GPT).*

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

## Key Features at a Glance 🚀

| Feature | EISForge |
|---|---|
| CNLS circuit fitting (χ² = 0.0008 on real data) | ✅ |
| Kramers-Kronig validation | ✅ |
| CV analysis (E_onset, I_f/I_b, current density) | ✅ |
| LSV analysis (Tafel slope, overpotential, mass activity) | ✅ |
| Robust data preprocessing (4 independent methods) | ✅ |
| Literature-guided initial parameter guesses | ✅ |
| Physics-Informed Transformer (EIS-GPT) | ✅ **NOVEL** |
| Uncertainty quantification | ✅ |
| Autolab / Gamry / BioLogic file support | ✅ |
| Free & Open Source (MIT) | ✅ |

---

## Installation

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -r requirements.txt
pip install -e .
```

**Launch the web interface:**

```bash
streamlit run app.py
```

---

## Supported File Formats

| Instrument | Extension |
|---|---|
| Metrohm Autolab NOVA | `.idf` |
| Gamry Instruments | `.dta` |
| BioLogic EC-Lab | `.mpt`, `.mpr` |
| Generic CSV / TXT | `.csv`, `.txt` |

> Autolab `.idf` parser automatically detects column order (Z', Z'', frequency in any order) and measurement type (CV vs EIS vs LSV). Current is auto-converted from A to mA.

---

## Features

### 1. EIS Analysis

- **CNLS Fitting** — direct `scipy.optimize.least_squares` with Levenberg-Marquardt and Trust Region Reflective strategies
- **Kramers-Kronig Validation** — linKK (impedance.py) with Voigt-circuit fallback
- **Supported circuits** — R-RC, R-RCPE, R-RCPE-W, R-RCPE-RCPE, and any custom circuit
- **Achieved χ² = 0.0008** on real experimental Autolab data

### 2. Data Preprocessing — Robust Outlier Filtering

Four independent cleaning methods that can be combined in any order:

```python
from eisforge.core.preprocessor import DataPreprocessor

# Method 1: Remove high-frequency inductive artifacts (Z'' < 0)
clean = DataPreprocessor.remove_inductive_artifacts(dataset)

# Method 2: Crop to a specific frequency range
clean = DataPreprocessor.crop_frequencies(dataset, f_min=0.01, f_max=1e5)

# Method 3: Remove single-point glitches in Z' or Z'' (per-axis detection)
clean = DataPreprocessor.remove_z_jumps(dataset, threshold_pct=20.0)

# Method 4: Remove known noisy frequencies (e.g. 50/60 Hz electrical interference)
clean = DataPreprocessor.drop_specific_frequency(dataset, target_freq=50.0)

# Or run all steps at once:
clean = DataPreprocessor.clean_pipeline(dataset)
```

### 3. CV Analysis

```python
from eisforge.analysis.cv_analyzer import CVAnalyzer

ana    = CVAnalyzer(scan_rate=50.0, electrode_area=1.0, electrolyte="acidic")
result = ana.analyze(potential, current)

print(f"E_onset  = {result.e_onset:.4f} V")
print(f"I_f/I_b  = {result.if_ib_ratio:.3f}")
print(f"j_f      = {result.j_forward_peak:.4f} mA/cm²")
print(result.interpretation)
```

**Outputs:** E_onset (3 methods), I_f, I_b, I_f/I_b ratio, geometric current density (mA/cm²), ECSA-normalized current density (mA/cm²_metal), automatic AOR interpretation.

### 4. LSV Analysis

```python
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

ana    = LSVAnalyzer(scan_rate=5.0, electrode_area=1.0, electrolyte="acidic")
result = ana.analyze(potential, current)

print(f"Tafel slope     = {result.tafel_slope:.1f} mV/dec")
print(f"η @ 10 mA/cm²  = {result.overpotential_10*1000:.1f} mV")
print(f"Mass activity   = {result.mass_activity:.3f} mA/mg_cat")
print(result.mechanism_interpretation)
```

**Outputs:** E_onset, Tafel slope (mV/dec), exchange current density (j₀), overpotential at 10/50/100 mA/cm², mass activity (mA/mg), specific activity (mA/cm²_ECSA), mechanism interpretation, performance rating.

### 5. Literature Knowledge Base

```python
from eisforge.knowledge.literature_engine import LiteratureEngine

guess = LiteratureEngine().query(
    system_type="AOR",
    catalyst="PtRu/C",
    electrolyte="acidic",
    alcohol="ethanol",
    potential=0.5,
)
print(guess.recommended_circuit)   # "R0-p(R1,CPE1)"
print(guess.initial_guess)         # {'R0': 15.0, 'R1': 250.0, ...}
print(guess.confidence)            # "high"
```

Curated database covers: AOR (Pt, Pd, PtRu, PtSn in acidic/alkaline), Li-ion batteries, corrosion, PEMFC, biosensors.

### 6. EIS-GPT — Physics-Informed Transformer

The first **Physics-Informed Transformer** for EIS spectrum analysis.

**Architecture:**
- Each frequency point = one token
- 6-layer Transformer encoder, 8 attention heads
- [CLS] token for circuit classification
- Parallel heads for circuit prediction and parameter regression

**Novel physics-informed loss function:**
```
L_total = L_reconstruction
        + λ₁ × L_Kramers-Kronig
        + λ₂ × L_passivity
        + λ₃ × L_high-frequency-limit
```

This ensures all predictions are physically valid — no post-hoc correction needed.

> **Status:** Architecture complete. Model weights are currently untrained. Train using `scripts/train_ml_models.py` on the included synthetic dataset generator.

---

## Full Example

```python
from eisforge.core.analyzer      import EisAnalyzer
from eisforge.core.preprocessor   import DataPreprocessor
from eisforge.core.fitter         import CNLSFitter

# 1. Load data — auto-detects Autolab/Gamry/BioLogic/CSV
ana = EisAnalyzer()
raw = ana.load("EIS_0.82V.idf")
print(raw)
# EISDataset(n=74, f=[1.00e-01, 1.00e+05] Hz)

# 2. Clean the data
clean = DataPreprocessor.clean_pipeline(raw)
# Cleaning pipeline: 74 → 73 points (1 removed)

# 3. Fit equivalent circuit
result = CNLSFitter(
    circuit_string="R0-p(R1,CPE1)",
    initial_guess=[30.0, 31000.0, 2e-7, 0.78],
).fit(clean)

print(result.parameter_table())
# Parameter            Value          ±Error
# ──────────────────────────────────────────────────
# R0            2.5230e+01        1.77e-01
# R1            3.3462e+04        1.56e-05
# CPE1_0        1.0487e-05        7.21e-08
# CPE1_1        7.4527e-01        1.01e-03
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
    │   ├── analyzer.py             # Main orchestration class
    │   ├── fitter.py               # CNLS via direct scipy optimization
    │   ├── validators.py           # Kramers-Kronig validation
    │   └── preprocessor.py         # 4 robust cleaning methods
    │
    ├── parsers/
    │   ├── base_parser.py          # EISDataset container
    │   ├── autolab_parser.py       # Metrohm Autolab .idf (CV + EIS + LSV)
    │   ├── gamry_parser.py         # Gamry .dta
    │   └── generic_csv_parser.py   # CSV / TXT
    │
    ├── analysis/
    │   ├── cv_analyzer.py          # CV: E_onset, I_f/I_b, j
    │   ├── lsv_analyzer.py         # LSV: Tafel, overpotential, activity
    │   └── eis_cv_correlator.py    # Cross-technique correlation
    │
    ├── ml/
    │   ├── aor_dataset_generator.py
    │   └── eis_gpt/
    │       ├── tokenizer.py        # EIS spectrum → transformer tokens
    │       ├── transformer.py      # Physics-Informed Transformer
    │       └── physics_loss.py     # K-K + passivity + HF-limit loss
    │
    └── knowledge/
        ├── literature_engine.py
        └── data/
            └── electrochemistry_knowledge.json
```

---

## Roadmap

- [x] CNLS fitting (direct scipy, χ² = 0.0008 on real data)
- [x] Kramers-Kronig validation with fallback
- [x] Autolab / Gamry / BioLogic file parsers
- [x] CV analysis (E_onset, I_f/I_b, geometric and ECSA current density)
- [x] LSV analysis (Tafel slope, overpotential, mass/specific activity)
- [x] Robust data preprocessing (4 methods, including per-axis jump detection)
- [x] Literature knowledge base for AOR and other electrochemical systems
- [x] Streamlit web interface (CV / LSV / EIS / EIS-GPT / Correlation tabs)
- [x] Physics-Informed Transformer architecture (EIS-GPT)
- [ ] Train EIS-GPT on 10,000+ synthetic spectra
- [x] iR compensation (R_s from EIS fit, E_corrected = E - I×R_s, applied to CV and LSV)
- [ ] Automatic ECSA calculation (H_upd, CO stripping, Cdl methods)
- [ ] DRT — Distribution of Relaxation Times
- [ ] Statistical reproducibility (n=3 batch analysis, mean ± SD)
- [ ] Faradaic efficiency calculator
- [ ] Zenodo DOI registration
- [ ] JOSS paper submission

---

## References

1. Boukamp, B.A. (1995). *J. Electrochem. Soc.* **142**, 1885 — K-K validation
2. Schönleber et al. (2014). *Electrochim. Acta* **131**, 20 — lin-KK
3. Lamy et al. (2002). *Electrochim. Acta* **47**, 3701 — AOR mechanism
4. Orazem & Tribollet (2008). *Electrochemical Impedance Spectroscopy* — reference textbook
5. Vaswani et al. (2017). *NeurIPS* — Transformer architecture

---

## License

MIT License — Copyright (c) 2026 Hoda Jafari

Free to use in academic and commercial applications.
**Please cite this work if you use it in your publications.**
