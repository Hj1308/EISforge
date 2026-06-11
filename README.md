# EISForge 🔬⚡

> **Advanced open-source framework for Electrochemical Impedance Spectroscopy (EIS) analysis with Physics-Informed Machine Learning**
>
> *Open-source EIS framework combining classical CNLS fitting, CV/LSV analysis, and a Physics-Informed Transformer (EIS-GPT).*

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

> ⚠️ **Note:** A Zenodo DOI will be registered upon the first official release (v0.1.0). Until then, please cite the GitHub URL above.

---

## Key Features at a Glance 🚀

| Feature | Status |
|---|---|
| CNLS circuit fitting (χ² = 0.0008 on real data) | ✅ Implemented |
| Kramers-Kronig validation | ✅ Implemented |
| CV analysis (E_onset, I_f/I_b, current density) | ✅ Implemented |
| LSV analysis (Tafel slope, overpotential, mass activity) | ✅ Implemented |
| Robust data preprocessing (4 independent methods) | ✅ Implemented |
| Literature-guided initial parameter guesses | ✅ Implemented |
| Autolab (.idf) / Gamry (.dta) / CSV parsers | ✅ Implemented |
| Streamlit web interface (5 analysis tabs) | ✅ Implemented |
| Physics-Informed Transformer architecture (EIS-GPT) | ⚠️ Architecture only — **weights untrained** |
| BioLogic (.mpr/.mpt) parser | 🔄 In progress (v0.2) |
| Zahner (.ism) parser | 🔄 Planned (v0.3) |
| DRT — Distribution of Relaxation Times | 🔄 Planned (v0.3) |
| Train EIS-GPT on synthetic spectra | 🔄 Planned (v0.4) |
| Free & Open Source (MIT) | ✅ |

---

## Installation

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -e .
```

This installs all required dependencies automatically via `pyproject.toml`.

**Launch the web interface:**

```bash
streamlit run app.py
```

---

## Quick Start — Real Working Example

```python
from eisforge.core.analyzer    import EisAnalyzer
from eisforge.core.preprocessor import DataPreprocessor
from eisforge.core.fitter       import CNLSFitter

# 1. Load data — supports Autolab (.idf), Gamry (.dta), CSV
ana = EisAnalyzer()
raw = ana.load("examples/synthetic_randles.csv")   # or your own .idf/.dta file
print(raw)
# EISDataset(n=60, f=[1.00e-02, 1.00e+05] Hz)

# 2. Clean the data
clean = DataPreprocessor.clean_pipeline(raw)

# 3. Fit equivalent circuit
result = CNLSFitter(
    circuit_string="R0-p(R1,CPE1)",
    initial_guess=[10.0, 150.0, 50e-6, 0.85],
).fit(clean)

print(result.parameter_table())
# Parameter        Value        ±Error
# R0           1.000e+01    ~0
# R1           1.500e+02    ~0
# CPE1_0       5.000e-05    ~0
# Reduced χ² ≈ 0.0008
```

> 📁 Example data files are in the `examples/` folder. Run `python examples/generate_synthetic_data.py` to regenerate them.

---

## Supported File Formats

| Instrument | Extension | Status |
|---|---|---|
| Metrohm Autolab NOVA | `.idf` | ✅ |
| Gamry Instruments | `.dta` | ✅ |
| Generic CSV / TXT | `.csv`, `.txt` | ✅ |
| BioLogic EC-Lab | `.mpt`, `.mpr` | 🔄 In progress (v0.2) |
| Zahner | `.ism` | 🔄 Planned (v0.3) |

> Autolab `.idf` parser automatically detects column order and measurement type (CV vs EIS vs LSV). Current is auto-converted from A to mA.

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

### 4. LSV Analysis

```python
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

ana    = LSVAnalyzer(scan_rate=5.0, electrode_area=1.0, electrolyte="acidic")
result = ana.analyze(potential, current)

print(f"Tafel slope     = {result.tafel_slope:.1f} mV/dec")
print(f"η @ 10 mA/cm²  = {result.overpotential_10*1000:.1f} mV")
print(f"Mass activity   = {result.mass_activity:.3f} mA/mg_cat")
```

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
print(guess.recommended_circuit)  # "R0-p(R1,CPE1)"
print(guess.initial_guess)        # {'R0': 15.0, 'R1': 250.0, ...}
print(guess.confidence)           # "high"
```

### 6. EIS-GPT — Physics-Informed Transformer

> ⚠️ **Current Status:** The Transformer **architecture** is fully implemented and tested. Model **weights are untrained** — training on synthetic data is planned for v0.4. Do not use for inference yet.

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

---

## Project Structure

```
EISforge-/
├── app.py                          # Streamlit web interface (5 tabs)
├── pyproject.toml                  # Package metadata + dependencies
├── requirements.txt
│
├── examples/
│   ├── synthetic_randles.csv       # Ready-to-use example EIS data
│   ├── synthetic_warburg.csv       # Randles + Warburg example
│   └── generate_synthetic_data.py  # Script to regenerate examples
│
├── tests/
│   ├── test_eis_fitting.py         # Unit tests: Randles circuit math
│   └── test_parsers.py             # Unit tests: CSV parser
│
└── eisforge/
    ├── core/
    │   ├── analyzer.py
    │   ├── fitter.py
    │   ├── validators.py
    │   └── preprocessor.py
    ├── parsers/
    │   ├── base_parser.py
    │   ├── autolab_parser.py
    │   ├── gamry_parser.py
    │   ├── generic_csv_parser.py
    │   └── biologic_parser.py      # BioLogic .mpr/.mpt (in progress)
    ├── analysis/
    │   ├── cv_analyzer.py
    │   ├── lsv_analyzer.py
    │   └── eis_cv_correlator.py
    ├── ml/
    │   ├── aor_dataset_generator.py
    │   └── eis_gpt/
    │       ├── tokenizer.py
    │       ├── transformer.py
    │       └── physics_loss.py
    └── knowledge/
        ├── literature_engine.py
        └── data/
            └── electrochemistry_knowledge.json
```

---

## Roadmap

- [x] CNLS fitting (direct scipy, χ² = 0.0008 on real data)
- [x] Kramers-Kronig validation with fallback
- [x] Autolab / Gamry / CSV file parsers
- [x] CV analysis (E_onset, I_f/I_b, geometric and ECSA current density)
- [x] LSV analysis (Tafel slope, overpotential, mass/specific activity)
- [x] Robust data preprocessing (4 methods)
- [x] Literature knowledge base
- [x] Streamlit web interface
- [x] Physics-Informed Transformer architecture (EIS-GPT) — untrained
- [x] iR compensation
- [x] CI/CD via GitHub Actions
- [ ] BioLogic (.mpr/.mpt) parser — **in progress (v0.2)**
- [ ] Train EIS-GPT on 10,000+ synthetic spectra (v0.4)
- [ ] Zahner (.ism) parser (v0.3)
- [ ] DRT — Distribution of Relaxation Times (v0.3)
- [ ] Automatic ECSA calculation (H_upd, CO stripping, Cdl methods)
- [ ] Statistical reproducibility (n=3 batch analysis, mean ± SD)
- [ ] Faradaic efficiency calculator
- [ ] Zenodo DOI registration
- [ ] JOSS paper submission

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

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
