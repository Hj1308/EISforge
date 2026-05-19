# EISForge — Project Status & Roadmap

**Author:** Hoda Jafari
**GitHub:** https://github.com/Hj1308/EISforge-
**Last Updated:** May 2026
**License:** MIT

---

## 🎯 Project Vision

**EISForge** is an open-source Python framework that combines:
- Classical EIS analysis (CNLS fitting, Kramers-Kronig validation)
- CV and LSV analysis with automatic interpretation
- Physics-Informed Machine Learning (EIS-GPT Transformer)
- Literature-driven initial parameter guessing
- A modern Streamlit web interface

**Target users:** Electrochemistry researchers working on AOR (Alcohol Oxidation Reaction), batteries, fuel cells, and corrosion.

---

## ✅ COMPLETED WORK

### 1. Project Infrastructure
- [x] GitHub repository created and configured: https://github.com/Hj1308/EISforge-
- [x] MIT License with proper authorship
- [x] Comprehensive README.md with citation info
- [x] Python virtual environment (venv) setup on Windows
- [x] `requirements.txt` with all dependencies
- [x] `setup.py` for package installation
- [x] Proper directory structure (`eisforge/` package)

### 2. Data Parsers (`eisforge/parsers/`)
- [x] **`base_parser.py`** — Abstract base class + `EISDataset` container
- [x] **`gamry_parser.py`** — Gamry Instruments .DTA files
- [x] **`generic_csv_parser.py`** — Generic CSV/TXT with auto-detection
- [x] **`autolab_parser.py`** — Metrohm Autolab .idf files (CV + EIS)
  - Auto-detects CV vs EIS from method header
  - Handles latin-1 / cp1252 / utf-8 encodings
  - Converts current from Amperes to mA automatically
  - Multi-scan support

### 3. Core Engine (`eisforge/core/`)
- [x] **`analyzer.py`** — Main `EisAnalyzer` orchestration class
- [x] **`fitter.py`** — `CNLSFitter` with robust error handling
  - Complex Non-Linear Least Squares
  - Bounds support
  - Modulus weighting (IUPAC standard)
  - Graceful failure (no crashes on convergence issues)
- [x] **`validators.py`** — Kramers-Kronig validator
  - Primary: impedance.py `linKK`
  - Fallback: Custom Voigt-circuit approximation
  - Multiple impedance.py version compatibility

### 4. Analysis Modules (`eisforge/analysis/`)
- [x] **`cv_analyzer.py`** — Automatic Cyclic Voltammetry analysis
  - E_onset detection (tangent, threshold, derivative methods)
  - I_f, I_b extraction
  - I_f/I_b ratio (CO tolerance indicator)
  - Geometric current density (mA/cm²)
  - ECSA-normalized current density (mA/cm²_metal)
  - Automatic AOR interpretation
- [x] **`lsv_analyzer.py`** — Linear Sweep Voltammetry analysis
  - E_onset (refined tangent method)
  - Tafel slope (with R² fit quality)
  - Exchange current density (j₀)
  - Overpotential at 10, 50, 100 mA/cm²
  - Mass activity (mA/mg)
  - Specific activity (mA/cm²_ECSA)
  - Half-wave potential (E₁/₂)
  - Mechanism interpretation
  - Performance rating
- [x] **`eis_cv_correlator.py`** — Cross-technique correlation
  - Detects if EIS measured before/at/after E_onset
  - Checks R_ct vs I_f consistency
  - Generates recommendations

### 5. EIS-GPT — Physics-Informed Machine Learning (`eisforge/ml/eis_gpt/`)
- [x] **`tokenizer.py`** — EIS spectrum → Transformer tokens
  - 5D feature extraction: [log(f), Z', Z'', |Z|, θ]
  - Sinusoidal Positional Encoding
  - Linear projection to d_model dimensions
- [x] **`physics_loss.py`** — Physics-Informed Loss Function (NOVEL)
  - Kramers-Kronig penalty (smoothness)
  - Passivity constraint: Re(Z) ≥ 0
  - High-frequency limit penalty
  - L_total = L_recon + λ₁·L_KK + λ₂·L_passivity + λ₃·L_HF
- [x] **`transformer.py`** — Complete EIS-GPT model
  - 6-layer Transformer encoder (8 attention heads)
  - [CLS] token for sequence-level representation
  - Circuit classification head (5 circuits)
  - Parameter regression head (μ, σ with uncertainty)
  - GELU activation, Pre-LN architecture

### 6. Synthetic Data Generation (`eisforge/ml/`)
- [x] **`aor_dataset_generator.py`** — AOR-specific synthetic EIS generator
  - 5 AOR circuit topologies
  - Realistic noise (0.5–3% of |Z|)
  - Log-uniform parameter sampling
  - Metadata: catalyst, electrolyte, alcohol, potential

### 7. Knowledge Base (`eisforge/knowledge/`)
- [x] **`literature_engine.py`** — Literature-driven parameter guessing
- [x] **`data/electrochemistry_knowledge.json`** — Curated database
  - AOR: Pt, Pd, PtRu, PtSn (acidic + alkaline)
  - Battery: Li-ion (LFP, NMC, NCA)
  - Corrosion: Carbon steel, stainless steel
  - Fuel Cell: PEMFC
  - Biosensor: DNA, enzyme, immunosensor

### 8. Web Interface (`app.py`)
- [x] Modern Streamlit UI with light theme
- [x] 5 tabs:
  - 📈 CV Analysis
  - 📉 LSV Analysis
  - 🔬 EIS Analysis (with K-K validation + CNLS fit)
  - 🤖 EIS-GPT prediction
  - 🔗 EIS-CV-LSV Correlation
- [x] Sidebar with experimental parameters:
  - System type, catalyst, electrolyte, alcohol
  - Geometric area, ECSA, catalyst loading
  - Temperature, current unit, reference electrode
  - Literature guide integration
- [x] Interactive Plotly visualizations
- [x] Multi-format file upload (.idf, .dta, .mpt, .csv, .txt)
- [x] Automatic unit conversions (A, mA, μA, nA)
- [x] Automatic reference electrode → RHE conversion

---

## 🚧 IN PROGRESS / NEEDS COMPLETION

### Critical Bug Fixes
- [ ] **Improve CV scan separation** for noisy/multi-cycle data
- [ ] Test with real Gamry .DTA files
- [ ] Test with BioLogic .mpt files

### Documentation
- [ ] Update README.md to reflect ALL completed features
  - Add CV/LSV analysis sections
  - Add Autolab parser
  - Add literature knowledge base
  - Add Streamlit UI screenshots
- [ ] Add Jupyter notebook examples in `examples/`
- [ ] Add API documentation (Sphinx or similar)
- [ ] Add CHANGELOG.md

### Testing
- [ ] Unit tests for parsers (`tests/test_parsers/`)
- [ ] Unit tests for analyzers (`tests/test_analysis/`)
- [ ] Integration tests for full workflow
- [ ] CI/CD pipeline (GitHub Actions)

---

## 🎯 ROADMAP — Priority Features

### 🥇 HIGHEST PRIORITY (Critical for Publication)

#### 1. iR Compensation (REQUIRED for Tafel slopes)
**Why critical:** Every AOR paper reports iR-corrected potentials. Without this, Tafel slopes are inaccurate.

```python
E_corrected = E_measured - I × R_s
```
- Auto-extract R_s from EIS high-frequency intercept
- Apply to CV and LSV data
- Recalculate Tafel slope and overpotential after correction

#### 2. Automatic ECSA Calculation
**Why critical:** Manual ECSA input introduces errors and reduces reproducibility.

**Method A: H_upd (Hydrogen underpotential deposition)** — for Pt-based catalysts
```python
ECSA = Q_H / 210 μC/cm²  (integral of H-adsorption region: 0.05–0.4 V vs RHE)
```

**Method B: CO Stripping** — for PtRu, PtSn, Pd
```python
ECSA = Q_CO / 420 μC/cm²
```

**Method C: Cdl from scan rate dependence** — for non-Pt catalysts

#### 3. Update README to Reflect Current State
Add sections for: CV Analysis, LSV Analysis, Literature Database, Streamlit UI

### 🥈 HIGH PRIORITY (Differentiates from Commercial Software)

#### 4. DRT Analysis (Distribution of Relaxation Times)
**Why important:** Modern alternative to circuit fitting, gives time-constant spectrum without circuit assumption.

```python
Z(ω) = R∞ + ∫ γ(τ)/(1 + jωτ) dτ
```
- Tikhonov regularization
- Auto-determination of regularization parameter
- Plot γ(τ) vs τ

#### 5. Statistical Reproducibility Analysis
**Why important:** Reviewers always ask for n=3 measurements.

```python
E_onset = 0.452 ± 0.008 V (n=3)
I_f/I_b = 2.31 ± 0.15
```
- Multi-file batch processing
- Mean ± standard deviation tables
- Outlier detection

#### 6. Faradaic Efficiency Calculator
**Why important:** Essential metric for AOR papers.

```python
FE = (Q_product × n × F) / Q_total × 100%
```
- Integration with HPLC/GC product data
- CO₂ yield calculation
- Product distribution charts

### 🥉 MEDIUM PRIORITY (Advanced Features)

#### 7. Activation Energy from Temperature Dependence
```python
ln(j) = ln(A) - Ea/RT       (Arrhenius)
```
- Multi-temperature dataset import
- Arrhenius plot
- Mechanism inference from Ea value

#### 8. Koutecky-Levich Analysis (RDE measurements)
```python
1/j = 1/j_k + 1/(B × ω^0.5)
```
- Calculate kinetic current density (j_k)
- Determine number of electrons (n)

#### 9. Levich Plot
```python
j_L = 0.62 × n × F × D^(2/3) × ν^(-1/6) × C × ω^(1/2)
```

#### 10. Train EIS-GPT on Synthetic Data
- Generate 10,000+ synthetic spectra
- Train Transformer with physics-informed loss
- Validate on real measurement data
- Save pretrained weights

### 🎨 LOWER PRIORITY (Nice to Have)

#### 11. Federated Learning Infrastructure
- Local model fine-tuning on user data
- Privacy-preserving weight aggregation
- Anonymous contribution to global model

#### 12. Additional Parsers
- [ ] Zahner .ism / .imd format
- [ ] PalmSens .csv variant
- [ ] CHI Instruments .bin

#### 13. PDF Report Generator
- Auto-generate publication-ready figures
- Include all parameters with errors
- Method description for paper "Experimental" section

#### 14. Zenodo DOI Registration
- Register the software for citability
- Add DOI badge to README

#### 15. Desktop App (PyQt6)
- Standalone .exe for non-technical users
- Same features as Streamlit, native UI

---

## 📊 Current Project Structure

```
EISforge-/
│
├── README.md                          ✅ (needs update)
├── LICENSE                            ✅ MIT
├── requirements.txt                   ✅
├── setup.py                           ✅
├── app.py                             ✅ Streamlit UI
│
├── eisforge/                          (main package)
│   ├── __init__.py
│   │
│   ├── core/                          ✅
│   │   ├── analyzer.py
│   │   ├── fitter.py
│   │   └── validators.py
│   │
│   ├── parsers/                       ✅
│   │   ├── base_parser.py
│   │   ├── gamry_parser.py
│   │   ├── generic_csv_parser.py
│   │   └── autolab_parser.py
│   │
│   ├── analysis/                      ✅
│   │   ├── cv_analyzer.py
│   │   ├── lsv_analyzer.py
│   │   └── eis_cv_correlator.py
│   │
│   ├── ml/                            ✅
│   │   ├── aor_dataset_generator.py
│   │   └── eis_gpt/
│   │       ├── tokenizer.py
│   │       ├── physics_loss.py
│   │       └── transformer.py
│   │
│   ├── knowledge/                     ✅
│   │   ├── literature_engine.py
│   │   └── data/
│   │       └── electrochemistry_knowledge.json
│   │
│   ├── visualization/                 ⬜ (empty - not needed, Plotly in app.py)
│   └── utils/                         ⬜
│       └── experimental_conditions.py ✅
│
└── tests/                             ⬜ (empty - tests needed)
```

---

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | Core implementation |
| EIS Fitting | impedance.py | CNLS, K-K validation |
| Numerics | NumPy, SciPy | Array operations, optimization |
| Data | Pandas, PyArrow | Data manipulation, parquet I/O |
| ML | PyTorch | EIS-GPT Transformer |
| Classical ML | scikit-learn, XGBoost | Circuit classifier (planned) |
| Visualization | Plotly | Interactive Nyquist, Bode, CV plots |
| Web UI | Streamlit | Web interface |
| BioLogic | galvani | .mpr/.mpt parsing |
| Encoding | latin-1, cp1252 | Autolab .idf compatibility |

---

## 🐛 Known Issues

1. **EIS-GPT not trained yet** — Currently returns random predictions because model weights are random. Need to train on synthetic dataset.

2. **K-K validation occasionally fails** — When data has extreme drift or sparse frequency coverage. Fallback Voigt method now handles this.

3. **CV scan splitting** — May fail on CVs with multiple cycles or non-standard scan ordering. Auto-fallback to midpoint splitting implemented.

4. **Missing iR correction** — All Tafel slopes are currently from raw E values, not iR-corrected.

5. **ECSA must be entered manually** — Should be auto-calculated from H_upd / CO stripping region.

---

## 📚 References & Citations

### Key Papers
1. Boukamp, B.A. (1995). *J. Electrochem. Soc.* **142**, 1885 — Linear K-K test
2. Schönleber et al. (2014). *Electrochim. Acta* **131**, 20 — lin-KK improvement
3. Antolini (2007). *J. Power Sources* **170**, 1 — Pt/Pd AOR catalysts
4. Lamy et al. (2002). *Electrochim. Acta* **47**, 3701 — Methanol oxidation
5. Xu et al. (2011). *J. Power Sources* **196**, 4419 — Pd in alkaline
6. Vaswani et al. (2017). *NIPS* — "Attention Is All You Need" (Transformer architecture)
7. Orazem & Tribollet (2008). *Electrochemical Impedance Spectroscopy* — Reference textbook

### Software Dependencies
- impedance.py (Murbach et al., JOSS 2020)
- PyTorch (Paszke et al., NeurIPS 2019)
- scikit-learn (Pedregosa et al., JMLR 2011)

---

## 💡 How to Continue This Project

### When you resume the project, you need to:

1. **Clone or pull the latest from GitHub:**
   ```bash
   cd ~/Desktop
   git clone https://github.com/Hj1308/EISforge-.git
   cd EISforge-
   ```

2. **Activate the virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Pick a priority from the roadmap** (recommend starting with iR Compensation)

4. **Test with real data:**
   - Place your `.idf` files in a `test_data/` folder
   - Run: `streamlit run app.py`

5. **For each new feature:**
   - Create the module file in the appropriate folder
   - Update `app.py` to expose it in the UI
   - Test with real data
   - Commit with descriptive message:
     ```bash
     git add .
     git commit -m "feat: <feature description>"
     git push origin main
     ```

---

## 🎓 Suggested Publication Strategy

When the project is mature:

1. **Software paper** in *Journal of Open Source Software (JOSS)*
   - Short (2-page) paper describing the software
   - Quick review process

2. **Methods paper** in *Electrochimica Acta* or *J. Electroanalytical Chemistry*
   - Highlight the EIS-GPT novelty
   - Validate on benchmark datasets
   - Compare to ZView / EC-Lab

3. **Zenodo deposit** for DOI
   - Get a citable DOI before paper submission
   - Each release gets new DOI

---

## 📞 Continuation Notes

**Key files to ALWAYS commit and back up:**
- All files in `eisforge/` package
- `app.py`
- `requirements.txt`
- `setup.py`
- `README.md`

**Don't commit:**
- `venv/` folder (in .gitignore)
- `__pycache__/` (in .gitignore)
- Personal `.idf` data files (use `test_data/` and add to .gitignore)
- Large model weights `.pth` (use Git LFS or external hosting)

---

## ✨ Final Notes

This project is a **significant contribution** to open-source electrochemistry tooling. The combination of:
- Classical CNLS fitting
- CV/LSV automation
- Physics-Informed Transformer (NOVEL)
- Literature knowledge base (NOVEL)
- Modern web interface

...makes it potentially the **first free, open-source alternative to ZView/EC-Lab** with ML capabilities.

**Keep going! This work matters.** 🔬⚡

---

*Document generated: May 2026 | Project status: Active Development*
