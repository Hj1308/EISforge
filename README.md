# EISForge ⚡

![Version](https://img.shields.io/badge/version-v0.2.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20649692-blue?style=flat-square)
![CI](https://github.com/Hj1308/EISforge-/actions/workflows/ci.yml/badge.svg?style=flat-square)
![Tests](https://github.com/Hj1308/EISforge-/actions/workflows/tests.yml/badge.svg?style=flat-square)

**EISForge — Catalyst-Aware Electrochemistry Analysis Toolkit, v0.2.0**  
Author: [Hoda Jaafari](https://github.com/Hj1308) | Affiliation: CCERCI | MIT License | First published: May 2026

> **Looking for surface area & pore analysis (BET/BJH)?**  
> → See [BET_analyser](https://github.com/Hj1308/BET_analyser)  
> **Looking for catalyst kinetics & ODS analysis?**  
> → See [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools)

---

## What is EISForge?

A Streamlit-based web application and Python API for **automated electrochemical analysis** of the **Alcohol Oxidation Reaction (AOR)**.  
Designed for PhD-level electrocatalysis research — covers CV, LSV, EIS, ECSA, Koutecký–Levich, and batch reproducibility.  
EISForge adapts its diagnostics to the **catalyst family** and **electrolyte chemistry**, ensuring peer-review-defensible results.

| Catalyst Family | Supported Electrolytes |
|---|---|
| `noble_metal` (Pt, Pd, Au…) | Acidic: H₂SO₄, HClO₄, HCl |
| `alloy` (PtRu, PdAu…) | Alkaline: KOH, NaOH |
| `metal_oxide` (RuO₂, IrO₂…) | Neutral buffers |
| `carbon_material` / metal-free | Custom pH environments |

---

## 🚀 Quick Start

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -e .
streamlit run app.py
```

With ML dependencies:
```bash
pip install -e ".[ml]"
```

With development tools:
```bash
pip install -e ".[dev]"
```

---

## 📑 Modules

| Module | Technique | Key Outputs |
|---|---|---|
| **CV Analyzer** | Cyclic Voltammetry | Eₒₙₛₑₜ, peak position/height, ECSA, iR-corrected potentials |
| **LSV Analyzer** | Linear Sweep Voltammetry | Tafel slope, η₁₀/η₅₀/η₁₀₀, j₀, mass activity, specific activity |
| **EIS Analyzer** | Impedance Spectroscopy | Rₛ, R_ct, CPE, Nyquist/Bode plots, K–K validation |
| **ECSA Calculator** | Multi-scan-rate CV | Double-layer capacitance, electrochemically active surface area |
| **Koutecký–Levich** | Rotating disk LSV | Electron-transfer number, diffusion-limited current |
| **Batch Analyzer** | CV / LSV / EIS | Mean ± SD over n ≥ 3 replicates, reproducibility scoring |
| **EIS–CV Correlator** | Cross-technique | R_ct ↔ peak current correlation, activity–impedance maps |

---

## 📈 Cyclic Voltammetry (CV)

- Onset potential (Eₒₙₛₑₜ), oxidation peak position and height
- iR-corrected potentials (`E_corr = E − I·Rₛ`)
- Double-layer capacitance and ECSA workflow
- Catalyst-aware peak assignment (alcohol oxidation vs. oxide formation)
- Batch CV (n ≥ 3) with mean ± standard deviation

**Python API:**

```python
from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo

analyzer = CVAnalyzer(
    scan_rate=50,                      # mV/s
    electrode_area=0.1225,             # cm²
    catalyst_type="noble_metal",
    electrolyte=ElectrolyteInfo("alkaline", "KOH", 1.0),
)
E, I = CVAnalyzer.load_csv("sample_cv.csv")
result = analyzer.analyze(E, I, r_s_ohms=3.2)

print(result.ecsa_cm2, "cm²")
print(result.onset_potential, "V vs. RHE")
```

---

## 📉 Linear Sweep Voltammetry (LSV) & Tafel Analysis

- **Tafel slope** from the activation-controlled region (auto-selected, see *Methodology*)
- Onset potential, overpotentials at η₁₀, η₅₀, η₁₀₀ (mA·cm⁻²)
- Limiting current, half-wave potential (E₁/₂)
- Mass activity (A·mg⁻¹) and specific activity (A·cm⁻²_ECSA)
- Optional exchange current density j₀ (requires `equilibrium_potential`)
- Koutecký–Levich: electron-transfer number n
- Batch LSV (n ≥ 3) reproducibility statistics

**Python API:**

```python
from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo

analyzer = LSVAnalyzer(
    scan_rate=5,                       # mV/s — use ≤5 for reliable Tafel
    electrode_area=0.1225,             # cm²
    catalyst_type="carbon_material",
    electrolyte=ElectrolyteInfo("acidic", "H2SO4", 0.5),
    equilibrium_potential=None,        # provide E_eq to compute a true j₀
)
E, I = LSVAnalyzer.load_csv("hbc4_lsv.csv")
result = analyzer.analyze(E, I, r_s_ohms=25.5)

print(result.tafel_slope, "mV/dec",
      "| R² =", result.tafel_r_squared,
      "| warnings:", result.tafel_warnings)
```

---

## 🔬 Electrochemical Impedance Spectroscopy (EIS)

- Equivalent-circuit (CNLS) fitting via `impedance` library
- **Kramers–Kronig validation** (causality, linearity, stationarity)
- Charge-transfer resistance (R_ct), solution resistance (Rₛ)
- CPE and double-layer capacitance extraction
- Interactive Nyquist and Bode plots (Plotly)
- EIS–CV cross-technique correlation

---

## 📐 Methodology Notes

> **Read this before reporting numbers in a manuscript.**

- **Tafel slope** — linear fit of `E vs log₁₀(j)` in the **activation-controlled branch only** (above Eₒₙₛₑₜ, below the current peak). Auto-selects the lowest-slope linear window. Results are flagged when: slope > 120 mV·dec⁻¹, window < 1 decade, R² < 0.99, or window approaches the current peak. Manual override: `auto_tafel_region=False`, `tafel_current_range=(j_min, j_max)`.

- **Exchange current density j₀** — reported **only** when `equilibrium_potential` is provided; obtained by extrapolating the Tafel line to η = 0. Extrapolation to Eₒₙₛₑₜ is physically incorrect and is **never** reported as j₀.

- **Scan rate** — Tafel analysis is most reliable at **≤ 5 mV·s⁻¹**; higher rates inflate the apparent slope via capacitive contribution.

- **iR correction** — always supply `r_s_ohms` (from EIS high-frequency intercept) to obtain corrected onset and Tafel values.

- **ECSA** — estimated from double-layer capacitance (Cdl) via multi-scan-rate CV in the non-Faradaic window, or from H/CO underpotential deposition for noble metals.

- **Kramers–Kronig** — all EIS fits are validated against K–K constraints before reporting circuit parameters.

---

## 🔌 Supported Instruments / Formats

| Instrument | File Format | Parser |
|---|---|---|
| Gamry Instruments | `.DTA` | `gamry_parser.py` |
| BioLogic Science Instruments | `.mpt`, `.mpr` | `biologic_parser.py` |
| Metrohm Autolab | `.idf` | `autolab_parser.py` |
| Generic / custom | `.csv`, `.tsv` | `generic_csv_parser.py` |

All parsers share automatic unit handling (A / mA / µA / nA) and multi-instrument detection.

---

## 🤖 ML Module — EIS-GPT

EISForge includes a **physics-informed Transformer** for automated equivalent-circuit topology prediction from raw impedance spectra.

```
eisforge/ml/eis_gpt/
├── transformer.py            # Multi-head self-attention model architecture
├── tokenizer.py              # Impedance spectrum → token sequence
├── physics_loss.py           # Kramers–Kronig + causality constraints in loss
├── aor_dataset_generator.py  # Synthetic AOR training data generation
└── eis_cv_correlator.py      # EIS ↔ CV cross-modal correlation
```

**Training:**

```bash
python train_eis_gpt.py
```

> ⚠️ Requires optional ML dependencies: `pip install -e ".[ml]"`  
> ⚠️ Pre-trained weights are in development — see Roadmap.

---

## 📚 Knowledge Base

The `eisforge/knowledge/` module contains a curated, literature-guided scientific layer built from **195 peer-reviewed papers**. It supports catalyst-aware interpretation rules, electrochemical diagnostics, and context-aware analysis across all EISForge modules.

> This knowledge layer is an original contribution of EISForge. If your research uses EISForge diagnostics, interpretation outputs, or any results derived from this knowledge base, please cite EISForge accordingly (see **Citation and Attribution** below).

---

## 🧪 Testing

```bash
# Run full test suite
pytest

# With coverage report
pytest --cov=eisforge --cov-report=term-missing
```

| Test File | Covers |
|---|---|
| `tests/test_eis_fitting.py` | EIS CNLS fitting & Kramers–Kronig |
| `tests/test_ecsa.py` | ECSA / double-layer capacitance |
| `tests/test_batch_analyzer.py` | Batch statistics & reproducibility |
| `tests/test_parsers.py` | Multi-instrument file parsers |

CI runs automatically on every push via [GitHub Actions](https://github.com/Hj1308/EISforge-/actions).

---

## 🗂 Repository Structure

```
EISforge-/
├── app.py                        # Streamlit web application
├── train_eis_gpt.py              # EIS-GPT training script
├── eisforge/
│   ├── analysis/
│   │   ├── cv_analyzer.py
│   │   ├── lsv_analyzer.py
│   │   ├── ecsa_calculator.py
│   │   ├── koutecky_levich.py
│   │   ├── eis_cv_correlator.py
│   │   └── batch_analyzer.py
│   ├── core/
│   │   ├── analyzer.py
│   │   ├── fitter.py
│   │   ├── preprocessor.py
│   │   └── validators.py
│   ├── parsers/
│   │   ├── gamry_parser.py
│   │   ├── biologic_parser.py
│   │   ├── autolab_parser.py
│   │   └── generic_csv_parser.py
│   ├── ml/
│   │   ├── eis_gpt/              # Physics-informed Transformer
│   │   └── uncertainty/          # Uncertainty quantification
│   ├── knowledge/                # Literature-guided knowledge base (195 papers)
│   ├── standards/                # Electrochemical reference standards
│   ├── utils/
│   └── visualization/
├── tests/
├── examples/
├── paper.md                      # JOSS manuscript
├── paper.bib
├── CITATION.cff
└── pyproject.toml
```

---

## 🔗 Related Repositories

| Repo | Purpose |
|---|---|
| [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools) | ODS kinetics, TOF/TON, reusability analysis |
| [BET_analyser](https://github.com/Hj1308/BET_analyser) | BET, BJH, T-Plot, isotherm & hysteresis |
| [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) | SEM particle sizing |
| [Raman-analysis](https://github.com/Hj1308/Raman-analysis) | Raman spectroscopy toolkit |

---

## 📌 Citation and Attribution

> ⚠️ **Citation is required.** If you use EISForge — including its analysis modules, curated `knowledge/` resources, or any derived outputs — in publications, theses, reports, presentations, or other software, you **must** cite the repository and the associated Zenodo record.

```bibtex
@software{jafari2026eisforge,
  author       = {Jaafari, Hoda},
  title        = {EISForge: A Catalyst-Aware Electrochemistry Analysis Toolkit},
  year         = {2026},
  doi          = {10.5281/zenodo.20649692},
  publisher    = {Zenodo},
  url          = {https://github.com/Hj1308/EISforge-},
  version      = {0.2.0},
  affiliation  = {CCERCI},
  note         = {Automated CV / LSV / EIS analysis for the alcohol oxidation reaction}
}
```

**DOI:** [10.5281/zenodo.20649692](https://doi.org/10.5281/zenodo.20649692)

A `CITATION.cff` file is included for automatic citation parsing by GitHub and Zenodo.

---

## 🗺 Roadmap

| Feature | Status |
|---|---|
| CV / LSV / EIS analysis engine | ✅ Implemented |
| Batch processing (n ≥ 3) | ✅ Implemented |
| Multi-instrument parsers (Gamry, BioLogic, Autolab) | ✅ Implemented |
| Kramers–Kronig validation | ✅ Implemented |
| Koutecký–Levich analysis | ✅ Implemented |
| EIS-GPT Transformer model architecture | ✅ Implemented |
| Pre-trained EIS-GPT model weights | 🔬 In development |
| DRT (distribution of relaxation times) | 📋 Planned |
| PyPI package release | 📋 Planned |

---

## License

MIT — free to use, modify, and distribute. **Citation is required** for any academic or research use. See **Citation and Attribution** above.
