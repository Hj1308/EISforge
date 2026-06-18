# EISForge ⚡

> **A catalyst-aware electrochemistry analysis toolkit for the Alcohol Oxidation Reaction (AOR).**
>
> Automated **CV**, **LSV**, and **EIS** analysis with iR correction, kinetic-region
> Tafel fitting, onset detection, ECSA estimation, and batch (n ≥ 3) reproducibility —
> wrapped in a Streamlit app and a clean Python API.

---

## Author & Citation

**Hoda Jafari** · 📧 hoda.jaafari@gmail.com · 🔗 https://github.com/Hj1308 · 📅 First published: May 2026

If you use EISForge in your research, please cite:

```bibtex
@software{jafari2026eisforge,
  author    = {Jafari, Hoda},
  title     = {EISForge: A Catalyst-Aware Electrochemistry Analysis Toolkit},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Hj1308/EISforge-},
  note      = {Automated CV / LSV / EIS analysis for the alcohol oxidation reaction}
}
```

---

## What EISForge does (implemented)

EISForge analyses three standard electrochemical measurements and adapts its
diagnostics to the **catalyst family** (`noble_metal`, `alloy`, `metal_oxide`,
`carbon_material` / metal-free) and the **electrolyte** (acidic / alkaline, with
specific handling for H₂SO₄, HClO₄, HCl, KOH, NaOH, …).

### 📈 Cyclic Voltammetry (CV)
- Onset potential (Eₒₙₛₑₜ), oxidation peak position and height
- iR-corrected potentials
- Double-layer capacitance / ECSA workflow
- Batch CV (n ≥ 3) with mean ± standard deviation

### 📉 Linear Sweep Voltammetry (LSV)
- **Tafel slope from the activation-controlled region** (see *Methodology* below)
- Onset potential, overpotentials at 10 / 50 / 100 mA·cm⁻²
- Limiting current, half-wave potential
- Mass activity (per mg) and specific activity (per cm²_ECSA / cm²_BET)
- Optional exchange current density (j₀) when an equilibrium potential is provided
- Batch LSV (n ≥ 3) with mean ± standard deviation

### 🔬 Electrochemical Impedance Spectroscopy (EIS)
- Equivalent-circuit (CNLS) fitting and Kramers–Kronig validation
- Charge-transfer resistance, CPE / double-layer capacitance
- Nyquist and Bode visualisation

All three share iR compensation (`E_corr = E − I·Rₛ`), unit handling
(A / mA / µA / nA), and multi-instrument file parsing.

---

## Methodology notes (read before publishing numbers)

These are the choices that determine whether a reported value survives peer review:

- **Tafel slope.** The slope is taken from a linear fit of `E vs log₁₀(j)` over the
  **activation-controlled branch only** (above onset, below the current peak). The
  analyzer auto-selects the lowest-slope linear window (mass transport can only
  *raise* the apparent slope), and flags the result when the slope exceeds
  ~120 mV·dec⁻¹, when the window spans < 1 decade of current, when R² < 0.99, or
  when the window approaches the current peak. You can also fix the current window
  manually (`auto_tafel_region=False`, `tafel_current_range=(j_min, j_max)`).
- **Exchange current density (j₀).** Reported **only** if you supply an
  `equilibrium_potential`; it is then obtained by extrapolating the Tafel line to
  zero overpotential (η = 0). Extrapolating to Eₒₙₛₑₜ is *not* a true j₀ and is not
  reported as one.
- **Scan rate.** Kinetic Tafel analysis from LSV is most reliable at ≤ 5 mV·s⁻¹;
  higher rates inflate the apparent slope through capacitive / double-layer current.
- **iR correction.** Always pass `r_s_ohms` (e.g. Rₛ from EIS) so onset and Tafel are
  computed on iR-corrected potentials.

---

## Installation

```bash
git clone https://github.com/Hj1308/EISforge-.git
cd EISforge-
pip install -r requirements.txt
```

---

## Quick start

### Streamlit app

```bash
streamlit run app.py
```

Upload a CV / LSV / EIS file, set the electrode area, catalyst type, electrolyte
and (optionally) Rₛ, and read the analysed parameters and plots.

### Python API

```python
from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo

analyzer = LSVAnalyzer(
    scan_rate=5,                       # mV/s
    electrode_area=0.1225,             # cm^2
    catalyst_type="carbon_material",   # metal-free / carbon
    electrolyte=ElectrolyteInfo("acidic", "H2SO4", 0.5),
    equilibrium_potential=None,        # set to E_eq (same frame) to get a true j0
)

E, I = LSVAnalyzer.load_csv("hbc4_lsv.csv")   # potential (V), current (mA)
result = analyzer.analyze(E, I, r_s_ohms=25.5)  # iR-corrected

print(result.summary())
print(result.tafel_slope, "mV/dec",
      "| R^2 =", result.tafel_r_squared,
      "| window =", result.tafel_region,
      "| warnings:", result.tafel_warnings)
```

---

## Supported instruments / formats

- Gamry Instruments (`.DTA`)
- BioLogic (`.mpt`, `.mpr`)
- Metrohm Autolab (`.idf`)
- Zahner (`.ism`)
- Generic CSV / TSV

---

## Roadmap (planned — not yet implemented)

The following are research directions, not current features:

- [ ] **EIS-GPT** — a physics-informed Transformer that predicts equivalent-circuit
      topology directly from a spectrum, with Kramers–Kronig / causality / passivity
      constraints in the loss.
- [ ] Pre-trained foundation model and uncertainty quantification for EIS.
- [ ] DRT (distribution of relaxation times) analysis.
- [ ] Zenodo DOI registration.

These items describe intended work; the API and behaviour above are what currently
ships.

---

## License

MIT License — Copyright (c) 2026 Hoda Jafari. Free to use in research and commercial
applications. Please cite this work if you use it in your publications.
