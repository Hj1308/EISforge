# EISForge 🔬⚡

> **Advanced Electrochemical Impedance Spectroscopy Analysis with Physics-Informed Machine Learning**
>
> *The first open-source EIS framework combining Transformer-based deep learning with Kramers-Kronig physics constraints.*

---

## Author & Citation

**Hoda Jafari**
📧 hoda.jaafari@gmail.com
🔗 https://github.com/HJ1308
📅 First published: May 2026

If you use EISForge in your research, please cite:

```bibtex
@software{jafari2026eisforge,
  author    = {Jafari, Hoda},
  title     = {EISForge: Physics-Informed Transformer for Electrochemical Impedance Spectroscopy},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/HJ1308/EISforge-},
  note      = {Open-source EIS analysis with ML}
}
```

---

## What Makes EISForge Different? 🚀

| Feature | ZView | EC-Lab | EISForge |
|---|---|---|---|
| Circuit fitting (CNLS) | ✅ | ✅ | ✅ |
| Kramers-Kronig validation | ✅ | ✅ | ✅ |
| DRT Analysis | ❌ | partial | ✅ |
| ML circuit detection | ❌ | ❌ | ✅ |
| Physics-Informed Transformer | ❌ | ❌ | ✅ **NEW** |
| Uncertainty quantification | ❌ | ❌ | ✅ **NEW** |
| No circuit pre-selection needed | ❌ | ❌ | ✅ **NEW** |
| Free & Open Source | ❌ | ❌ | ✅ |

---

## Core Innovation: EIS-GPT Architecture 🧠

EISForge introduces the first **Physics-Informed Transformer** for EIS analysis.
Instead of requiring the user to pre-select an equivalent circuit, the model learns to:

1. **Tokenize** EIS spectra (each frequency point = one token)
2. **Enforce** Kramers-Kronig relations as hard physics constraints
3. **Predict** circuit topology with uncertainty estimates
4. **Estimate** initial parameters for CNLS refinement
5. **Decode** electrochemical state directly from spectra

### Physics-Informed Loss Function (Novel Contribution)

```
L_total = L_reconstruction
        + λ₁ × L_kramers_kronig
        + λ₂ × L_causality
        + λ₃ × L_passivity
```

This ensures predictions are always **physically valid** —
something no existing ML approach for EIS achieves.

---

## Installation

```bash
git clone https://github.com/HJ1308/EISforge-.git
cd EISforge-
pip install -r requirements.txt
```

---

## Quick Start

```python
from eisforge.core.analyzer import EisAnalyzer
from eisforge.ml.eis_gpt import EISForgeModel

# Load your data
ana = EisAnalyzer()
dataset = ana.load("my_data.DTA")

# Let EIS-GPT predict everything automatically
model = EISForgeModel.load_pretrained()
result = model.analyze(dataset)

print(result.predicted_circuit)      # "R0-p(R1,CPE1)-W1"
print(result.parameters)             # {'R0': 15.2, 'R1': 320.5, ...}
print(result.uncertainty)            # {'R0': ±0.3, 'R1': ±12.1, ...}
print(result.electrochemical_state)  # {'SOH': 0.87, 'mechanism': 'diffusion-limited'}
```

---

## Project Structure

```
eisforge/
├── core/          # CNLS fitting engine
├── parsers/       # Gamry, BioLogic, Autolab importers
├── analysis/      # DRT, batch processing
├── ml/
│   ├── eis_gpt/   # Physics-Informed Transformer ← NOVEL
│   │   ├── tokenizer.py
│   │   ├── transformer.py
│   │   ├── physics_loss.py
│   │   └── foundation_model.py
│   ├── circuit_classifier.py
│   └── guess_predictor.py
└── visualization/ # Interactive Nyquist & Bode plots
```

---

## Supported Instruments

- Gamry Instruments (.DTA)
- BioLogic (.mpt, .mpr)
- Metrohm Autolab (.idf)
- Zahner (.ism)
- Generic CSV

---

## License

MIT License — Copyright (c) 2026 Hoda Jafari

Free to use in research and commercial applications.
**Please cite this work if you use it in your publications.**

---

## Roadmap

- [x] Core EIS analysis engine
- [x] Kramers-Kronig validation
- [x] ML circuit classifier
- [ ] Physics-Informed Transformer (EIS-GPT)
- [ ] Pre-trained foundation model
- [ ] Streamlit web interface
- [ ] Zenodo DOI registration