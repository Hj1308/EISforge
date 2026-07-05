# patch18_readme_v030.py
# Single combined README update:
#   A) New capabilities: live-demo link, Scan-Rate & Chronoamperometry in the
#      Modules table, Roadmap fixes, fuller Testing table.
#   B) Version bump 0.2.0 -> 0.3.0 (badge, subtitle, BibTeX version).
#   C) DOI: keep the CONCEPT DOI (…20649692) as the citable DOI everywhere
#      (it always resolves to the latest version), and add a line noting the
#      version-specific DOI for v0.3.0 (…21209400).
import shutil, sys

PATH = "README.md"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# ── A1: live-demo badge + link ────────────────────────────────────────────────
(
'''![CI](https://github.com/Hj1308/EISforge/actions/workflows/ci.yml/badge.svg?style=flat-square)''',
'''![CI](https://github.com/Hj1308/EISforge/actions/workflows/ci.yml/badge.svg?style=flat-square)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://eisforge-app.streamlit.app/)

**\U0001F310 Try it now (no install):** [eisforge-app.streamlit.app](https://eisforge-app.streamlit.app/)'''
),
# ── B1: version badge ─────────────────────────────────────────────────────────
(
'''![Version](https://img.shields.io/badge/version-v0.2.0-blue?style=flat-square)''',
'''![Version](https://img.shields.io/badge/version-v0.3.0-blue?style=flat-square)'''
),
# ── B2: subtitle version ──────────────────────────────────────────────────────
(
'''**EISForge — Catalyst-Aware Electrochemistry Analysis Toolkit, v0.2.0**''',
'''**EISForge — Catalyst-Aware Electrochemistry Analysis Toolkit, v0.3.0**'''
),
# ── A2: modules table rows ────────────────────────────────────────────────────
(
'''| **EIS–CV Correlator** | Cross-technique | R_ct ↔ peak current correlation, activity–impedance maps |''',
'''| **EIS–CV Correlator** | Cross-technique | R_ct ↔ peak current correlation, activity–impedance maps |
| **Scan-Rate Kinetics** | Multi-rate CV | b-value (log I – log ν), Randles–Ševčík linearity, mechanism (diffusion/adsorption/mixed) |
| **Chronoamperometry** | i–t hold | Current retention (%), steady-state current, initial drop — descriptive stability |'''
),
# ── A3: roadmap ───────────────────────────────────────────────────────────────
(
'''| Rule-based EIS interpretation | ✅ Implemented |
| Excel export of EIS results | ✅ Implemented |
| Pre-trained EIS-GPT model weights | 🔬 In development (v0.4) |
| Chronoamperometry (i–t stability) tab | 📋 Planned |
| DRT (distribution of relaxation times) | 📋 Planned |
| PyPI package release | 📋 Planned |''',
'''| Rule-based EIS interpretation | ✅ Implemented |
| Excel export of EIS results | ✅ Implemented |
| Scan-rate kinetics (b-value, Randles–Ševčík) | ✅ Implemented |
| Chronoamperometry (i–t stability) tab | ✅ Implemented |
| Live web demo (Streamlit Cloud) | ✅ Implemented |
| Pre-trained EIS-GPT model weights | 🔬 In development (v0.4) |
| DRT (distribution of relaxation times) | 📋 Planned |
| PyPI package release | 📋 Planned |'''
),
# ── A4: testing table ─────────────────────────────────────────────────────────
(
'''| Test File | Covers |
|---|---|
| `tests/test_eis_fitting.py` | EIS CNLS fitting & Kramers–Kronig |
| `tests/test_ecsa.py` | ECSA / double-layer capacitance |
| `tests/test_batch_analyzer.py` | Batch statistics & reproducibility |
| `tests/test_parsers.py` | Multi-instrument file parsers |''',
'''| Test File | Covers |
|---|---|
| `tests/test_eis_fitting.py` | EIS CNLS fitting & Kramers–Kronig |
| `tests/test_eis_interpreter.py` | Rule-based EIS interpretation |
| `tests/test_suggestion_engine.py` | AICc multi-model circuit ranking |
| `tests/test_inductive_aor.py` | Pseudo-inductive / NDR circuits |
| `tests/test_scan_rate_analyzer.py` | Scan-rate kinetics (b-value, Randles–Ševčík) |
| `tests/test_ca_analyzer.py` | Chronoamperometry stability metrics |
| `tests/test_ecsa.py` | ECSA / double-layer capacitance |
| `tests/test_batch_analyzer.py` | Batch statistics & reproducibility |
| `tests/test_parsers.py` | Multi-instrument file parsers |'''
),
# ── C1: BibTeX version ────────────────────────────────────────────────────────
(
'''  version      = {0.2.0},''',
'''  version      = {0.3.0},'''
),
# ── C2: DOI line — keep concept DOI, note version DOI ─────────────────────────
(
'''**DOI:** [10.5281/zenodo.20649692](https://doi.org/10.5281/zenodo.20649692)''',
'''**DOI (all versions — cite this):** [10.5281/zenodo.20649692](https://doi.org/10.5281/zenodo.20649692)  
**DOI (this release, v0.3.0):** [10.5281/zenodo.21209400](https://doi.org/10.5281/zenodo.21209400)'''
),
]

n = 0
for i, (old, new) in enumerate(PATCHES, 1):
    if new in s:
        print(f"[{i}/{len(PATCHES)}] already applied")
        continue
    if old not in s:
        print(f"ERROR step {i}: OLD block not found. Aborting.")
        sys.exit(1)
    s = s.replace(old, new, 1)
    n += 1
    print(f"[{i}/{len(PATCHES)}] OK")

if n:
    shutil.copy(PATH, PATH + ".bak_patch18")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH)
else:
    print("Nothing to do.")
