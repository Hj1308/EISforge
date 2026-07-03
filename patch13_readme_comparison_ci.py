# patch13_readme_comparison_ci.py
# Level-4 documentation tasks:
#   1) README: EIS section updated with the new capabilities (robust Huber
#      fitting, smooth ZView-style fit curve, AICc multi-model suggestion,
#      rule-based interpretation, Excel export, C_dl from EIS, iR auto-link).
#   2) README: comparison table vs other open-source EIS tools
#      (impedance.py, pyimpspec, DearEIS) inserted before "Supported
#      Instruments".
#   3) README: Roadmap rows updated.
#   4) Removes the duplicate CI workflow: keeps .github/workflows/ci.yml
#      (fast, no torch) and deletes tests.yml (3-version matrix with full
#      torch install — redundant and slow). One green workflow is enough.
import os, shutil, sys

PATH = r"README.md"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# ── 1. EIS section: new capabilities ─────────────────────────────────────────
(
"""## \U0001F52C Electrochemical Impedance Spectroscopy (EIS)

- Equivalent-circuit (CNLS) fitting via `impedance` library
- **Kramers\u2013Kronig validation** (causality, linearity, stationarity)
- Charge-transfer resistance (R_ct), solution resistance (R\u209b)
- CPE and double-layer capacitance extraction
- Interactive Nyquist and Bode plots (Plotly)
- EIS\u2013CV cross-technique correlation""",
"""## \U0001F52C Electrochemical Impedance Spectroscopy (EIS)

- Equivalent-circuit (CNLS) fitting via `impedance` library
- **Robust fitting** \u2014 optional Huber IRLS re-weighting (`robust=True`):
  stray points are automatically down-weighted, ZView-like tolerance
  without deleting data
- **Smooth fit overlay** \u2014 fitted model rendered on 400 log-spaced
  frequencies (continuous curve, not segments between data points)
- **Multi-model circuit suggestion** \u2014 candidate circuits fitted and
  ranked by AICc (Burnham & Anderson convention)
- **Rule-based physical interpretation** \u2014 deterministic report of R\u209b,
  per-arc effective capacitance (Brug), time constants, NDR /
  pseudo-inductive / Warburg fingerprints, and fit-quality assessment
- **C_dl from EIS** \u2014 per-area effective capacitance (\u03bcF/cm\u00b2) for direct
  comparison with multi-scan-rate CV values
- **Excel export** \u2014 multi-sheet .xlsx (Summary, Fit_Parameters, Data,
  Fit_Curve) ready for Origin/Excel manuscript figures
- **Kramers\u2013Kronig validation** (causality, linearity, stationarity)
- Charge-transfer resistance (R_ct), solution resistance (R\u209b) \u2014 R\u2080 is
  auto-linked to the iR-compensation field in the CV/LSV tabs
- Low-frequency **pseudo-inductive / NDR** circuit topologies for alcohol
  electrooxidation (adsorbed-intermediate relaxation), incl. negative
  faradaic resistance fitting
- Interactive Nyquist and Bode plots (Plotly)
- EIS\u2013CV cross-technique correlation""",
),
# ── 2. comparison table before Supported Instruments ─────────────────────────
(
"""## \U0001F50C Supported Instruments / Formats""",
"""## \u2696 Comparison with Other Open-Source EIS Tools

| Capability | **EISForge** | impedance.py | pyimpspec | DearEIS |
|---|---|---|---|---|
| CNLS equivalent-circuit fitting | \u2705 | \u2705 | \u2705 | \u2705 |
| Kramers\u2013Kronig validation | \u2705 | \u2705 | \u2705 | \u2705 |
| Robust (outlier-tolerant) fitting | \u2705 Huber IRLS | \u274c | \u274c | \u274c |
| Multi-model ranking (AICc) | \u2705 | \u274c | \u26a0 (\u03c7\u00b2 only) | \u26a0 |
| Rule-based physical interpretation | \u2705 | \u274c | \u274c | \u274c |
| CV / LSV / Tafel in the same tool | \u2705 | \u274c | \u274c | \u274c |
| Koutecky\u2013Levich analysis | \u2705 | \u274c | \u274c | \u274c |
| AOR pseudo-inductive / NDR circuits | \u2705 | \u26a0 manual | \u26a0 manual | \u26a0 manual |
| Web UI (no code required) | \u2705 Streamlit | \u274c | \u274c | \u2705 desktop |
| Ivium IDF parser | \u2705 | \u274c | \u274c | \u274c |
| Excel export of fits | \u2705 | \u274c | \u26a0 CSV | \u2705 |

*impedance.py is used internally by EISForge for circuit evaluation \u2014 the
comparison refers to what each tool offers out of the box. pyimpspec and
DearEIS additionally offer DRT analysis, which is on the EISForge roadmap.*

---

## \U0001F50C Supported Instruments / Formats""",
),
# ── 3. roadmap refresh ─────────────────────────────────────────────────────────
(
"""| EIS-GPT Transformer model architecture | \u2705 Implemented |
| Pre-trained EIS-GPT model weights | \U0001F52C In development |
| DRT (distribution of relaxation times) | \U0001F4CB Planned |
| PyPI package release | \U0001F4CB Planned |""",
"""| EIS-GPT Transformer model architecture | \u2705 Implemented |
| Robust (Huber IRLS) CNLS fitting | \u2705 Implemented |
| AICc multi-model circuit suggestion | \u2705 Implemented |
| Rule-based EIS interpretation | \u2705 Implemented |
| Excel export of EIS results | \u2705 Implemented |
| Pre-trained EIS-GPT model weights | \U0001F52C In development (v0.4) |
| Chronoamperometry (i\u2013t stability) tab | \U0001F4CB Planned |
| DRT (distribution of relaxation times) | \U0001F4CB Planned |
| PyPI package release | \U0001F4CB Planned |""",
),
]

n_applied = 0
for i, (old, new) in enumerate(PATCHES, 1):
    if new in s:
        print(f"[{i}/{len(PATCHES)}] already applied, skipping")
        continue
    if old not in s:
        print(f"ERROR at step {i}: OLD block not found. Aborting, no changes written.")
        sys.exit(1)
    s = s.replace(old, new, 1)
    n_applied += 1
    print(f"[{i}/{len(PATCHES)}] OK")

if n_applied:
    shutil.copy(PATH, PATH + ".bak_patch13")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: README.md.bak_patch13)")

# ── 4. remove duplicate workflow ───────────────────────────────────────────────
WF = os.path.join(".github", "workflows", "tests.yml")
if os.path.exists(WF):
    os.remove(WF)
    print("Removed duplicate workflow:", WF, "(ci.yml is kept)")
else:
    print("tests.yml already absent.")

print("\nNOTE: after this patch run:  git rm --cached .github/workflows/tests.yml")
