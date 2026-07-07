#!/usr/bin/env python3
"""
patch19_citation_v030.py
========================
EISForge — Patch 19: Update CITATION.cff to v0.3.0

Changes made:
  1. version: "0.2.0"  →  "0.3.0"
  2. date-released: "2026-06-18"  →  "2026-07-07"
  3. abstract: add scan-rate kinetics, chronoamperometry,
               robust Huber IRLS fitting, AICc multi-model ranking
  4. keywords: add new terms (scan-rate, chronoamperometry,
                Randles-Sevcik, Huber IRLS, AICc, Tafel analysis)
  5. Add preferred-citation block (best practice for Zenodo + GitHub)

Usage (Windows):
    copy "C:\\Users\\hoda\\Downloads\\patch19_citation_v030.py" "C:\\Users\\hoda\\Desktop\\EISforge\\"
    cd C:\\Users\\hoda\\Desktop\\EISforge
    python patch19_citation_v030.py

Expected:  5 OK
Then:
    git add CITATION.cff
    git add -f patch19_citation_v030.py
    git commit -m "CITATION.cff: v0.3.0 — scan-rate, chrono, robust fit, AICc keywords"
    git push origin main
"""

import re, sys
from pathlib import Path

TARGET = Path("CITATION.cff")

def read():
    return TARGET.read_text(encoding="utf-8")

def write(text):
    TARGET.write_text(text, encoding="utf-8")

def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition

# ── READ ──────────────────────────────────────────────────────────────────────
original = read()
text = original

# ── CHANGE 1: version ─────────────────────────────────────────────────────────
text = re.sub(r'version:\s*"0\.2\.0"', 'version: "0.3.0"', text)

# ── CHANGE 2: date-released ───────────────────────────────────────────────────
text = re.sub(r'date-released:\s*"2026-06-18"', 'date-released: "2026-07-07"', text)

# ── CHANGE 3: abstract (replace whole block) ──────────────────────────────────
new_abstract = (
    "abstract: >\n"
    "  EISForge is an open-source Python framework for automated electrochemical\n"
    "  analysis of the Alcohol Oxidation Reaction (AOR). It combines classical\n"
    "  CNLS fitting (with robust Huber IRLS re-weighting) and Kramers-Kronig\n"
    "  validation, CV/LSV/EIS analysis, ECSA calculation, Koutecky-Levich\n"
    "  analysis, scan-rate kinetics (b-value, Randles-Sevcik linearity),\n"
    "  chronoamperometry stability metrics, AICc multi-model circuit ranking,\n"
    "  rule-based EIS interpretation, a literature-guided knowledge base\n"
    "  (195 peer-reviewed papers), and a Physics-Informed Transformer architecture\n"
    "  (EIS-GPT). Supports Gamry (.dta), BioLogic (.mpt/.mpr), Autolab (.idf),\n"
    "  and generic CSV file formats. Includes a live Streamlit web demo.\n"
)
text = re.sub(
    r"abstract:.*?(?=\nkeywords:)",
    new_abstract,
    text,
    flags=re.DOTALL
)

# ── CHANGE 4: keywords — append new terms ─────────────────────────────────────
extra_keywords = (
    "  - scan-rate kinetics\n"
    "  - chronoamperometry\n"
    "  - Randles-Sevcik analysis\n"
    "  - b-value\n"
    "  - robust fitting\n"
    "  - Huber IRLS\n"
    "  - AICc model selection\n"
    "  - Tafel analysis\n"
    "  - batch reproducibility\n"
)
text = re.sub(
    r"(  - open source\n)",
    r"\1" + extra_keywords,
    text
)

# ── CHANGE 5: preferred-citation block (append before references) ─────────────
preferred_block = (
    "\npreferred-citation:\n"
    "  type: software\n"
    "  title: \"EISForge: A Catalyst-Aware Electrochemistry Analysis Toolkit\"\n"
    "  authors:\n"
    "    - family-names: Jafari\n"
    "      given-names: Hoda\n"
    "      affiliation: \"CCERCI (Chemistry and Chemical Engineering Research Center of Iran)\"\n"
    "  version: \"0.3.0\"\n"
    "  date-released: \"2026-07-07\"\n"
    "  doi: 10.5281/zenodo.20649692\n"
    "  url: \"https://github.com/Hj1308/EISforge\"\n"
    "  repository-code: \"https://github.com/Hj1308/EISforge\"\n"
    "  license: MIT\n"
)
text = re.sub(
    r"\nreferences:",
    preferred_block + "\nreferences:",
    text,
    count=1
)

# ── WRITE ─────────────────────────────────────────────────────────────────────
write(text)
updated = read()

# ── VERIFY ───────────────────────────────────────────────────────────────────
print("\npatch19_citation_v030.py — verification")
print("=" * 45)
all_ok = True
checks = [
    ("version updated to 0.3.0",         'version: "0.3.0"'                 in updated),
    ("date-released updated to 2026-07-07", 'date-released: "2026-07-07"'   in updated),
    ("abstract includes scan-rate",       "scan-rate kinetics"               in updated),
    ("abstract includes Huber IRLS",      "Huber IRLS"                       in updated),
    ("preferred-citation block present",  "preferred-citation:"              in updated),
]
for label, cond in checks:
    ok = check(label, cond)
    all_ok = all_ok and ok

print()
if all_ok:
    print("✅  All 5 checks passed — CITATION.cff updated successfully.")
    print()
    print("Next steps:")
    print("  git add CITATION.cff")
    print("  git add -f patch19_citation_v030.py")
    print('  git commit -m "CITATION.cff: v0.3.0 — scan-rate, chrono, robust fit, AICc keywords"')
    print("  git push origin main")
else:
    print("❌  Some checks FAILED. Review CITATION.cff manually.")
    sys.exit(1)
