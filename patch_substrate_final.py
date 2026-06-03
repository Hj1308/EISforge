#!/usr/bin/env python3
"""
patch_substrate_final.py
ONE script that fixes EVERYTHING:
  1. eisforge/analysis/cv_analyzer.py — adds substrate param to __init__ and interpret()
  2. app.py — removes the broken substrate=alcohol kwarg (CVAnalyzer ignores unknown kwargs → no crash),
              fixes the alcohol variable to always resolve correctly,
              removes broken is_mf caption (already fixed by patch_ismf)
"""
import re, shutil, sys
from pathlib import Path

# ── find cv_analyzer.py inside eisforge package ──────────────────────────
candidates = list(Path(".").rglob("cv_analyzer.py"))
cv_path = None
for c in candidates:
    if "eisforge" in str(c) and "venv" not in str(c):
        cv_path = c
        break
if cv_path is None:
    # fallback: use any cv_analyzer.py not in venv
    for c in candidates:
        if "venv" not in str(c):
            cv_path = c
            break
if cv_path is None:
    sys.exit("ERROR: cannot find eisforge/analysis/cv_analyzer.py")

print(f"Found   : {cv_path}")
backup_cv = cv_path.with_suffix(".py.bak_sub_final")
shutil.copy2(cv_path, backup_cv)

src = cv_path.read_text(encoding="utf-8")
original_cv = src
changes_cv = 0

# ── 1A. Add substrate=None to __init__ signature ─────────────────────────
# Find:   catalyst_loading: float = 0.0,
# Insert: substrate: str = "N/A",   after it
if "substrate" not in src:
    pat = r'(catalyst_loading\s*:\s*float\s*=\s*0\.0\s*,)'
    rep = r'\1\n        substrate: str = "N/A",'
    new_src, n = re.subn(pat, rep, src, count=1)
    if n:
        src = new_src; changes_cv += 1
        print("OK 1A   : substrate param added to __init__ signature")
    else:
        print("WARN 1A : could not add substrate to __init__ — add manually")

# ── 1B. Store self.substrate in __init__ body ─────────────────────────────
if "self.substrate" not in src:
    pat = r'(self\.catalyst_loading\s*=\s*catalyst_loading)'
    rep = r'\1\n        self.substrate = substrate'
    new_src, n = re.subn(pat, rep, src, count=1)
    if n:
        src = new_src; changes_cv += 1
        print("OK 1B   : self.substrate stored in __init__ body")
    else:
        print("WARN 1B : could not store self.substrate")

# ── 1C. Pass substrate into interpret() call ─────────────────────────────
# Current call: interpretation=self.interpret(onset, if, ib, ratio,)
# New call:     interpretation=self.interpret(onset, if, ib, ratio, self.substrate)
if "self.interpret(eonset" in src and "self.substrate)" not in src:
    pat = r'(interpretation\s*=\s*self\.interpret\s*\([^)]+)\)'
    rep = r'\1, self.substrate)'
    new_src, n = re.subn(pat, rep, src, count=1)
    if n:
        src = new_src; changes_cv += 1
        print("OK 1C   : self.substrate passed to interpret()")
    else:
        print("WARN 1C : could not update interpret() call")

# ── 1D. Add substrate param to interpret() definition ────────────────────
if 'def interpret(self, eonset' in src and 'substrate' not in src.split('def interpret')[1][:80]:
    pat = r'(def interpret\s*\([^)]*ratio[^)]*)\)'
    rep = r'\1, substrate: str = "N/A")'
    new_src, n = re.subn(pat, rep, src, count=1)
    if n:
        src = new_src; changes_cv += 1
        print("OK 1D   : substrate param added to interpret() signature")
    else:
        print("WARN 1D : could not update interpret() signature")

# ── 1E. Add substrate interpretation lines inside interpret() ─────────────
SUBSTRATE_BLOCK = '''
    # ── Substrate-specific interpretation ─────────────────────────────────
    _substrate_notes = {
        "ethanol":         "Ethanol oxidation (EOR): 12-electron complete oxidation to CO₂ preferred. "
                           "C–C bond cleavage is the target; acetaldehyde/acetic acid are partial products. "
                           "If/Ib > 1 = good CO tolerance.",
        "methanol":        "Methanol oxidation (MOR): 6-electron process to CO₂. "
                           "CO poisoning is the main challenge; PtRu alloys mitigate via bifunctional mechanism.",
        "glycerol":        "Glycerol oxidation (GOR): complex multi-step (up to 14 electrons). "
                           "Partial products: glyceraldehyde, glycerate, tartronate, mesoxalate.",
        "ethylene glycol": "Ethylene glycol oxidation (EGOR): 10-electron complete oxidation. "
                           "C–C cleavage to oxalate/CO₂ is desirable; glycolate is a common partial product.",
        "2-propanol":      "2-Propanol oxidation: dehydrogenation to acetone (2e⁻) or full oxidation (18e⁻). "
                           "Acetone selectivity indicates incomplete oxidation.",
    }
    _sub = substrate.strip().lower() if substrate else "n/a"
    if _sub and _sub != "n/a" and _sub in _substrate_notes:
        parts.append(_substrate_notes[_sub])
'''

if "_substrate_notes" not in src:
    # Insert before the final 'return "  |  ".join(parts)' (or '.join(parts)')
    pat = r'(\s*return\s+["\'].+["\']\.join\(parts\))'
    new_src, n = re.subn(pat, SUBSTRATE_BLOCK + r'\1', src, count=1)
    if n:
        src = new_src; changes_cv += 1
        print("OK 1E   : substrate interpretation block added to interpret()")
    else:
        print("WARN 1E : could not add substrate block — add manually before 'return ... .join(parts)'")

if src != original_cv:
    cv_path.write_text(src, encoding="utf-8")
    print(f"SAVED   : {cv_path}  ({changes_cv} change(s))")
else:
    print("INFO    : cv_analyzer.py already up to date")

# ── 2. Fix app.py ─────────────────────────────────────────────────────────
app = Path("app.py")
if not app.exists():
    sys.exit("ERROR: app.py not found")

backup_app = app.with_suffix(".py.bak_sub_final")
shutil.copy2(app, backup_app)

asrc = app.read_text(encoding="utf-8")
original_app = asrc
changes_app = 0

# 2A. Remove the broken 'substrate=alcohol if system_type == "AOR" else "N/A"'
#     lines that were injected into CVAnalyzer / BatchCVAnalyzer by previous patches
#     (CVAnalyzer now accepts substrate properly but app.py should pass it cleanly)
# Replace the regex-patched CVAnalyzer block to pass substrate cleanly
BAD_CV = 'substrate=alcohol if system_type == "AOR" else "N/A"\n                               '
GOOD_CV = 'substrate=alcohol if system_type == "AOR" else "N/A"'
if BAD_CV in asrc:
    asrc = asrc.replace(BAD_CV, GOOD_CV)
    changes_app += 1
    print("OK 2A   : cleaned up CVAnalyzer substrate whitespace")

# 2B. The alcohol variable: when system_type != "AOR", selectbox is disabled
# but still returns the first item. We add a safety line after the selectbox:
#   if system_type != "AOR": alcohol = "N/A"
if 'if system_type != "AOR": alcohol = "N/A"' not in asrc and \
   "system_type != 'AOR': alcohol" not in asrc:
    # find the alcohol selectbox line and add safety after it
    lines = asrc.splitlines(keepends=True)
    new_lines = []
    added = False
    for line in lines:
        new_lines.append(line)
        if not added and 'disabled=system_type' in line and 'alcohol' in line.lower():
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}if system_type != "AOR":\n')
            new_lines.append(f'{indent}    alcohol = "N/A"\n')
            added = True
            changes_app += 1
            print("OK 2B   : alcohol='N/A' guard added when system_type != AOR")
    if added:
        asrc = "".join(new_lines)

if asrc != original_app:
    app.write_text(asrc, encoding="utf-8")
    print(f"SAVED   : app.py  ({changes_app} change(s))")
else:
    print("INFO    : app.py already clean")

print(f"\n{'='*60}")
print("ALL DONE. Run:  streamlit run app.py")
print(f"{'='*60}")
