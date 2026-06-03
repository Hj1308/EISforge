#!/usr/bin/env python3
"""
apply_cv_substrate_patch_v3.py
Fixes WARN 2, WARN 5, WARN 6 left over from v2.
Run from EISForge root: python apply_cv_substrate_patch_v3.py
"""

import re, shutil, sys
from pathlib import Path

for candidate in [
    Path("eisforge/analysis/cv_analyzer.py"),
    Path("cv_analyzer.py"),
]:
    if candidate.exists():
        TARGET = candidate
        break
else:
    sys.exit("ERROR: cv_analyzer.py not found.")

backup = TARGET.with_suffix(".py.bak3")
shutil.copy2(TARGET, backup)
print(f"Backup saved : {backup}")
print(f"Patching     : {TARGET}\n")

src = TARGET.read_text(encoding="utf-8")
original_src = src

# ── PATCH 2: substrate param in __init__ ────────────────────────────────
if 'substrate' in src and re.search(r'def __init__.*?substrate', src, re.DOTALL):
    print("SKIP 2 : substrate already in __init__")
else:
    new_lines = []
    added2 = False
    for line in src.splitlines(keepends=True):
        new_lines.append(line)
        if not added2 and re.search(r'catalystloading', line) and re.search(r'[=:]', line):
            # This is the catalystloading parameter line — add substrate after
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}substrate: str = "N/A",\n')
            added2 = True
    if added2:
        src = "".join(new_lines)
        print("OK 2   : substrate param inserted after catalystloading")
    else:
        print("WARN 2 : could not add substrate param — do manually")

# ── PATCH 3b: self.substrate = substrate in body ─────────────────────────
if 'self.substrate' not in src:
    new_lines = []
    added3 = False
    for line in src.splitlines(keepends=True):
        new_lines.append(line)
        if not added3 and re.search(r'self\.catalystloading\s*=\s*catalystloading', line):
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}self.substrate = substrate\n')
            added3 = True
    if added3:
        src = "".join(new_lines)
        print("OK 3b  : self.substrate = substrate added")
    else:
        print("WARN 3b: add self.substrate = substrate after self.catalystloading manually")
else:
    print("SKIP 3 : self.substrate already present")

# ── PATCH 5: substrate=self.substrate before interpretation= ────────────
if 'substrate=self.substrate' in src:
    print("SKIP 5 : already present")
else:
    new_lines = []
    added5 = False
    for line in src.splitlines(keepends=True):
        if not added5 and re.search(r'interpretation\s*=\s*self\.interpret', line):
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}substrate=self.substrate,\n')
            added5 = True
        new_lines.append(line)
    if added5:
        src = "".join(new_lines)
        print("OK 5   : substrate=self.substrate added before interpretation=")
    else:
        print("WARN 5 : add substrate=self.substrate manually in return CVAnalysisResult(...)")

# ── PATCH 6: substrate interpret block ───────────────────────────────────
INTERP_LINES = [
    '        # --- Substrate-specific interpretation ---\n',
    '        _sub = getattr(self, "substrate", "N/A")\n',
    '        if _sub and _sub.strip().lower() not in ("n/a", "na", "none", ""):\n',
    '            _sub_notes = {\n',
    '                "ethanol": (\n',
    '                    "Ethanol oxidation (EOR): 12-electron complete oxidation to CO2 preferred. "\n',
    '                    "IfIb > 1 = good CO tolerance; low ratio = acetaldehyde/acetic acid partial products."\n',
    '                ),\n',
    '                "methanol": (\n',
    '                    "Methanol oxidation (MOR): 6-electron process to CO2. "\n',
    '                    "CO poisoning is the main challenge; PtRu alloys mitigate via bifunctional mechanism."\n',
    '                ),\n',
    '                "2-propanol": (\n',
    '                    "2-Propanol oxidation (IOR): cleaner 2-electron pathway to acetone, minimal CO poisoning. "\n',
    '                    "Lower onset vs ethanol; monitor acetone accumulation."\n',
    '                ),\n',
    '                "ethylene glycol": (\n',
    '                    "Ethylene glycol oxidation (EGOR): up to 10 electrons. "\n',
    '                    "Partial products: glycolaldehyde, glycolic acid, oxalic acid."\n',
    '                ),\n',
    '                "ethylene-glycol": (\n',
    '                    "Ethylene glycol oxidation (EGOR): up to 10 electrons. "\n',
    '                    "Partial products: glycolaldehyde, glycolic acid, oxalic acid."\n',
    '                ),\n',
    '                "glycerol": (\n',
    '                    "Glycerol oxidation (GOR): complex multi-step (up to 14 electrons). "\n',
    '                    "Partial products: glyceraldehyde, glycerate, tartronate, mesoxalate."\n',
    '                ),\n',
    '            }\n',
    '            _note = _sub_notes.get(_sub.strip().lower())\n',
    '            if _note:\n',
    '                parts.append(_note)\n',
]

if '_sub_notes' in src:
    print("SKIP 6 : substrate interpret block already present")
else:
    new_lines = []
    added6 = False
    in_interpret = False
    for line in src.splitlines(keepends=True):
        # Track being inside interpret()
        if re.search(r'def\s+interpret\s*\(', line):
            in_interpret = True
        elif in_interpret and re.match(r'    def \w', line):
            in_interpret = False

        # Insert block before return join(parts) while inside interpret
        if in_interpret and not added6 and re.search(r'return\s+.*join\s*\(\s*parts\s*\)', line):
            new_lines.extend(INTERP_LINES)
            added6 = True

        new_lines.append(line)

    if not added6:
        # Broad fallback: any return join(parts)
        new_lines2 = []
        for line in src.splitlines(keepends=True):
            if not added6 and re.search(r'return\s+.*join\s*\(\s*parts\s*\)', line):
                new_lines2.extend(INTERP_LINES)
                added6 = True
            new_lines2.append(line)
        if added6:
            new_lines = new_lines2

    if added6:
        src = "".join(new_lines)
        print("OK 6   : substrate interpret block added before return join(parts)")
    else:
        print("WARN 6 : could not find return join(parts) — paste block manually")

# ── Write ─────────────────────────────────────────────────────────────────
if src == original_src:
    print("\nNo changes written.")
else:
    TARGET.write_text(src, encoding="utf-8")
    print(f"\nv3 PATCH COMPLETE — {TARGET} updated.")
    print(f"Backup: {backup}")
