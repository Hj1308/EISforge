#!/usr/bin/env python3
"""
patch_app_substrate.py
Patches app.py to pass substrate= to CVAnalyzer and BatchCVAnalyzer,
and adds carbon-material-specific metric labels.
Run from EISForge root: python patch_app_substrate.py
"""
import re, shutil, sys
from pathlib import Path

TARGET = Path("app.py")
if not TARGET.exists():
    sys.exit("ERROR: app.py not found. Run from EISForge root.")

backup = TARGET.with_suffix(".py.bak_sub")
shutil.copy2(TARGET, backup)
print(f"Backup: {backup}")

src = TARGET.read_text(encoding="utf-8")
original = src
changes = 0

# ── PATCH A: CVAnalyzer(...) — add substrate=alcohol ────────────────────
# Current code (from file):
#   ana = CVAnalyzer(scan_rate=srcv, electrode_area=area, ecsa=ecsa,
#                   catalyst_loading=loading, onset_method=om,
#                   electrolyte=el, catalyst_type=catalyst_type)
# We find the CVAnalyzer(...) constructor call and add substrate=alcohol

if "substrate=alcohol" in src:
    print("SKIP A : CVAnalyzer already has substrate=alcohol")
else:
    # Match CVAnalyzer( ... catalyst_type=catalyst_type) and add substrate=
    # The closing paren follows catalyst_type=catalyst_type
    pat = r'(ana\s*=\s*CVAnalyzer\s*\([^)]*?catalyst_type\s*=\s*catalyst_type\s*)'
    def add_substrate_cv(m):
        txt = m.group(1)
        # Insert substrate=alcohol before the closing paren region
        return txt.rstrip() + ',\n                               substrate=alcohol if system_type == "AOR" else "N/A"\n                               '
    new_src, n = re.subn(pat, add_substrate_cv, src, count=1, flags=re.DOTALL)
    if n == 0:
        # Line-by-line fallback: find "catalyst_type=catalyst_type" INSIDE CVAnalyzer block
        lines = src.splitlines(keepends=True)
        new_lines = []
        in_ana_block = False
        added_a = False
        for line in lines:
            if not added_a and 'ana = CVAnalyzer(' in line:
                in_ana_block = True
            if in_ana_block and not added_a and 'catalyst_type=catalyst_type' in line:
                indent = re.match(r'^(\s*)', line).group(1)
                # add substrate after this line
                new_lines.append(line.rstrip('\n').rstrip(',') + ',\n')
                new_lines.append(f'{indent}substrate=alcohol if system_type == "AOR" else "N/A")\n')
                # remove original closing paren if next line has it
                in_ana_block = False
                added_a = True
                changes += 1
                continue
            new_lines.append(line)
        if added_a:
            src = "".join(new_lines)
            print("OK A   : substrate=alcohol added to CVAnalyzer (line-by-line)")
        else:
            print("WARN A : Could not patch CVAnalyzer — do manually:\n"
                  "         Add  substrate=alcohol if system_type=='AOR' else 'N/A',\n"
                  "         after catalyst_type=catalyst_type in the CVAnalyzer(...) call")
    else:
        src = new_src
        changes += 1
        print("OK A   : substrate=alcohol added to CVAnalyzer (regex)")

# ── PATCH B: BatchCVAnalyzer(...) — add substrate=alcohol ───────────────
if "substrate=alcohol" in src and src.count("substrate=alcohol") >= 2:
    print("SKIP B : BatchCVAnalyzer already has substrate=alcohol")
else:
    lines = src.splitlines(keepends=True)
    new_lines = []
    added_b = False
    in_batch = False
    for line in lines:
        if not added_b and 'BatchCVAnalyzer(' in line:
            in_batch = True
        if in_batch and not added_b and 'rsohms=actual_rs' in line:
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(line.rstrip('\n').rstrip(',') + ',\n')
            new_lines.append(f'{indent}substrate=alcohol if system_type == "AOR" else "N/A",\n')
            in_batch = False
            added_b = True
            changes += 1
            continue
        new_lines.append(line)
    if added_b:
        src = "".join(new_lines)
        print("OK B   : substrate=alcohol added to BatchCVAnalyzer")
    else:
        print("WARN B : Could not patch BatchCVAnalyzer — add substrate= manually after rsohms=actual_rs")

# ── PATCH C: Carbon Material specific UI labels ──────────────────────────
# In the Cdl metric display, add a helpful caption for metal-free
# Find:  c4.metric("Cdl", ...)  and add a note below it
if "ECSA via C_dl" in src or "C_dl method" in src:
    print("SKIP C : Carbon material note already present")
else:
    lines = src.splitlines(keepends=True)
    new_lines = []
    added_c = False
    for line in lines:
        new_lines.append(line)
        if not added_c and 'cdl_mFcm2' in line and 'metric' in line and 'mF' in line:
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(
                f'{indent}if is_mf:\n'
                f'{indent}    st.caption("🧪 Carbon material: ECSA via C_dl method. '
                f'No IfIb — CO poisoning pathway absent.")\n'
            )
            added_c = True
            changes += 1
    if added_c:
        src = "".join(new_lines)
        print("OK C   : Carbon material Cdl caption added")
    else:
        print("INFO C : Cdl metric line not found — skipped (optional cosmetic patch)")

# ── Write ────────────────────────────────────────────────────────────────
if src == original:
    print("\nNo changes written.")
else:
    TARGET.write_text(src, encoding="utf-8")
    print(f"\napp.py PATCH COMPLETE ({changes} change(s) applied)")
    print(f"Backup: {backup}")
