#!/usr/bin/env python3
"""
patch_ismf.py — fixes NameError: 'is_mf' not defined at L312
The caption block needs to be inside the 'if cv_r in st.session_state:' block,
properly indented. We replace the raw 'if is_mf:' block with a safe inline check.
"""
import shutil, sys
from pathlib import Path

TARGET = Path("app.py")
if not TARGET.exists():
    sys.exit("ERROR: app.py not found.")

backup = TARGET.with_suffix(".py.bak_ismf")
shutil.copy2(TARGET, backup)

src = TARGET.read_text(encoding="utf-8")

# Replace the two lines added by patch_b_final for Patch C
# Old (wrong indentation / out of scope):
#             if is_mf:
#                 st.caption("🧪 Carbon material: ...")
# New: safe one-liner using catalyst_type directly (always in scope)
OLD = (
    '            if is_mf:\n'
    '                st.caption("🧪 Carbon material: ECSA via C\u1d05\u2097 method. '
    'No I\u209f/I\u2099 \u2014 CO poisoning pathway absent.")\n'
)
NEW = (
    '            if catalyst_type == "carbonmaterial":\n'
    '                st.caption("🧪 Carbon material: ECSA via C_dl method. '
    'No If/Ib \u2014 CO poisoning pathway absent.")\n'
)

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    TARGET.write_text(src, encoding="utf-8")
    print("OK : is_mf replaced with catalyst_type == 'carbonmaterial'")
    print(f"Backup: {backup}")
else:
    # Fallback: find and fix by line number
    lines = src.splitlines(keepends=True)
    fixed = False
    for i, line in enumerate(lines):
        if 'if is_mf:' in line and i+1 < len(lines) and 'C_dl' in lines[i+1]:
            indent = '            '
            lines[i]   = f'{indent}if catalyst_type == "carbonmaterial":\n'
            lines[i+1] = (f'{indent}    st.caption("🧪 Carbon material: '
                          f'ECSA via C_dl method. No If/Ib \u2014 CO poisoning pathway absent.")\n')
            fixed = True
            print(f"OK : Fixed at line {i+1} via fallback")
            break
    if fixed:
        TARGET.write_text("".join(lines), encoding="utf-8")
        print(f"Backup: {backup}")
    else:
        print("WARN: Pattern not found. Fix manually:")
        print("  Replace:  if is_mf:")
        print("  With:     if catalyst_type == 'carbonmaterial':")
