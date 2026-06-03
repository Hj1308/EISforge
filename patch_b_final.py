#!/usr/bin/env python3
"""
patch_b_final.py
Fixes:
  WARN B – BatchCVAnalyzer: insert substrate= after r_s_ohms=actual_rs on L379
  INFO C – add 🧪 caption after c4.metric("C_dl", ...) on L311
Run from EISForge root: python patch_b_final.py
"""
import shutil, sys
from pathlib import Path

TARGET = Path("app.py")
if not TARGET.exists():
    sys.exit("ERROR: app.py not found.")

backup = TARGET.with_suffix(".py.bak_sub3")
shutil.copy2(TARGET, backup)
print(f"Backup  : {backup}")

lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)
changes = 0

new_lines = []
for i, line in enumerate(lines, start=1):

    # ── PATCH B: L379 — after r_s_ohms=actual_rs, in BatchCVAnalyzer block ──
    # Insert new line with substrate= BEFORE the closing paren on L380
    if i == 379 and "r_s_ohms=actual_rs," in line:
        # Keep this line as-is, then insert substrate line after
        new_lines.append(line)
        indent = "                        "   # 24 spaces, same as sibling args
        new_lines.append(
            f'{indent}substrate=alcohol if system_type == "AOR" else "N/A",\n'
        )
        changes += 1
        print("OK B    : substrate= inserted after L379 (r_s_ohms=actual_rs)")
        continue

    # ── PATCH C: L311 — after c4.metric("C_dl", ...) ────────────────────────
    if i == 311 and 'c4.metric("C_dl"' in line:
        new_lines.append(line)
        indent = "            "  # 12 spaces (same as c4.metric indent)
        new_lines.append(
            f'{indent}if is_mf:\n'
            f'{indent}    st.caption("🧪 Carbon material: ECSA via C\u1d05\u2097 method. '
            f'No I\u209f/I\u2099 — CO poisoning pathway absent.")\n'
        )
        changes += 1
        print("OK C    : Cdl caption inserted after L311 (c4.metric C_dl)")
        continue

    new_lines.append(line)

if changes:
    TARGET.write_text("".join(new_lines), encoding="utf-8")
    print(f"\nDone    : {changes} patch(es) applied → app.py updated.")
else:
    print("\nWARN    : No lines matched. Check line numbers manually.")
    print("          L379 should contain: r_s_ohms=actual_rs,")
    print("          L311 should contain: c4.metric(\"C_dl\"")
