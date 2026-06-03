#!/usr/bin/env python3
"""
fix_warn5_final.py — inserts substrate=self.substrate before line 507
Run from EISForge root: python fix_warn5_final.py
"""
import shutil, sys
from pathlib import Path

for candidate in [Path("eisforge/analysis/cv_analyzer.py"), Path("cv_analyzer.py")]:
    if candidate.exists():
        TARGET = candidate; break
else:
    sys.exit("ERROR: cv_analyzer.py not found.")

src = TARGET.read_text(encoding="utf-8")

if "substrate=self.substrate" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

backup = TARGET.with_suffix(".py.bak5")
shutil.copy2(TARGET, backup)

lines = src.splitlines(keepends=True)
new_lines = []
added = False

for line in lines:
    # Insert substrate= just before the interpretation= line inside CVAnalysisResult(...)
    if not added and "interpretation" in line and "self._interpret" in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "substrate          = self.substrate,\n")
        added = True
    new_lines.append(line)

if added:
    TARGET.write_text("".join(new_lines), encoding="utf-8")
    print("OK — substrate=self.substrate inserted before interpretation= (line 507)")
    print(f"Backup: {backup}")
else:
    print("WARN: could not find 'interpretation ... self._interpret' line.")
    print("Open cv_analyzer.py and add this line manually before line 507:")
    print('    substrate          = self.substrate,')
