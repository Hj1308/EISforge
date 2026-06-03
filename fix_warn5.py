#!/usr/bin/env python3
"""
fix_warn5.py  — adds substrate=self.substrate to CVAnalysisResult(...)
Run from EISForge root: python fix_warn5.py
"""
import re, shutil, sys
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
print(f"Backup: {backup}")

# Show all lines that contain 'interpretation' to understand the format
print("\n--- Lines containing 'interpretation' in the file ---")
for i, line in enumerate(src.splitlines(), 1):
    if "interpretation" in line.lower():
        print(f"  L{i:4d}: {repr(line)}")

print("\n--- Lines containing 'catalystloading' in the file ---")
for i, line in enumerate(src.splitlines(), 1):
    if "catalystloading" in line.lower() and "=" in line:
        print(f"  L{i:4d}: {repr(line)}")
