#!/usr/bin/env python3
"""
patch_app_b.py  — fixes WARN B (BatchCVAnalyzer substrate) and WARN C (Cdl caption)
Run from EISForge root: python patch_app_b.py
"""
import re, shutil, sys
from pathlib import Path

TARGET = Path("app.py")
if not TARGET.exists():
    sys.exit("ERROR: app.py not found.")

backup = TARGET.with_suffix(".py.bak_sub2")
shutil.copy2(TARGET, backup)

src = TARGET.read_text(encoding="utf-8")
original = src
changes = 0

# ── PATCH B: BatchCVAnalyzer substrate ───────────────────────────────────
# Show all lines near BatchCVAnalyzer to find the right anchor
print("--- BatchCVAnalyzer context ---")
lines = src.splitlines(keepends=True)
for i, line in enumerate(lines):
    if "BatchCVAnalyzer" in line or ("batch" in line.lower() and "actual_rs" in line):
        start = max(0, i-2)
        end = min(len(lines), i+8)
        for j in range(start, end):
            print(f"  L{j+1:4d}: {repr(lines[j].rstrip())}")
        print("  ---")

# ── PATCH C: find Cdl metric lines ───────────────────────────────────────
print("\n--- Lines with 'cdl' or 'Cdl' ---")
for i, line in enumerate(lines):
    if 'cdl' in line.lower() and ('metric' in line.lower() or 'mf' in line.lower()):
        print(f"  L{i+1:4d}: {repr(line.rstrip())}")
