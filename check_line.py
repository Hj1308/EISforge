from pathlib import Path
import re

# find cv_analyzer.py
candidates = [p for p in Path(".").rglob("cv_analyzer.py") if "venv" not in str(p)]
if not candidates:
    print("NOT FOUND"); exit()
cv = candidates[0]
lines = cv.read_text(encoding="utf-8").splitlines()
print(f"Total lines: {len(lines)}")
# show lines 835-860
for i in range(834, min(862, len(lines))):
    print(f"L{i+1:4d}: {lines[i]}")
