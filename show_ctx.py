from pathlib import Path
candidates = [p for p in Path(".").rglob("cv_analyzer.py") if "venv" not in str(p)]
cv = candidates[0]
print("File:", cv)
lines = cv.read_text(encoding="utf-8").splitlines(keepends=True)
for i in range(799, min(862, len(lines))):
    print(f"L{i+1:4d}|{lines[i]}", end="")
