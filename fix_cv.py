from pathlib import Path
candidates = [p for p in Path(".").rglob("cv_analyzer.py") if "venv" not in str(p)]
cv = candidates[0]
lines = cv.read_text(encoding="utf-8").splitlines(keepends=True)
# Remove duplicate substrate block (L829-L847, index 828-846)
del lines[828:847]
cv.write_text("".join(lines), encoding="utf-8")
print("DONE — lines 829-847 removed, cv_analyzer.py fixed")
