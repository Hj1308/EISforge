from pathlib import Path, shutil
import shutil

candidates = [p for p in Path(".").rglob("cv_analyzer.py") if "venv" not in str(p)]
cv = candidates[0]
shutil.copy2(cv, cv.with_suffix(".py.bak_return"))

lines = cv.read_text(encoding="utf-8").splitlines(keepends=True)

# L847 (index 846): '        return " | ".join(parts)' — wrong indent (8 spaces inside if block)
# Should be:        '    return " | ".join(parts)'      — 4 spaces (function level)
idx = 846  # 0-based
if 'return' in lines[idx] and 'join(parts)' in lines[idx]:
    lines[idx] = '    return " | ".join(parts)\n'
    cv.write_text("".join(lines), encoding="utf-8")
    print(f"FIXED L847: return dedented to function level")
else:
    print(f"L847 content: {repr(lines[idx])}")
    print("Not matched — searching...")
    for i, l in enumerate(lines):
        if 'return' in l and 'join(parts)' in l:
            print(f"  Found at L{i+1}: {repr(l)}")
            lines[i] = '    return " | ".join(parts)\n'
            cv.write_text("".join(lines), encoding="utf-8")
            print(f"  FIXED L{i+1}")
            break
