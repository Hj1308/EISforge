# patch14_repo_rename_urls.py
# Run AFTER renaming the GitHub repository EISforge- -> EISforge.
# Updates every hard-coded old URL / name across docs, metadata, and app
# footer, and removes the dead "Tests" badge (its workflow tests.yml was
# deleted in patch 13).
#
# NOTE: .gitignore is intentionally NOT touched — its "EISforge-/" entry
# refers to the stale local duplicate FOLDER on disk, not the GitHub URL.
import shutil, sys

# (file, old, new) — applied only where found; count reported at the end.
REPLACEMENTS = [
    # dead badge line (workflow removed in patch 13) — delete entirely
    ("README.md",
     "![Tests](https://github.com/Hj1308/EISforge-/actions/workflows/tests.yml/badge.svg?style=flat-square)\n",
     ""),
    # generic URL/name fixes
    ("README.md", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("README.md", "cd EISforge-", "cd EISforge"),
    ("README.md", "EISforge-/", "EISforge/"),
    ("CITATION.cff", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("PROJECT_STATUS.md", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("PROJECT_STATUS.md", "EISforge-/", "EISforge/"),
    ("pyproject.toml", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("app.py", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("CONTEXT.md", "Hj1308/EISforge-", "Hj1308/EISforge"),
    ("CONTEXT.md", "EISforge-/", "EISforge/"),
]

touched = {}
for path, old, new in REPLACEMENTS:
    try:
        s = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"skip (missing): {path}")
        continue
    n = s.count(old)
    if n == 0:
        continue
    if path not in touched:
        shutil.copy(path, path + ".bak_patch14")
        touched[path] = s
    touched[path] = touched[path].replace(old, new)
    print(f"[{path}] replaced {n} occurrence(s) of {old!r}")

for path, s in touched.items():
    open(path, "w", encoding="utf-8").write(s)
    print("Patched OK:", path)

# final verification
import re
leftovers = []
for path in {"README.md", "CITATION.cff", "PROJECT_STATUS.md",
             "pyproject.toml", "app.py", "CONTEXT.md"}:
    try:
        s = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        continue
    if "EISforge-" in s:
        leftovers.append(path)
if leftovers:
    print("\nWARNING: 'EISforge-' still present in:", ", ".join(leftovers))
    sys.exit(1)
print("\nAll references updated. No 'EISforge-' left in tracked docs/code.")
