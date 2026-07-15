# EISForge — Project Context for Coding Agents

## What this project is

EISForge is an open-source Python/Streamlit toolkit for electrochemical analysis
(CV, LSV, EIS, scan-rate kinetics, chronoamperometry) built around the Alcohol
Oxidation Reaction (AOR). It supports Ivium (.idf), Gamry (.dta), BioLogic
(.mpt/.mpr), and generic CSV formats. Live demo: eisforge-app.streamlit.app.
Repo: github.com/Hj1308/EISforge.

---

## ⚠️ MANDATORY WORKFLOW — read before touching any file

This is the single most important rule in this project. It exists because we
already lost a working fix once (the E_onset hybrid-detection patch) by
skipping it. **Do not skip any step below, even for a "small" change.**

### 1. Test before you patch — always

Never write a patch or modify a file based on reasoning alone. Before any
change to detection/fitting logic (onset detection, Tafel domain selection,
peak finding, circuit fitting, etc.):

1. Load **real experimental data** (a real `.idf`/`.txt` file from this
   project, not synthetic data only — synthetic data can hide sign-convention
   and range bugs that only show up on real files).
2. Run the *proposed* logic against it in isolation and print the actual
   numbers (E_onset, Tafel slope, R², etc.).
3. Sanity-check those numbers against any known-good reference values you
   have for this dataset/system before writing the patch itself.
4. Only after the logic is verified on real data do you write the patch file.

If you cannot get real data for a specific case, say so explicitly and flag
the result as unverified — do not present untested logic as working.

### 2. Every patch script must

- Create a `.bak` backup of the target file **before** modifying it.
- Use exact `OLD` → `NEW` string replacement, and check `if OLD in source`
  first. If the check fails, **stop and report it** — never silently no-op
  or fall back to guessing a different anchor string.
- Be a standalone, file-based Python script (`patch_xxx.py`) run with
  `python patch_xxx.py`. Avoid inline `python -c "..."` one-liners — Windows
  `cmd.exe` quoting breaks them unpredictably.
- Print a clear before/after summary (chars added, which checks passed)
  when it finishes, so the result is verifiable without re-opening the file.

### 3. One command at a time

Give one command, show the expected output, wait for the actual output
before proposing the next step. Do not chain multiple untested steps.

### 4. Document immediately — this is the step we skipped last time

The moment a patch is applied and verified working, add an entry to
`PATCHES.md` in the same commit:

```markdown
## patchNN — <short name>
**Date:** YYYY-MM-DD
**Files changed:** ...
**What it does:** ...
**Tested on:** <real files used> → <key results>
```

Then:

```
git add <changed files> PATCHES.md
git commit -m "patchNN: <short description>"
git push origin main
```

A patch that is not committed and not logged in `PATCHES.md` does not
count as done — it will be lost the next time something goes wrong locally.
(This is exactly how the E_onset hybrid-detection fix was lost: it was
written directly into `lsv_analyzer.py`, tested, and worked, but was never
committed or logged before the local file was deleted.)

### 5. Restart, don't rely on Streamlit's hot-reload

After editing any file that `app.py` imports (not `app.py` itself),
Streamlit's browser "Rerun" is not enough — fully restart:
`Ctrl+C`, then `streamlit run app.py` again.

### 6. Working directory check

The correct project folder is `C:\Users\hoda\Desktop\EISforg`
(not `C:\Users\hoda\eisforge`, which is a stale duplicate). Confirm `pwd`/`cd`
before running any patch script.

---

## Known gotchas (don't reintroduce these)

- **numpy ≥ 2.x removed `np.trapz`** — use `np.trapezoid` everywhere.
- **`import math` must be at the top-level module scope**, not inside a
  nested function (e.g. inside a sidebar callback) — causes `NameError`.
- **Ivium `.idf` files store anodic current with a sign convention that can
  be negative** depending on the scan. Any logic that filters on `j > 0`
  or compares raw signed current must first verify orientation — do not
  assume anodic = positive without checking.
- **`@st.cache_data` cache keys** must include a `content_hash` (e.g. via
  `hashlib.md5`), not just an index like `cycle_idx` — otherwise different
  files with the same index collide in the cache.

---

## Style

- Code comments and docstrings: English.
- Commit messages: `patchNN: <what changed>`, imperative mood.
- Prefer explicit, verbose validation output over silent success.
