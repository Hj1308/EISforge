# ========================================================
# EISForge Patch: Add substrate/alcohol to CVAnalyzer
# Author: Hoda Jafari — May 2026
# Apply: python apply_patch.py
# Run from: C:\Users\hoda\Desktop\EISforg\
# ========================================================

import re, sys, shutil
from pathlib import Path

# ---- PATCH 1: cv_analyzer.py ----------------------------------------
cv_path = Path("eisforge/analysis/cv_analyzer.py")
if not cv_path.exists():
    print("ERROR: cv_analyzer.py not found at", cv_path)
    sys.exit(1)

shutil.copy(cv_path, cv_path.with_suffix(".py.bak"))
src = cv_path.read_text(encoding="utf-8")

SUBSTRATE_CONSTANTS = """
# -- Substrate (alcohol) constants --
SUBSTRATE_ETHANOL         = "ethanol"
SUBSTRATE_METHANOL        = "methanol"
SUBSTRATE_ETHYLENE_GLYCOL = "ethylene glycol"
SUBSTRATE_GLYCEROL        = "glycerol"
SUBSTRATE_NA              = "N/A"

# Onset ranges per substrate in alkaline (V vs RHE)
SUBSTRATE_ONSET = {
    "ethanol":         {"noblemetal":(0.20,0.40),"alloy":(0.10,0.30),"metaloxide":(1.30,1.50),"carbonmaterial":(0.40,0.70)},
    "methanol":        {"noblemetal":(0.25,0.45),"alloy":(0.15,0.35),"metaloxide":(1.35,1.55),"carbonmaterial":(0.45,0.75)},
    "ethylene glycol": {"noblemetal":(0.30,0.50),"alloy":(0.20,0.40),"metaloxide":(1.40,1.60),"carbonmaterial":(0.50,0.80)},
    "glycerol":        {"noblemetal":(0.35,0.55),"alloy":(0.25,0.45),"metaloxide":(1.45,1.65),"carbonmaterial":(0.55,0.85)},
}
"""

# 1a — add constants after CATALYST_METAL_FREE line
match = re.search(r'^CATALYST_METAL_FREE\s*=.*$', src, re.MULTILINE)
if match and "SUBSTRATE_ETHANOL" not in src:
    src = src[:match.end()] + "\n" + SUBSTRATE_CONSTANTS + src[match.end():]
    print("OK 1a: substrate constants added")
else:
    print("-- 1a: skipped (already present or anchor missing)")

# 1b — add substrate param to CVAnalyzer.__init__ signature
if "substrate: str = " not in src:
    src = re.sub(
        r'(catalyst_loading:\s*float\s*=\s*0\.0,)',
        r'\1\n        substrate: str = "N/A",',
        src, count=1
    )
    print("OK 1b: substrate param added to __init__")
else:
    print("-- 1b: skipped")

# 1c — store self.substrate in body
if "self.substrate = substrate" not in src:
    src = src.replace(
        "self.catalyst_loading = catalyst_loading",
        "self.catalyst_loading = catalyst_loading\n        self.substrate = substrate",
        1
    )
    print("OK 1c: self.substrate stored")
else:
    print("-- 1c: skipped")

# 1d — add substrate to CVAnalysisResult dataclass
if "substrate: str = " not in src:
    src = src.replace(
        "catalyst_loading: float = 0.0",
        'catalyst_loading: float = 0.0\n    substrate: str = "N/A"',
        1
    )
    print("OK 1d: substrate field added to CVAnalysisResult")
else:
    print("-- 1d: skipped")

# 1e — pass substrate in analyze() return
if "substrate=self.substrate," not in src:
    src = src.replace(
        "catalyst_loading=self.catalyst_loading,",
        "catalyst_loading=self.catalyst_loading,\n            substrate=self.substrate,",
        1
    )
    print("OK 1e: substrate passed to CVAnalysisResult")
else:
    print("-- 1e: skipped")

# 1f — substrate note in interpret()
SUBSTRATE_NOTE = r"""
        # -- substrate note (injected by patch) --
        if hasattr(self, 'substrate') and self.substrate not in ("N/A", "", None):
            sub = self.substrate
            sub_ranges = SUBSTRATE_ONSET.get(sub, {}).get(ctype, None)
            if sub_ranges:
                lo_s, hi_s = sub_ranges
                qual = "excellent" if eonset <= lo_s else ("moderate" if eonset <= hi_s else "high overpotential")
                parts.append(f"Substrate {sub}: onset {eonset:.3f} V ({qual} for {sub})")
            else:
                parts.append(f"Substrate: {sub}")
            notes = {
                "glycerol":        "Glycerol (3C): complex multi-step oxidation; partial products include glycerate, tartronate, oxalate.",
                "ethylene glycol": "Ethylene glycol (2C): primary oxidation to glycolate/oxalate; check for complete vs partial oxidation.",
                "methanol":        "Methanol (1C): complete oxidation to CO2 feasible; CO poisoning risk on Pt-based catalysts.",
                "ethanol":         "Ethanol (2C): C-C cleavage required for complete oxidation; acetate/acetic acid common partial product.",
            }
            if sub in notes:
                parts.append(notes[sub])
"""

if "substrate note (injected by patch)" not in src:
    src = src.replace(
        "return '. '.join(parts)",
        SUBSTRATE_NOTE + "        return '. '.join(parts)",
        1
    )
    print("OK 1f: substrate interpretation block added")
else:
    print("-- 1f: skipped")

cv_path.write_text(src, encoding="utf-8")
print("\nOK: cv_analyzer.py saved  (backup: .py.bak)\n")

# ---- PATCH 2: app.py ----------------------------------------
app_path = Path("app.py")
if not app_path.exists():
    print("ERROR: app.py not found")
    sys.exit(1)

shutil.copy(app_path, app_path.with_suffix(".py.bak"))
app = app_path.read_text(encoding="utf-8")

# 2a — CVAnalyzer call (CV tab)
m = re.search(r'ana\s*=\s*CVAnalyzer\s*\(.*?\)', app, re.DOTALL)
if m:
    old = m.group(0)
    if "substrate=" not in old:
        new = old.rstrip(')') + ",\n                    substrate=alcohol if system_type == 'AOR' else 'N/A')"
        app = app[:m.start()] + new + app[m.end():]
        print("OK 2a: substrate=alcohol passed to CVAnalyzer")
    else:
        print("-- 2a: skipped")
else:
    print("WARN 2a: CVAnalyzer() call not found")

# 2b — BatchCVAnalyzer call
m2 = re.search(r'batch_ana\s*=\s*BatchCVAnalyzer\s*\(.*?\)', app, re.DOTALL)
if m2:
    old2 = m2.group(0)
    if "substrate=" not in old2:
        new2 = old2.rstrip(')') + ",\n                        substrate=alcohol if system_type == 'AOR' else 'N/A')"
        app = app[:m2.start()] + new2 + app[m2.end():]
        print("OK 2b: substrate=alcohol passed to BatchCVAnalyzer")
    else:
        print("-- 2b: skipped")
else:
    print("-- 2b: BatchCVAnalyzer not found (optional)")

app_path.write_text(app, encoding="utf-8")
print("OK: app.py saved  (backup: .py.bak)\n")

print("=" * 55)
print("PATCH COMPLETE")
print("Changes made:")
print("  cv_analyzer.py")
print("    + SUBSTRATE_* constants & SUBSTRATE_ONSET dict")
print("    + CVAnalyzer(substrate=...) parameter")
print("    + CVAnalysisResult.substrate field")
print("    + interpret() substrate-specific commentary")
print("  app.py")
print("    + CVAnalyzer call -> substrate=alcohol")
print("    + BatchCVAnalyzer call -> substrate=alcohol")
print("=" * 55)
