"""
One-time fixes required before running train_eis_gpt.py (v2).
Run from the project root: python apply_v2_fixes.py

Fix 1: aor_dataset_generator.py
  Circuit 3 declared 3 Warburg params (Wo1_R, Wo1_T, Wo1_phi) but
  impedance.py's Wo element takes exactly 2 (Wo1_R, Wo1_T).
  Result: every circuit-3 sample raised an exception and was silently
  rejected, so the model only ever trained on 4 of 5 classes.

Fix 2: transformer.py
  MAX_PARAMS was 7, but circuit 3 now has 9 parameters after Fix 1.
  Raise to 9 so no parameter targets get silently truncated.

Verify after running:
  python -c "
  from eisforge.ml.eis_gpt.aor_dataset_generator import AORDatasetGenerator
  from collections import Counter
  print(Counter(r.circuit_label for r in
      AORDatasetGenerator(n_samples_per_circuit=10).generate(verbose=False)))
  "
  Expected: Counter({0: 10, 1: 10, 2: 10, 3: 10, 4: 10})

Author: Hoda Jafari | 2026
"""

import re
from pathlib import Path

ok = True

# ── Fix 1: Wo1 parameter count ────────────────────────────────────────
p1 = Path("eisforge/ml/eis_gpt/aor_dataset_generator.py")
if not p1.exists():
    print(f"[1/2] ERROR: {p1} not found — run from project root.")
    ok = False
else:
    s = p1.read_text(encoding="utf-8")
    if '"Wo1_phi"' in s:
        s = s.replace('"Wo1_R", "Wo1_T", "Wo1_phi"', '"Wo1_R", "Wo1_T"')
        s = re.sub(r'\n\s*"Wo1_phi":\s*\([^)]*\),[^\n]*', "", s)
        p1.write_text(s, encoding="utf-8")
        print(f"[1/2] Fixed Wo1 parameter count (3→2) in {p1}")
    else:
        print(f"[1/2] Already fixed: {p1}")

# ── Fix 2: MAX_PARAMS ─────────────────────────────────────────────────
p2 = Path("eisforge/ml/eis_gpt/transformer.py")
if not p2.exists():
    print(f"[2/2] ERROR: {p2} not found — run from project root.")
    ok = False
else:
    s = p2.read_text(encoding="utf-8")
    if "MAX_PARAMS = 7" in s:
        s = s.replace("MAX_PARAMS = 7", "MAX_PARAMS = 9")
        p2.write_text(s, encoding="utf-8")
        print(f"[2/2] MAX_PARAMS 7→9 in {p2}")
    elif "MAX_PARAMS = 9" in s:
        print(f"[2/2] Already fixed: {p2}")
    else:
        print(f"[2/2] WARNING: MAX_PARAMS line not found in {p2} — check manually.")
        ok = False

# ── Summary ───────────────────────────────────────────────────────────
if ok:
    print("\n✓ All fixes applied. Verify with:")
    print('  python -c "'
          'from eisforge.ml.eis_gpt.aor_dataset_generator import AORDatasetGenerator; '
          'from collections import Counter; '
          'print(Counter(r.circuit_label for r in '
          'AORDatasetGenerator(n_samples_per_circuit=10).generate(verbose=False)))"')
    print("  Expected: Counter({0: 10, 1: 10, 2: 10, 3: 10, 4: 10})")
    print("\nThen run:")
    print("  python train_eis_gpt.py --samples-per-circuit 200 --epochs 5  # smoke test")
    print("  python train_eis_gpt.py --samples-per-circuit 2000 --epochs 50  # full run")
else:
    print("\n✗ Some fixes failed — see messages above.")
