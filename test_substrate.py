"""
test_substrate.py — Quick test for substrate-aware CVAnalyzer
Run from EISForge root: python test_substrate.py
"""
import numpy as np
from eisforge.analysis.cv_analyzer import (
    CVAnalyzer,
    SUBSTRATE_ETHANOL,
    SUBSTRATE_GLYCEROL,
    SUBSTRATE_METHANOL,
)

# Synthetic CV data — realistic ethanol oxidation shape
E = np.linspace(0.0, 1.2, 300)
I_fwd = 2.5 * np.exp(-((E - 0.60) ** 2) / (2 * 0.06**2))
I_bwd = 1.8 * np.exp(-((E[::-1] - 0.45) ** 2) / (2 * 0.07**2))
I = np.concatenate([I_fwd, I_bwd]) + np.random.normal(0, 0.02, 600)
E_full = np.concatenate([E, E[::-1]])

print("=" * 65)
print("TEST 1: Pt catalyst | 1M KOH | Ethanol")
print("=" * 65)
ana = CVAnalyzer.for_noble_metal(
    electrolyte_compound="KOH",
    concentration=1.0,
    scan_rate=50,
    electrode_area=0.196,
    substrate=SUBSTRATE_ETHANOL,
)
result = ana.analyze(E_full, I)
print(result.summary())
print(f"\nresult.substrate = {result.substrate!r}\n")

print("=" * 65)
print("TEST 2: PtRu alloy | 1M KOH | Methanol")
print("=" * 65)
ana2 = CVAnalyzer.for_alloy(
    electrolyte_compound="KOH",
    concentration=1.0,
    scan_rate=50,
    electrode_area=0.196,
    substrate=SUBSTRATE_METHANOL,
)
result2 = ana2.analyze(E_full, I)
print(result2.summary())
print(f"\nresult2.substrate = {result2.substrate!r}\n")

print("=" * 65)
print("TEST 3: NiO metal-oxide | 1M KOH | Glycerol")
print("=" * 65)
ana3 = CVAnalyzer.for_metal_oxide(
    electrolyte_compound="KOH",
    concentration=1.0,
    scan_rate=50,
    electrode_area=0.196,
    substrate=SUBSTRATE_GLYCEROL,
)
result3 = ana3.analyze(E_full, I)
print(result3.summary())
print(f"\nresult3.substrate = {result3.substrate!r}\n")
