import numpy as np
from eisforge.analysis.ecsa_calculator import calculate_ecsa

E = np.linspace(0, 1.2, 200)
I = np.sin(E * np.pi) * 1e-3

for cat in ["noble_metal", "alloy", "metal_oxide", "carbon_material"]:
    r = calculate_ecsa(E, I, cat, scan_rate=50, catalyst_loading_mg=0.2)
    print(cat, "->", r.method)
    print("  ECSA =", round(r.ecsa_cm2, 4), "cm2")
    print()
print("All methods OK!")
