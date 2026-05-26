"""
EISForge — Carbon Material Electrochemical Standards
Author: Hoda Jafari | May 2026

Sources (~200 peer-reviewed papers):
  - Yellatur et al. (2022), Nanotechnology
  - Fu et al. (2023), Advanced Science
  - Maxakato et al. (2024)
  - Brix et al. (2022), ChemElectroChem
  - Gholipour (2021), Final Thesis
  - Pumera (2013), Electrochemistry Communications
  - Tamiji & Nezamzadeh-Ejhieh (2019)
  - Wang et al. (2019), Catalysts
  - Teran-Salgado et al. (2019)
  - Ayman et al. (2023), Journal of Chemistry
  - Matthews (2023)
  - + gemini-code supplement (Hoda Jafari, May 2026)

This file is the SINGLE source of truth for all carbon material
electrochemical limits in EISForge. Import it into cv_analyzer.py,
lsv_analyzer.py, and app.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1.  C_dl REFERENCE RANGES  (μF/cm²)
#     Source: literature compilation + functionalization data (Section 5.1)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CdlRange:
    material: str
    cdl_min_uF: float   # μF/cm²  lower bound
    cdl_max_uF: float   # μF/cm²  upper bound
    note: str = ""


CDL_RANGES: dict[str, CdlRange] = {
    # Pristine / lightly functionalised graphene / rGO
    "graphene": CdlRange(
        "Graphene / rGO",
        cdl_min_uF=10.0,
        cdl_max_uF=50.0,
        note="Pristine or reduced graphene oxide; low defect density",
    ),
    # Single- or multi-wall CNT (oxidised / sulfonated)
    "cnt": CdlRange(
        "Carbon Nanotubes (SWCNT / MWCNT)",
        cdl_min_uF=15.0,
        cdl_max_uF=60.0,
        note="Functionalised CNT; HNO3 / sulfonated surface",
    ),
    # N-doped carbon (graphene, carbon black, porous)
    "n_doped_carbon": CdlRange(
        "N-doped Carbon",
        cdl_min_uF=20.0,
        cdl_max_uF=80.0,
        note="N-doping increases C_dl vs undoped. Fu et al. 2023.",
    ),
    # Oxidatively functionalised carbon (H2O2 / HNO3 treated)
    # Ref: untreated 4.84 mF/cm² → C-H2O2 11.44 mF/cm² (Section 5.1)
    "oxidised_carbon": CdlRange(
        "Oxidised / Functionalised Carbon",
        cdl_min_uF=4.0,   # convert: 4.84 mF/cm² × 1000 ≈ 4840 μF/cm²?
        cdl_max_uF=12.0,  # NOTE: these were reported as mF/cm² in paper →
        note=(             # 4.84–11.44 mF/cm² = 4840–11440 μF/cm² is CPE not C_dl
              "Raw data from paper: 4.84–11.44 mF/cm² (high-surface-area / CPE). "
              "For normalised GC-type C_dl use 'carbon_material' range instead."),
    ),
    # Activated / high-surface-area porous carbon
    "activated_carbon": CdlRange(
        "Activated / Porous Carbon",
        cdl_min_uF=50.0,
        cdl_max_uF=300.0,
        note="High BET surface area (1700+ m²/g for porous N-C). Matthews 2023.",
    ),
    # Vulcan XC-72 / carbon black
    "carbon_black": CdlRange(
        "Carbon Black (Vulcan XC-72 / XC-72R)",
        cdl_min_uF=20.0,
        cdl_max_uF=80.0,
        note="Standard conductive support; featureless CV. Teran-Salgado 2019.",
    ),
    # Glassy Carbon bare electrode
    "glassy_carbon": CdlRange(
        "Glassy Carbon (bare GC disk)",
        cdl_min_uF=10.0,
        cdl_max_uF=40.0,
        note="GC measured in non-Faradaic window vs RHE in 0.1 M KOH. Brix 2022.",
    ),
    # Carbon Paste Electrode (CuO-modified reported 12.90 mF/cm² → keep as note)
    "carbon_paste": CdlRange(
        "Carbon Paste Electrode (CPE)",
        cdl_min_uF=10.0,
        cdl_max_uF=100.0,
        note="Modified CPE (CuO): 12.90 mF/cm² reported. Tamiji 2019.",
    ),
    # Generic fallback for any carbon_material
    "carbon_material": CdlRange(
        "General Carbon Material",
        cdl_min_uF=10.0,
        cdl_max_uF=300.0,
        note="Use specific subtype for tighter bounds. Covers all literature values.",
    ),
}

# Hard limits — ANY carbon beyond these → almost certainly noise / wrong method
CDL_ABSOLUTE_MIN_uF: float = 1.0
CDL_NOISE_THRESHOLD_uF: float = 500.0  # above this → severe noise


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TAFEL SLOPE RANGES  (mV/dec)
#     Source: gemini-code supplement + AOR literature
# ═══════════════════════════════════════════════════════════════════════════

TAFEL_RANGES: dict[str, dict] = {
    "alkaline": {
        "min": 75.0,
        "max": 250.0,
        "normal_min": 75.0,
        "normal_max": 180.0,
        "note": "KOH / NaOH media. gemini-code + Fu 2023.",
    },
    "acidic": {
        "min": 60.0,
        "max": 250.0,
        "normal_min": 60.0,
        "normal_max": 120.0,
        "note": "H2SO4 / HClO4 media.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  R_ct RANGES  (Ω)
#     Source: gemini-code + EIS table (Section 3)
#     Ni-Co/GC: 10.59–36.77 Ω (very active, metal oxide)
#     Pd/Au/GC:  1850 Ω (2-propanol in NaOH)
#     Pt/C on graphite: 207–312 Ω
# ═══════════════════════════════════════════════════════════════════════════

RCT_RANGES: dict[str, Tuple[float, float]] = {
    "alkaline": (1000.0, 30000.0),   # Ω  — gemini-code
    "acidic":   (200.0,  50000.0),   # Ω  — Pt/C on graphite 207–312 Ω + margin
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  E_onset TYPICAL RANGE (V vs RHE)
#     Source: gemini-code supplement
# ═══════════════════════════════════════════════════════════════════════════

ONSET_RHE_TYPICAL: Tuple[float, float] = (0.40, 0.80)

# Recommended E_onset detection method for carbon (featureless CV)
ONSET_METHOD_DEFAULT: str = "derivative"


# ═══════════════════════════════════════════════════════════════════════════
# 5.  RECOMMENDED EEC  (Equivalent Electrical Circuits)
#     Source: gemini-code + Gholipour 2021 + EIS table (Section 3)
# ═══════════════════════════════════════════════════════════════════════════

RECOMMENDED_EEC: dict[str, dict] = {
    # Default for carbon / metal-free in alkaline AOR
    "carbon_alkaline": {
        "circuit": "R0-p(R1,CPE1)",
        "p0": [30.0, 15000.0, 2e-5, 0.82],
        "note": (
            "Randles-CPE: R0 (solution resistance) || R1 (charge transfer) / CPE1. "
            "Gholipour 2021: Q=0.04489 F·s^(a-1), a=0.88 for Ni-Co/GC in KOH. "
            "Use Wo variant if low-frequency diffusion tail visible."
        ),
    },
    # Carbon in alkaline with visible Warburg diffusion tail
    "carbon_alkaline_warburg": {
        "circuit": "R0-p(R1,CPE1)-Wo1",
        "p0": [30.0, 15000.0, 2e-5, 0.82, 500.0, 0.5],
        "note": "Add Warburg element for porous carbon / diffusion-limited systems.",
    },
    # Carbon in acid
    "carbon_acidic": {
        "circuit": "R0-p(R1,CPE1)",
        "p0": [20.0, 8000.0, 1e-5, 0.85],
        "note": "Higher R_ct expected vs alkaline. Pt/C on graphite: 207–312 Ω.",
    },
    # gemini-code original suggestion (kept for compatibility)
    "carbon_gemini": {
        "circuit": "R0-p(CPE1,p(R1,Wo1))",   # R_s(CPE(R_ct W_o)) in gemini notation
        "p0": [30.0, 2e-5, 0.82, 15000.0, 500.0, 0.5],
        "note": "Gemini-code variant: nested CPE + Warburg. Use for porous carbon.",
    },
    # Noble metal (Pt, Pd) — for comparison
    "noble_metal": {
        "circuit": "R0-p(R1,CPE1)",
        "p0": [10.0, 500.0, 2e-6, 0.90],
        "note": "Standard Randles-CPE for noble metal AOR.",
    },
    # Alloy (PtRu, PtSn)
    "alloy": {
        "circuit": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "p0": [10.0, 200.0, 5e-6, 0.85, 2000.0, 1e-5, 0.75],
        "note": "Two-RC: adsorption intermediate + charge transfer loops.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 6.  STANDARD EXPERIMENTAL CONDITIONS
#     Source: Section 7 of Carbon_Materials_Knowledge_Base.md
# ═══════════════════════════════════════════════════════════════════════════

STANDARD_CONDITIONS: dict = {
    "scan_rate_cv_mV_s":      50,     # baseline CV scan rate
    "scan_rate_cdl_mV_s":     [5, 10, 20, 30, 40],   # for C_dl determination
    "scan_rate_lsv_mV_s":     5,      # baseline LSV
    "eis_freq_range_Hz":      (1e-2, 1e5),            # standard full spectrum
    "electrolyte_alkaline_M": 0.1,    # KOH standard
    "nafion_binder_wt":       0.05,   # 0.05–5 wt%
    "purge_time_min":         15,     # Ar / N2 purge before measurement
    "gc_area_cm2":            0.0707, # standard 3 mm GC disk
}


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CARBON MATERIAL SUBTYPE MAP  (for UI dropdown → key lookup)
# ═══════════════════════════════════════════════════════════════════════════

CARBON_SUBTYPE_MAP: dict[str, str] = {
    "Graphene / rGO":                    "graphene",
    "N-doped Graphene":                  "n_doped_carbon",
    "Carbon Nanotubes (CNT / MWCNT)":    "cnt",
    "N-doped CNT":                       "n_doped_carbon",
    "Activated / Porous Carbon":         "activated_carbon",
    "Carbon Black (Vulcan XC-72)":       "carbon_black",
    "Glassy Carbon (GC)":               "glassy_carbon",
    "Carbon Paste Electrode":           "carbon_paste",
    "Other / Unknown":                   "carbon_material",
}


# ═══════════════════════════════════════════════════════════════════════════
# 8.  VALIDATION DATACLASS & VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    passed: bool
    severity: str          # "ok" | "warning" | "error"
    message: str
    suggested_action: str = ""


class CarbonValidator:
    """
    Smart validation of electrochemical parameters for carbon/metal-free catalysts.
    All thresholds sourced from Carbon_Materials_Knowledge_Base.md + gemini-code.

    Usage in cv_analyzer.py:
        from eisforge.standards.carbon_standards import CarbonValidator
        result = CarbonValidator.validate_cdl(cdl_mF_cm2=r.cdl_mF_cm2)
    """

    @staticmethod
    def validate_cdl(
        cdl_mF_cm2: float,
        material_key: str = "carbon_material",
        user_min_uF: Optional[float] = None,
        user_max_uF: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate C_dl value against literature range.

        Parameters
        ----------
        cdl_mF_cm2   : float  — value in mF/cm² (as returned by CVAnalyzer)
        material_key : str    — key from CDL_RANGES (e.g. "graphene", "cnt")
        user_min_uF  : float  — optional manual lower bound in μF/cm²
        user_max_uF  : float  — optional manual upper bound in μF/cm²
        """
        cdl_uF = cdl_mF_cm2 * 1000.0   # mF → μF

        # Determine active range
        if user_min_uF is not None and user_max_uF is not None:
            lo, hi = user_min_uF, user_max_uF
            range_label = f"user-defined {lo:.0f}–{hi:.0f} μF/cm²"
        else:
            ref = CDL_RANGES.get(material_key, CDL_RANGES["carbon_material"])
            lo, hi = ref.cdl_min_uF, ref.cdl_max_uF
            range_label = f"literature {lo:.0f}–{hi:.0f} μF/cm² ({ref.material})"

        # Hard error: noise / measurement artifact
        if cdl_uF > CDL_NOISE_THRESHOLD_uF:
            return ValidationResult(
                passed=False,
                severity="error",
                message=(
                    f"⛔ C_dl = {cdl_uF:.1f} μF/cm² exceeds noise threshold "
                    f"({CDL_NOISE_THRESHOLD_uF:.0f} μF/cm²). "
                    "Likely: severe noise, wrong scan rate, or incorrect baseline."
                ),
                suggested_action=(
                    "Use 5–50 mV/s scan rate in purely non-Faradaic window. "
                    "Verify baseline subtraction. Re-measure or check data quality."
                ),
            )

        # Below absolute minimum
        if cdl_uF < CDL_ABSOLUTE_MIN_uF:
            return ValidationResult(
                passed=False,
                severity="warning",
                message=(
                    f"⚠ C_dl = {cdl_uF:.2f} μF/cm² is unusually low. "
                    "Possible: very low surface area or incorrect area normalisation."
                ),
                suggested_action="Verify geometric area and electrode preparation.",
            )

        # Within range
        if lo <= cdl_uF <= hi:
            return ValidationResult(
                passed=True,
                severity="ok",
                message=f"✅ C_dl = {cdl_uF:.1f} μF/cm² — within {range_label}.",
            )

        # Above range (but below noise threshold)
        if cdl_uF > hi:
            return ValidationResult(
                passed=False,
                severity="warning",
                message=(
                    f"⚠ C_dl = {cdl_uF:.1f} μF/cm² above {range_label}. "
                    "Possible: Faradaic contribution, high surface area, or noise."
                ),
                suggested_action=(
                    "Confirm measurement in non-Faradaic window. "
                    "If activated/porous carbon, select 'activated_carbon' subtype."
                ),
            )

        # Below range
        return ValidationResult(
            passed=False,
            severity="warning",
            message=(
                f"⚠ C_dl = {cdl_uF:.1f} μF/cm² below {range_label}. "
                "Possible: low surface area or poor electrode contact."
            ),
            suggested_action="Verify geometric area and electrode surface preparation.",
        )

    @staticmethod
    def validate_tafel(
        tafel_mV_dec: float,
        electrolyte: str = "alkaline",
    ) -> ValidationResult:
        ref = TAFEL_RANGES.get(electrolyte, TAFEL_RANGES["alkaline"])
        lo, hi     = ref["min"], ref["max"]
        lo_n, hi_n = ref["normal_min"], ref["normal_max"]

        if lo_n <= tafel_mV_dec <= hi_n:
            return ValidationResult(
                passed=True, severity="ok",
                message=f"✅ Tafel = {tafel_mV_dec:.1f} mV/dec — normal range for carbon in {electrolyte} ({lo_n:.0f}–{hi_n:.0f} mV/dec).",
            )
        if lo <= tafel_mV_dec <= hi:
            return ValidationResult(
                passed=True, severity="ok",
                message=(
                    f"⚠ Tafel = {tafel_mV_dec:.1f} mV/dec — acceptable but outside typical "
                    f"({lo_n:.0f}–{hi_n:.0f} mV/dec). Mixed kinetics possible."
                ),
            )
        return ValidationResult(
            passed=False, severity="warning",
            message=(
                f"⛔ Tafel = {tafel_mV_dec:.1f} mV/dec — outside acceptable range "
                f"({lo:.0f}–{hi:.0f} mV/dec). Check potential window and fit quality."
            ),
            suggested_action="Narrow Tafel fitting to linear region only.",
        )

    @staticmethod
    def validate_onset(e_onset_V: float) -> ValidationResult:
        lo, hi = ONSET_RHE_TYPICAL
        if e_onset_V <= lo:
            return ValidationResult(
                passed=True, severity="ok",
                message=f"✅ E_onset = {e_onset_V:.3f} V vs RHE — excellent (below typical lower bound {lo:.2f} V).",
            )
        if lo <= e_onset_V <= hi:
            return ValidationResult(
                passed=True, severity="ok",
                message=f"✅ E_onset = {e_onset_V:.3f} V vs RHE — within typical range for carbon AOR catalysts ({lo:.2f}–{hi:.2f} V).",
            )
        return ValidationResult(
            passed=False, severity="warning",
            message=(
                f"⚠ E_onset = {e_onset_V:.3f} V vs RHE — above typical ({hi:.2f} V). "
                "Poor catalytic activity or incorrect RHE conversion."
            ),
            suggested_action="Check reference electrode calibration and RHE conversion factor.",
        )

    @staticmethod
    def validate_rct(r_ct_ohm: float, electrolyte: str = "alkaline") -> ValidationResult:
        lo, hi = RCT_RANGES.get(electrolyte, RCT_RANGES["alkaline"])
        if lo <= r_ct_ohm <= hi:
            return ValidationResult(
                passed=True, severity="ok",
                message=f"✅ R_ct = {r_ct_ohm:.1f} Ω — within expected range for carbon in {electrolyte} ({lo:.0f}–{hi:.0f} Ω).",
            )
        if r_ct_ohm < lo:
            return ValidationResult(
                passed=True, severity="ok",
                message=f"✅ R_ct = {r_ct_ohm:.1f} Ω — below typical lower bound ({lo:.0f} Ω). Excellent charge transfer.",
            )
        return ValidationResult(
            passed=False, severity="warning",
            message=(
                f"⚠ R_ct = {r_ct_ohm:.1f} Ω — above expected range ({lo:.0f}–{hi:.0f} Ω). "
                "High charge transfer resistance; poor catalytic activity."
            ),
            suggested_action="Check catalyst loading, electrode preparation, and electrolyte concentration.",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9.  EEC SUGGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def suggest_eec(
    catalyst_type: str = "carbon_material",
    electrolyte: str = "alkaline",
    user_circuit: Optional[str] = None,
    has_warburg: bool = False,
) -> dict:
    """
    Returns suggested EEC circuit, initial guess p0, source, and note.
    If user_circuit is provided and non-empty, returns it unchanged (user override).

    Parameters
    ----------
    catalyst_type : str  — "carbon_material", "noble_metal", "alloy", "metal_oxide"
    electrolyte   : str  — "alkaline" | "acidic"
    user_circuit  : str  — if set, overrides auto-suggestion
    has_warburg   : bool — if True, suggests Warburg variant for carbon
    """
    if user_circuit and user_circuit.strip():
        return {
            "circuit": user_circuit.strip(),
            "p0": None,
            "source": "user",
            "note": "User-defined circuit — auto-suggestion overridden.",
        }

    if catalyst_type == "carbon_material":
        if has_warburg:
            key = "carbon_alkaline_warburg" if electrolyte == "alkaline" else "carbon_acidic"
        elif electrolyte == "alkaline":
            key = "carbon_alkaline"
        else:
            key = "carbon_acidic"
    elif catalyst_type == "noble_metal":
        key = "noble_metal"
    elif catalyst_type == "alloy":
        key = "alloy"
    else:
        key = "carbon_alkaline"   # safe fallback

    rec = RECOMMENDED_EEC[key].copy()
    rec["source"] = "auto"
    rec["key"] = key
    return rec
