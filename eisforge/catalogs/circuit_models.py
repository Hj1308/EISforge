"""
eisforge.catalogs.circuit_models
================================
Curated registry of standard equivalent-circuit models for EIS fitting,
indexed by (catalyst_type, EIS_region, electrolyte).

Design goals
------------
* **Open/Closed**: adding a new catalyst family or circuit topology only
  requires appending entries to ``CIRCUIT_MAP`` — no changes to
  ``EISCVCorrelator`` or any other analysis class.
* **Machine-readable**: each entry is a ``CircuitModel`` dataclass whose
  ``notation`` field is directly passable to ``impedance.py`` / ``CNLSFitter``.
* **Human-readable**: ``description`` and ``rationale`` fields explain the
  physical basis so recommendations are peer-review defensible.

Notation convention
-------------------
Follows the ``impedance.py`` string syntax:
    ``R0-p(R1,CPE1)-p(R2,CPE2)``
    ``R0-p(R1,CPE1)-W1``
    ``R0-p(R1,CPE1)-p(R2,CPE2-Wo2)``

Where:
    ``R``   = pure resistance
    ``CPE`` = Constant Phase Element
    ``W``   = semi-infinite Warburg (diffusion)
    ``Wo``  = finite-length ("open") Warburg (pore diffusion)
    ``p(X,Y)`` = X and Y in parallel

Author: Hoda Jaafari | June 2026
License: MIT
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from eisforge.analysis.cv_analyzer import (
    CATALYST_NOBLE_METAL,
    CATALYST_ALLOY,
    CATALYST_METAL_OXIDE,
    CATALYST_METAL_FREE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CircuitModel dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircuitModel:
    """
    Immutable descriptor for a standard EIS equivalent-circuit model.

    Attributes
    ----------
    name : str
        Short human-readable identifier (e.g. ``"Randles"`` or ``"Two-RC"``).
    notation : str
        ``impedance.py``-compatible circuit string. Passable directly to
        ``CNLSFitter.fit(circuit=model.notation)``.
    description : str
        One-sentence description of the circuit topology.
    rationale : str
        Physical justification for recommending this model in the given context.
    elements : list of str
        Ordered list of element labels as they appear in ``notation``.
    references : list of str
        DOI or citation strings that support this recommendation.
    """
    name         : str
    notation     : str
    description  : str
    rationale    : str
    elements     : List[str]         = field(default_factory=list)
    references   : List[str]         = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}: {self.notation}"


# ---------------------------------------------------------------------------
# Standard circuit library
# ---------------------------------------------------------------------------

class StandardCircuits:
    """
    Registry of canonical EIS equivalent-circuit models.

    Each class attribute is a ``CircuitModel`` instance. Import selectively::

        from eisforge.catalogs.circuit_models import StandardCircuits
        model = StandardCircuits.TWO_RC
        print(model.notation)  # 'R0-p(R1,CPE1)-p(R2,CPE2)'
    """

    # ------------------------------------------------------------------
    # 1. Simple Randles — noble metals at post-onset, clean conditions
    # ------------------------------------------------------------------
    RANDLES = CircuitModel(
        name        = "Randles",
        notation    = "R0-p(R1,CPE1)",
        description = "Single RC loop: solution resistance + one charge-transfer/double-layer arc.",
        rationale   = (
            "Suitable for clean noble-metal surfaces (Pt, Pd, Au) at potentials "
            "well into the activation-controlled region where a single faradaic "
            "process dominates and mass-transport effects are negligible."
        ),
        elements    = ["R0", "R1", "CPE1"],
        references  = [
            "doi:10.1002/celc.201600402",  # rGO-modified electrode, ChemElectroChem 2017
            "doi:10.1039/d1ra01841h",       # EOR noble metals review, RSC Adv 2021
        ],
    )

    # ------------------------------------------------------------------
    # 2. Randles + Warburg — diffusion-limited noble metals / alloys
    # ------------------------------------------------------------------
    RANDLES_WARBURG = CircuitModel(
        name        = "Randles-Warburg",
        notation    = "R0-p(R1,CPE1)-W1",
        description = "Single RC loop with semi-infinite Warburg diffusion tail.",
        rationale   = (
            "Use when low-frequency Warburg tail is visible in Nyquist plot — indicates "
            "mass-transport limitation of fuel (methanol/ethanol) to the electrode surface. "
            "Common for Pt and Pd catalysts at high current densities or in dilute electrolyte."
        ),
        elements    = ["R0", "R1", "CPE1", "W1"],
        references  = [
            "doi:10.1016/j.electacta.2021.138274",
            "doi:10.1039/d1ra01841h",
        ],
    )

    # ------------------------------------------------------------------
    # 3. Two-RC — alloys / poisoning / two distinct processes
    # ------------------------------------------------------------------
    TWO_RC = CircuitModel(
        name        = "Two-RC",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2)",
        description = "Two parallel RC loops in series: two distinct interfacial processes.",
        rationale   = (
            "Essential for alloy catalysts (PtRu, PdAu, PtCu) and CO-poisoned noble "
            "metals where two semi-circles are visible in the Nyquist plot. "
            "R1/CPE1 corresponds to the high-frequency charge-transfer arc; "
            "R2/CPE2 to the low-frequency surface-reconstruction or CO-stripping arc. "
            "Also recommended when I_f/I_b < 1.0 (CO poisoning suspected)."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2"],
        references  = [
            "doi:10.1021/acsaem.5c02906",  # rGO@NiCrFe EIS/MOR, ACS AEM 2025
            "doi:10.1039/d5ra02228b",       # CNT@rGO@Cu2S, RSC Adv 2025
        ],
    )

    # ------------------------------------------------------------------
    # 4. Two-RC + Warburg — alloys with diffusion at high overpotential
    # ------------------------------------------------------------------
    TWO_RC_WARBURG = CircuitModel(
        name        = "Two-RC-Warburg",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2)-W1",
        description = "Two RC arcs followed by semi-infinite Warburg diffusion.",
        rationale   = (
            "Use for alloy catalysts in the post-onset region when both "
            "a surface-process arc AND a low-frequency Warburg tail are observed. "
            "Indicates concurrent CO poisoning/surface reconstruction AND "
            "mass-transport limitation."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2", "W1"],
        references  = [
            "doi:10.1039/d1ra01841h",
            "doi:10.1016/j.electacta.2021.138274",
        ],
    )

    # ------------------------------------------------------------------
    # 5. Metal oxide — one RC + finite-length Warburg (pore/film)
    # ------------------------------------------------------------------
    OXIDE_ONE_RC_PORE = CircuitModel(
        name        = "Oxide-OneRC-PoreWarburg",
        notation    = "R0-p(R1,CPE1)-Wo1",
        description = "Single RC arc with finite-length (open) Warburg for oxide film/pore diffusion.",
        rationale   = (
            "Appropriate for metal oxides (NiO, MnO2) in pre-onset or onset region "
            "where ion diffusion through the oxide film or porous layer is rate-limiting. "
            "Wo (finite-length Warburg) better models bounded diffusion within the film "
            "than semi-infinite W, which assumes unbounded diffusion."
        ),
        elements    = ["R0", "R1", "CPE1", "Wo1"],
        references  = [
            "doi:10.1002/slct.202405083",  # Me-PANI@GO, ChemistrySelect 2024
            "doi:10.1002/adma.202405664",  # C-MFEC review, Adv. Mater. 2024
        ],
    )

    # ------------------------------------------------------------------
    # 6. Metal oxide — two RC (redox process + direct oxidation)
    # ------------------------------------------------------------------
    OXIDE_TWO_RC = CircuitModel(
        name        = "Oxide-TwoRC",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2)",
        description = "Two RC arcs: M(OH)x/MOOx redox process + direct surface oxidation arc.",
        rationale   = (
            "Recommended for metal oxides (Co3O4, NiO, IrO2) in alkaline electrolyte at "
            "post-onset potential, where two distinct processes are active: "
            "(i) the M(OH)x ⇌ MOOx surface redox conversion (high-frequency arc) and "
            "(ii) the direct AOR electrocatalytic step (low-frequency arc). "
            "In acidic media, the low-frequency arc may shift due to oxide dissolution kinetics."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2"],
        references  = [
            "doi:10.1021/acsaem.5c02906",
            "doi:10.1039/d3nr04502a",
        ],
    )

    # ------------------------------------------------------------------
    # 7. Metal oxide — two RC + finite Warburg (alkaline, porous)
    # ------------------------------------------------------------------
    OXIDE_TWO_RC_PORE = CircuitModel(
        name        = "Oxide-TwoRC-PoreWarburg",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2-Wo2)",
        description = "Two RC arcs with finite Warburg nested in second loop (porous oxide film).",
        rationale   = (
            "Best choice for porous/nanostructured metal oxides in alkaline electrolyte. "
            "The finite Warburg Wo2 nested inside the second RC loop captures ion diffusion "
            "through the porous oxide layer. Distinguishes outer surface activity (R1/CPE1) "
            "from inner pore-limited process (R2/CPE2+Wo2)."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2", "Wo2"],
        references  = [
            "doi:10.1039/d5ra02228b",
            "doi:10.1002/adma.202405664",
        ],
    )

    # ------------------------------------------------------------------
    # 8. Carbon/metal-free — two RC + inter-particle contact resistance
    # ------------------------------------------------------------------
    CARBON_TWO_RC = CircuitModel(
        name        = "Carbon-TwoRC",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2)",
        description = "Two RC arcs: inter-particle contact resistance + double-layer/faradaic arc.",
        rationale   = (
            "Recommended for carbon-based metal-free electrocatalysts (N-doped C, CNT, rGO) "
            "in acidic electrolyte. The first loop (R1/CPE1) captures the high-frequency "
            "inter-particle or grain-boundary contact resistance, which is characteristic of "
            "powder-pressed carbon electrodes. The second loop (R2/CPE2) is the true "
            "double-layer + faradaic charge-transfer process."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2"],
        references  = [
            "doi:10.1002/adma.202405664",  # C-MFEC review, Adv. Mater. 2024
            "doi:10.1039/d5ra02228b",
        ],
    )

    # ------------------------------------------------------------------
    # 9. Carbon/metal-free — two RC + pore Warburg (alkaline)
    # ------------------------------------------------------------------
    CARBON_POROUS = CircuitModel(
        name        = "Carbon-Porous",
        notation    = "R0-p(R1,CPE1)-p(R2,CPE2)-Wo1",
        description = "Two RC arcs with finite-length Warburg: contact + charge-transfer + pore diffusion.",
        rationale   = (
            "Use for mesoporous N-doped carbon or graphene-based catalysts (N-rGO, N-CNT) "
            "in alkaline electrolyte. Three distinct processes are expected: "
            "(i) inter-particle/contact resistance (R1), "
            "(ii) double-layer capacitance and charge-transfer at basal plane defect sites (R2), "
            "(iii) ion diffusion through mesopores (Wo, finite-length). "
            "The CPE exponent (n) is typically 0.7–0.85 for these heterogeneous surfaces "
            "— significantly below 1.0, indicating surface disorder from N-doping."
        ),
        elements    = ["R0", "R1", "CPE1", "R2", "CPE2", "Wo1"],
        references  = [
            "doi:10.1002/adma.202405664",
            "doi:10.1039/d3nr04502a",  # EIS for carbon nitride / graphitic carbon
            "doi:10.1039/d5ra02228b",
        ],
    )

    # ------------------------------------------------------------------
    # 10. Default safe fallback
    # ------------------------------------------------------------------
    DEFAULT_FALLBACK = CircuitModel(
        name        = "Randles-CPE",
        notation    = "R0-p(R1,CPE1)",
        description = "Minimal Randles cell with CPE — safe starting point for any system.",
        rationale   = (
            "No specific model could be determined for this catalyst-region-electrolyte "
            "combination. Use this as an initial fit, then inspect the Nyquist plot manually "
            "for additional arcs or a low-frequency Warburg tail before selecting a more "
            "appropriate topology."
        ),
        elements    = ["R0", "R1", "CPE1"],
        references  = [],
    )


# ---------------------------------------------------------------------------
# CIRCUIT_MAP
# Key: (catalyst_type, eis_region, electrolyte)
# Value: CircuitModel
# ---------------------------------------------------------------------------
# Covering 20 canonical combinations across 4 catalyst families,
# 3 EIS regions (pre-onset, onset, post-onset), and 2 electrolytes.
# ---------------------------------------------------------------------------

#: Lookup key type alias for clarity
CircuitKey = Tuple[str, str, str]  # (catalyst_type, region, electrolyte)

CIRCUIT_MAP: Dict[CircuitKey, CircuitModel] = {

    # ── Noble Metal (Pt, Pd, Au, Rh) ──────────────────────────────────────
    # Acidic (H2SO4, HClO4)
    (CATALYST_NOBLE_METAL, "pre-onset",  "acidic")  : StandardCircuits.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "acidic")  : StandardCircuits.RANDLES,
    (CATALYST_NOBLE_METAL, "post-onset", "acidic")  : StandardCircuits.RANDLES_WARBURG,
    # Alkaline (KOH, NaOH) — lower R_ct, often cleaner single arc at onset
    (CATALYST_NOBLE_METAL, "pre-onset",  "alkaline"): StandardCircuits.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "alkaline"): StandardCircuits.RANDLES,
    (CATALYST_NOBLE_METAL, "post-onset", "alkaline"): StandardCircuits.RANDLES_WARBURG,

    # ── Alloy (PtRu, PdAu, PtSn, PtCu) ───────────────────────────────────
    # Acidic — CO poisoning more severe; two-RC almost always needed
    (CATALYST_ALLOY, "pre-onset",  "acidic")  : StandardCircuits.TWO_RC,
    (CATALYST_ALLOY, "onset",      "acidic")  : StandardCircuits.TWO_RC,
    (CATALYST_ALLOY, "post-onset", "acidic")  : StandardCircuits.TWO_RC_WARBURG,
    # Alkaline — bifunctional mechanism active, but CO poisoning less severe
    (CATALYST_ALLOY, "pre-onset",  "alkaline"): StandardCircuits.TWO_RC,
    (CATALYST_ALLOY, "onset",      "alkaline"): StandardCircuits.TWO_RC,
    (CATALYST_ALLOY, "post-onset", "alkaline"): StandardCircuits.RANDLES_WARBURG,

    # ── Metal Oxide (NiO, Co3O4, MnO2, IrO2) ─────────────────────────────
    # Acidic — oxide dissolution risk; simpler film model suffices
    (CATALYST_METAL_OXIDE, "pre-onset",  "acidic")  : StandardCircuits.OXIDE_ONE_RC_PORE,
    (CATALYST_METAL_OXIDE, "onset",      "acidic")  : StandardCircuits.OXIDE_TWO_RC,
    (CATALYST_METAL_OXIDE, "post-onset", "acidic")  : StandardCircuits.OXIDE_TWO_RC,
    # Alkaline — M(OH)x/MOOx conversion active; porous film important
    (CATALYST_METAL_OXIDE, "pre-onset",  "alkaline"): StandardCircuits.OXIDE_ONE_RC_PORE,
    (CATALYST_METAL_OXIDE, "onset",      "alkaline"): StandardCircuits.OXIDE_TWO_RC,
    (CATALYST_METAL_OXIDE, "post-onset", "alkaline"): StandardCircuits.OXIDE_TWO_RC_PORE,

    # ── Carbon / Metal-Free (N-doped C, CNT, rGO, graphene) ───────────────
    # Acidic — inter-particle contact resistance dominates
    (CATALYST_METAL_FREE, "pre-onset",  "acidic")  : StandardCircuits.CARBON_TWO_RC,
    (CATALYST_METAL_FREE, "onset",      "acidic")  : StandardCircuits.CARBON_TWO_RC,
    (CATALYST_METAL_FREE, "post-onset", "acidic")  : StandardCircuits.CARBON_TWO_RC,
    # Alkaline — pore diffusion of OH- adds Warburg tail
    (CATALYST_METAL_FREE, "pre-onset",  "alkaline"): StandardCircuits.CARBON_POROUS,
    (CATALYST_METAL_FREE, "onset",      "alkaline"): StandardCircuits.CARBON_POROUS,
    (CATALYST_METAL_FREE, "post-onset", "alkaline"): StandardCircuits.CARBON_POROUS,
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def lookup_circuit(
    catalyst_type : str,
    region        : str,
    electrolyte   : str,
) -> CircuitModel:
    """
    Look up the recommended equivalent-circuit model for a given context.

    Lookup is performed in three steps (most-specific to least-specific):
    1. Exact key: ``(catalyst_type, region, electrolyte)``
    2. Partial key ignoring electrolyte: ``(catalyst_type, region, "acidic")``
       as a safe fallback when the electrolyte variant is missing.
    3. Hard fallback: ``StandardCircuits.DEFAULT_FALLBACK``.

    Parameters
    ----------
    catalyst_type : str
        One of the ``CATALYST_*`` constants from ``eisforge.analysis.cv_analyzer``.
    region : str
        ``'pre-onset'``, ``'onset'``, or ``'post-onset'``.
    electrolyte : str
        ``'acidic'`` or ``'alkaline'``.

    Returns
    -------
    CircuitModel
        Always returns a valid ``CircuitModel`` — never raises.

    Examples
    --------
    >>> from eisforge.catalogs.circuit_models import lookup_circuit, CATALYST_METAL_FREE
    >>> model = lookup_circuit(CATALYST_METAL_FREE, "onset", "alkaline")
    >>> print(model.notation)
    'R0-p(R1,CPE1)-p(R2,CPE2)-Wo1'
    >>> print(model.rationale[:60])
    'Use for mesoporous N-doped carbon or graphene-based catalyst'
    """
    electrolyte = electrolyte.lower().strip()
    region      = region.lower().strip()

    # Step 1: exact match
    key = (catalyst_type, region, electrolyte)
    model = CIRCUIT_MAP.get(key)
    if model is not None:
        logger.debug("circuit_models: exact match for key %s → %s", key, model.name)
        return model

    # Step 2: fallback — try acidic variant (conservative choice)
    fallback_key = (catalyst_type, region, "acidic")
    model = CIRCUIT_MAP.get(fallback_key)
    if model is not None:
        logger.warning(
            "circuit_models: no entry for electrolyte='%s'; "
            "using acidic fallback for key %s → %s",
            electrolyte, key, model.name,
        )
        return model

    # Step 3: hard fallback
    logger.warning(
        "circuit_models: no entry found for key %s; returning DEFAULT_FALLBACK.", key
    )
    return StandardCircuits.DEFAULT_FALLBACK
