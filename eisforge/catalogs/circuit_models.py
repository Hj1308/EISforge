"""
Catalog of suggested Equivalent Circuit Models for EIS.
Author: Hoda Jafari | May 2026

Maps (catalyst_type, EIS_region, electrolyte) to a recommended circuit string
compatible with the EISForge circuit parser.

Circuit syntax:
    R0 - p(R1, CPE1) - W1
    where p(A, B) denotes a parallel combination.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalyst type constants
# ---------------------------------------------------------------------------
try:
    from eisforge.analysis.cv_analyzer import (
        CATALYST_NOBLE_METAL,
        CATALYST_ALLOY,
        CATALYST_METAL_OXIDE,
        CATALYST_METAL_FREE,
    )
except ImportError:
    # Fallback for standalone / testing usage
    CATALYST_NOBLE_METAL = "noble_metal"
    CATALYST_ALLOY       = "alloy"
    CATALYST_METAL_OXIDE = "metal_oxide"
    CATALYST_METAL_FREE  = "carbon_material"


# ---------------------------------------------------------------------------
# Circuit model catalogue
# ---------------------------------------------------------------------------
class CircuitModel(str, Enum):
    """
    Standard EIS equivalent circuit representations.

    The string value follows the EISForge circuit parser syntax::

        R0-p(R1,CPE1)-W1

    where ``p(A,B)`` denotes a parallel combination.
    """

    # -- Single-RC models
    RANDLES_SIMPLE   = "R0-p(R1,CPE1)"                      # One RC  (simplest Randles)
    RANDLES_WARBURG  = "R0-p(R1,CPE1)-W1"                   # One RC + semi-infinite Warburg

    # -- Two-RC models
    TWO_RC           = "R0-p(R1,CPE1)-p(R2,CPE2)"          # Classical Pt / Pd two-step
    TWO_RC_WARBURG   = "R0-p(R1,CPE1)-p(R2,CPE2)-W1"       # Two RC + Warburg (most common for metals)

    # -- Porous / Carbon-specific
    POROUS_CARBON         = "R0-p(R1,CPE1)-p(R2,CPE2)-W2"        # W2 = finite-length Warburg (bounded)
    POROUS_CARBON_CONTACT = "R0-R1-p(R2,CPE1)-p(R3,CPE2)"        # Includes inter-particle contact R

    # -- Metal-oxide specific (two distinct redox steps)
    OXIDE_TWO_STEP   = "R0-p(R1,CPE1)-p(R2,CPE2)"

    # -- Default fallback
    DEFAULT          = "R0-p(R1,CPE1)"


# ---------------------------------------------------------------------------
# Lookup table: (catalyst_type, eis_region, electrolyte) -> CircuitModel
# ---------------------------------------------------------------------------
CIRCUIT_SUGGESTIONS: dict[tuple[str, str, str], CircuitModel] = {
    # ---- Noble Metals (Pt, Pd, Au, Rh) ------------------------------------
    (CATALYST_NOBLE_METAL, "pre-onset",  "acidic"):   CircuitModel.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "acidic"):   CircuitModel.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "post-onset", "acidic"):   CircuitModel.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "pre-onset",  "alkaline"): CircuitModel.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "alkaline"): CircuitModel.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "post-onset", "alkaline"): CircuitModel.TWO_RC_WARBURG,

    # ---- Alloys (PtRu, PtSn, PdAu, etc.) ----------------------------------
    (CATALYST_ALLOY, "pre-onset",  "acidic"):   CircuitModel.TWO_RC,
    (CATALYST_ALLOY, "onset",      "acidic"):   CircuitModel.TWO_RC_WARBURG,
    (CATALYST_ALLOY, "post-onset", "acidic"):   CircuitModel.TWO_RC_WARBURG,
    (CATALYST_ALLOY, "pre-onset",  "alkaline"): CircuitModel.TWO_RC,
    (CATALYST_ALLOY, "onset",      "alkaline"): CircuitModel.TWO_RC_WARBURG,
    (CATALYST_ALLOY, "post-onset", "alkaline"): CircuitModel.TWO_RC_WARBURG,

    # ---- Metal Oxides (NiO, Co₃O₄, MnO₂, etc.) ----------------------------
    (CATALYST_METAL_OXIDE, "pre-onset",  "acidic"):   CircuitModel.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "onset",      "acidic"):   CircuitModel.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "post-onset", "acidic"):   CircuitModel.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "pre-onset",  "alkaline"): CircuitModel.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "onset",      "alkaline"): CircuitModel.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "post-onset", "alkaline"): CircuitModel.OXIDE_TWO_STEP,

    # ---- Metal-Free / Carbon materials (N-C, CNT, rGO, graphene) ----------
    (CATALYST_METAL_FREE, "pre-onset",  "acidic"):   CircuitModel.POROUS_CARBON,
    (CATALYST_METAL_FREE, "onset",      "acidic"):   CircuitModel.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "post-onset", "acidic"):   CircuitModel.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "pre-onset",  "alkaline"): CircuitModel.POROUS_CARBON,
    (CATALYST_METAL_FREE, "onset",      "alkaline"): CircuitModel.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "post-onset", "alkaline"): CircuitModel.POROUS_CARBON_CONTACT,
}


def get_suggested_circuit(
    catalyst_type: str,
    eis_region: str,
    electrolyte: str = "acidic",
) -> CircuitModel:
    """
    Return the recommended equivalent circuit for the given experimental conditions.

    Parameters
    ----------
    catalyst_type : str
        One of the ``CATALYST_*`` constants from :mod:`eisforge.analysis.cv_analyzer`.
    eis_region : str
        ``'pre-onset'``, ``'onset'``, or ``'post-onset'``.
    electrolyte : str
        ``'acidic'`` or ``'alkaline'``.

    Returns
    -------
    CircuitModel
        Recommended circuit model.  Falls back to :attr:`CircuitModel.DEFAULT`
        with a warning when the exact combination is not in the catalogue.
    """
    key = (catalyst_type, eis_region, electrolyte)
    model = CIRCUIT_SUGGESTIONS.get(key)

    if model is None:
        logger.warning(
            "No specific circuit model for key %s. Using DEFAULT.", key
        )
        return CircuitModel.DEFAULT

    return model
