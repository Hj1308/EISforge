"""
Catalog of suggested Equivalent Circuit Models for EIS fitting.
Author: Hoda Jafari | Updated: June 2026
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Catalyst type constants (fallback if cv_analyzer is not importable) ──────
try:
    from eisforge.analysis.cv_analyzer import (
        CATALYST_NOBLE_METAL,
        CATALYST_ALLOY,
        CATALYST_METAL_OXIDE,
        CATALYST_METAL_FREE,
    )
except ImportError:
    CATALYST_NOBLE_METAL = "noble_metal"
    CATALYST_ALLOY = "alloy"
    CATALYST_METAL_OXIDE = "metal_oxide"
    CATALYST_METAL_FREE = "carbon_material"


@dataclass(frozen=True)
class CircuitModel:
    """
    Immutable equivalent circuit model descriptor.

    ``frozen=True`` prevents accidental mutation of catalog instances that
    are shared across the entire process.  Elements are stored as a tuple
    (immutable) so the dataclass remains hashable.
    """

    notation: str
    name: str
    rationale: str
    confidence: float = 1.0
    source: str = "catalog"   # "catalog" | "autoeis"
    elements: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Validate confidence is a probability in [0, 1].
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        # Auto-populate elements from notation when not provided explicitly.
        if not self.elements and self.notation:
            object.__setattr__(
                self,
                "elements",
                tuple(self._parse_notation(self.notation)),
            )

    @staticmethod
    def _parse_notation(notation: str) -> list[str]:
        """
        Split a circuit notation string into its top-level elements.

        Supports both parenthesis ``()`` and bracket ``[]`` styles used by
        different EIS software packages (e.g. ZView uses square brackets).

        Examples::

            "R0-p(R1,CPE1)-W1"        -> ["R0", "p(R1,CPE1)", "W1"]
            "R0-[R1,CPE1]-W1"         -> ["R0", "[R1,CPE1]", "W1"]
            "R0-p(R1,p(R2,CPE1))-W1"  -> ["R0", "p(R1,p(R2,CPE1))", "W1"]
        """
        parts: list[str] = []
        current = ""
        depth = 0
        for char in notation:
            if char in ("(", "["):
                depth += 1
                current += char
            elif char in (")", "]"):
                depth -= 1
                current += char
            elif char == "-" and depth == 0:
                if current:
                    parts.append(current)
                current = ""
            else:
                current += char
        if current:
            parts.append(current)
        return parts

    def __str__(self) -> str:
        return f"{self.name} ({self.notation}) [{self.source}]"


# ── Pre-defined catalog models ───────────────────────────────────────────────
class CircuitCatalog:
    """Container for predefined, immutable CircuitModel instances."""

    RANDLES_SIMPLE = CircuitModel(
        notation="R0-p(R1,CPE1)",
        name="Simple Randles",
        rationale=(
            "Single charge-transfer process with double-layer capacitance. "
            "Appropriate when mass transport is not limiting (e.g. pre-onset "
            "at oxide surfaces where no Faradaic reaction is active yet)."
        ),
    )

    RANDLES_WARBURG = CircuitModel(
        notation="R0-p(R1,CPE1)-W1",
        name="Randles + Warburg",
        rationale=(
            "Semi-infinite diffusion (Warburg, W1) coupled with a single "
            "charge-transfer arc. Use when one Faradaic process is rate-limited "
            "by bulk diffusion."
        ),
    )

    TWO_RC = CircuitModel(
        notation="R0-p(R1,CPE1)-p(R2,CPE2)",
        name="Two RC loops",
        rationale=(
            "Two distinct charge-transfer or surface-state processes "
            "(e.g. adsorption intermediate + main reaction). "
            "No diffusion limitation."
        ),
    )

    TWO_RC_WARBURG = CircuitModel(
        notation="R0-p(R1,CPE1)-p(R2,CPE2)-W1",
        name="Two RC + Warburg",
        rationale=(
            "Two charge-transfer processes with semi-infinite diffusion. "
            "Typical for noble-metal and alloy catalysts at onset and "
            "post-onset potentials."
        ),
    )

    # Metal oxide: same topology as TWO_RC but physically distinct loops.
    # CPE1 captures outer-surface redox; CPE2 captures inner-pore
    # pseudo-capacitance.  A separate model instance is kept so that
    # downstream code can distinguish by identity (``is`` check) rather
    # than by notation string comparison.
    OXIDE_TWO_STEP = CircuitModel(
        notation="R0-p(R1,CPE1)-p(R2,CPE2)",
        name="Metal Oxide Two-Step",
        rationale=(
            "CPE1: outer-surface double-layer and fast redox (M(OH)x ⇌ MOOH). "
            "CPE2: inner-pore pseudo-capacitance with diffusion-limited ion "
            "intercalation. Distinct from TWO_RC by physical interpretation, "
            "not by circuit topology."
        ),
    )

    POROUS_CARBON = CircuitModel(
        notation="R0-p(R1,CPE1)-p(R2,CPE2)-W2",
        name="Porous Carbon (finite Warburg)",
        rationale=(
            "Finite-length diffusion element (W2) models pore-confined ion "
            "transport in carbon-based catalysts. Inner CPE2 reflects "
            "capacitance deep inside the pore network."
        ),
    )

    POROUS_CARBON_CONTACT = CircuitModel(
        notation="R0-R1-p(R2,CPE1)-p(R3,CPE2)",
        name="Porous Carbon with Contact",
        rationale=(
            "Explicit inter-particle contact resistance (R1) for powder or "
            "ink-cast carbon catalysts. R2/CPE1: outer surface; "
            "R3/CPE2: inner pore network."
        ),
    )

    DEFAULT = RANDLES_SIMPLE


# Alloy-specific model added after class definition to keep the class body
# clean; ALLOY_TWO_STEP is still a proper class attribute of CircuitCatalog.
CircuitCatalog.ALLOY_TWO_STEP = CircuitModel(
    notation="R0-p(R1,CPE1)-p(R2,CPE2)-W1",
    name="Alloy Two-Step",
    rationale=(
        "R1/CPE1: CO-oxidation / dehydration step on Pt sites. "
        "R2/CPE2: bifunctional water-activation step on Ru (or second metal) sites. "
        "W1: bulk diffusion of alcohol to the electrode surface. "
        "Use for PtRu, PtSn, PdCu, and similar bimetallic systems."
    ),
)


# ── Lookup table: (catalyst_type, region, electrolyte) → CircuitModel ────────
_CIRCUIT_MAP: dict[tuple[str, str, str], CircuitModel] = {
    # ── Noble metals ──────────────────────────────────────────────────────────
    # Pre-onset: no significant Faradaic current; surface restructuring only.
    (CATALYST_NOBLE_METAL, "pre-onset",  "acidic"):   CircuitCatalog.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "acidic"):   CircuitCatalog.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "post-onset", "acidic"):   CircuitCatalog.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "pre-onset",  "alkaline"): CircuitCatalog.TWO_RC,
    (CATALYST_NOBLE_METAL, "onset",      "alkaline"): CircuitCatalog.TWO_RC_WARBURG,
    (CATALYST_NOBLE_METAL, "post-onset", "alkaline"): CircuitCatalog.TWO_RC_WARBURG,

    # ── Alloys (bifunctional mechanism requires dedicated model) ──────────────
    (CATALYST_ALLOY, "pre-onset",  "acidic"):   CircuitCatalog.TWO_RC,
    (CATALYST_ALLOY, "onset",      "acidic"):   CircuitCatalog.ALLOY_TWO_STEP,
    (CATALYST_ALLOY, "post-onset", "acidic"):   CircuitCatalog.ALLOY_TWO_STEP,
    (CATALYST_ALLOY, "pre-onset",  "alkaline"): CircuitCatalog.TWO_RC,
    (CATALYST_ALLOY, "onset",      "alkaline"): CircuitCatalog.ALLOY_TWO_STEP,
    (CATALYST_ALLOY, "post-onset", "alkaline"): CircuitCatalog.ALLOY_TWO_STEP,

    # ── Metal oxides (pre-onset: capacitive only; onset+: two-step redox) ─────
    (CATALYST_METAL_OXIDE, "pre-onset",  "acidic"):   CircuitCatalog.RANDLES_SIMPLE,
    (CATALYST_METAL_OXIDE, "onset",      "acidic"):   CircuitCatalog.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "post-onset", "acidic"):   CircuitCatalog.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "pre-onset",  "alkaline"): CircuitCatalog.RANDLES_SIMPLE,
    (CATALYST_METAL_OXIDE, "onset",      "alkaline"): CircuitCatalog.OXIDE_TWO_STEP,
    (CATALYST_METAL_OXIDE, "post-onset", "alkaline"): CircuitCatalog.OXIDE_TWO_STEP,

    # ── Metal-free / carbon materials ─────────────────────────────────────────
    (CATALYST_METAL_FREE, "pre-onset",  "acidic"):   CircuitCatalog.POROUS_CARBON,
    (CATALYST_METAL_FREE, "onset",      "acidic"):   CircuitCatalog.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "post-onset", "acidic"):   CircuitCatalog.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "pre-onset",  "alkaline"): CircuitCatalog.POROUS_CARBON,
    (CATALYST_METAL_FREE, "onset",      "alkaline"): CircuitCatalog.POROUS_CARBON_CONTACT,
    (CATALYST_METAL_FREE, "post-onset", "alkaline"): CircuitCatalog.POROUS_CARBON_CONTACT,
}


def lookup_circuit(
    catalyst_type: str,
    region: str,
    electrolyte: str = "acidic",
) -> CircuitModel:
    """
    Return the recommended ``CircuitModel`` for the given experimental conditions.

    Args:
        catalyst_type: one of the ``CATALYST_*`` constants.
        region: ``"pre-onset"``, ``"onset"``, or ``"post-onset"``.
        electrolyte: ``"acidic"`` (default) or ``"alkaline"``.

    Returns:
        The matched ``CircuitModel``, or ``CircuitCatalog.DEFAULT`` with a
        warning when no specific entry exists for the combination.
    """
    key = (catalyst_type, region, electrolyte)
    model = _CIRCUIT_MAP.get(key)
    if model is None:
        logger.warning(
            "No specific circuit model found for %s. Falling back to default.",
            key,
        )
        return CircuitCatalog.DEFAULT
    return model


__all__ = [
    "CircuitModel",
    "CircuitCatalog",
    "lookup_circuit",
]
