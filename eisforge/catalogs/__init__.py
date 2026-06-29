"""
eisforge.catalogs
=================
Curated scientific catalogs used across EISForge modules.

Currently exposes:
    circuit_models  — equivalent-circuit registry for EIS fitting
"""
from eisforge.catalogs.circuit_models import (
    CircuitModel,
    StandardCircuits,
    CIRCUIT_MAP,
    lookup_circuit,
)

__all__ = [
    "CircuitModel",
    "StandardCircuits",
    "CIRCUIT_MAP",
    "lookup_circuit",
]
