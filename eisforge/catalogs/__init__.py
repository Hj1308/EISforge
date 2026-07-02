"""
eisforge.catalogs
=================
Curated scientific catalogs used across EISForge modules.

Currently exposes:
    circuit_models  — equivalent-circuit registry for EIS fitting

NOTE: this __init__ previously imported names (``StandardCircuits``,
``CIRCUIT_MAP``) that never existed in circuit_models.py, so
``import eisforge.catalogs`` raised ImportError since the package was
created. Fixed to export the real API; ``StandardCircuits`` is kept as a
backward-compatible alias of ``CircuitCatalog``.
"""
from eisforge.catalogs.circuit_models import (
    CircuitModel,
    CircuitCatalog,
    lookup_circuit,
)

# Backward-compatible alias (the name the old broken __init__ promised).
StandardCircuits = CircuitCatalog

__all__ = [
    "CircuitModel",
    "CircuitCatalog",
    "StandardCircuits",
    "lookup_circuit",
]
