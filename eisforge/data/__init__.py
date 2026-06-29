"""
eisforge.data
=============
Central physical-property database for electrochemical calculations.

Exposes:
    PhysicalPropertyDB  — diffusion coefficients, kinematic viscosities,
                          Levich base constant, with linear temperature correction.
"""
from eisforge.data.physical_properties import PhysicalPropertyDB

__all__ = ["PhysicalPropertyDB"]
