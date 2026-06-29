"""
Physical Property Database for Electrochemistry.
Author: Hoda Jafari | May 2026

Manages diffusion coefficients and kinematic viscosities with temperature correction.

All returned values are in CGS units:
    D  : cm\u00b2/s  (diffusion coefficient)
    nu : cm\u00b2/s  (kinematic viscosity = dynamic / density)

Temperature correction uses a **linear approximation**:
    X(T) = X(25\u00b0C) \u00d7 [1 + coeff \u00d7 (T - 25)]

Valid range: 20\u201360\u00b0C. For wider ranges, use the Andrade equation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# JSON key for kinematic viscosity (updated from old 'viscosity' key)
_VISC_KEY = "kinematic_viscosity_cm2_per_s"


class PhysicalPropertyDB:
    """
    Central database for physical properties (D, \u03bd) with temperature correction.

    Loads data from a JSON configuration file. Custom values provided at call
    time override the database values and are returned without any temperature
    correction (the caller is assumed to have already corrected them).

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the JSON configuration file.  Defaults to
        ``<this_file_directory>/properties.json``.
    """

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "properties.json"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Properties file not found: {self.config_path}"
            )

        with open(self.config_path, "r") as fh:
            self._data = json.load(fh)

        self._default_temp: float = self._data.get("default_temperature", 25.0)
        self._visc_temp_coeff: float = self._data.get(
            "viscosity_temp_coeff_per_C", -0.021
        )

        # Support both the old key name and the new explicit key
        if _VISC_KEY in self._data:
            self._visc_table: dict = self._data[_VISC_KEY]
        elif "viscosity" in self._data:
            logger.warning(
                "properties.json uses deprecated key 'viscosity'. "
                "Rename it to 'kinematic_viscosity_cm2_per_s' and verify "
                "values are kinematic (cm\u00b2/s), not dynamic (Pa\u00b7s)."
            )
            self._visc_table = self._data["viscosity"]
        else:
            raise KeyError(
                "properties.json must contain 'kinematic_viscosity_cm2_per_s'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_diffusion_coeff(
        self,
        alcohol: str,
        temperature_C: float,
        custom_D: Optional[float] = None,
    ) -> float:
        """
        Return diffusion coefficient (cm\u00b2/s) at *temperature_C*.

        If *custom_D* is provided it is returned as-is (no correction).
        Falls back to ``1.0e-5 cm\u00b2/s`` with a warning when the alcohol
        key is absent from the database.
        """
        if custom_D is not None:
            return float(custom_D)

        entry = self._data["diffusion"].get(alcohol)
        if entry is None:
            logger.warning(
                "Diffusion coefficient for '%s' not in DB. "
                "Falling back to 1.0e-5 cm\u00b2/s.",
                alcohol,
            )
            return 1.0e-5

        D25: float = entry["value_25C"]
        coeff: float = entry.get("temp_coeff_per_C", 0.020)
        delta_T = temperature_C - self._default_temp
        D_T = D25 * (1.0 + coeff * delta_T)

        if D_T <= 0:
            logger.warning(
                "Temperature correction yielded D <= 0 at %.1f\u00b0C for '%s'. "
                "Clamping to 1e-8 cm\u00b2/s. Consider using a custom value.",
                temperature_C,
                alcohol,
            )
            return 1e-8

        return D_T

    def get_viscosity(
        self,
        electrolyte_key: str,
        temperature_C: float,
        custom_nu: Optional[float] = None,
    ) -> float:
        """
        Return **kinematic** viscosity (cm\u00b2/s) at *temperature_C*.

        If *custom_nu* is provided it is returned as-is.

        .. note::
            The temperature correction is a linear approximation valid only
            between 20\u201360\u00b0C.  Outside this range provide *custom_nu*
            or use the Andrade / WLF equation externally.
        """
        if custom_nu is not None:
            return float(custom_nu)

        nu25: float = self._visc_table.get(
            electrolyte_key, self._visc_table["default"]
        )
        delta_T = temperature_C - self._default_temp
        nu_T = nu25 * (1.0 + self._visc_temp_coeff * delta_T)

        if nu_T <= 0:
            logger.warning(
                "Viscosity correction yielded nu <= 0 at %.1f\u00b0C for '%s'. "
                "Clamping to 0.001 cm\u00b2/s.",
                temperature_C,
                electrolyte_key,
            )
            return 0.001

        return nu_T

    def get_levich_base(
        self,
        alcohol: str,
        electrolyte_key: str,
        concentration_M: float,
        temperature_C: float,
        D_custom: Optional[float] = None,
        nu_custom: Optional[float] = None,
    ) -> float:
        """
        Calculate the **n=1 Levich prefactor** B\u2080 (mA\u00b7s\u00b2/cm\u00b2).

        .. math::

            B_0 = 0.62 \\cdot F \\cdot D^{2/3} \\cdot \\nu^{-1/6} \\cdot C

        Multiply by *n* (number of electrons) to obtain the full Levich slope.

        Returns
        -------
        float
            B\u2080 in mA\u00b7s^0.5/cm\u00b2.
        """
        D = self.get_diffusion_coeff(alcohol, temperature_C, D_custom)
        nu = self.get_viscosity(electrolyte_key, temperature_C, nu_custom)
        C_mol_per_cm3 = concentration_M * 1e-3  # mol/L \u2192 mol/cm\u00b3

        FARADAY = 96485.0  # C/mol

        B0_A = (
            0.62
            * FARADAY
            * (D ** (2.0 / 3.0))
            * (nu ** (-1.0 / 6.0))
            * C_mol_per_cm3
        )
        return B0_A * 1000.0  # A \u2192 mA

    def to_dict(self) -> dict:
        """Return the raw database dictionary for inspection."""
        return dict(self._data)
