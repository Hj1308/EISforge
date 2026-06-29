"""
eisforge.data.physical_properties
==================================
Central database for physical constants used in electrochemical calculations.
Author: Hoda Jafari | June 2026

Handles:
    * Diffusion coefficients D(T) with linear temperature correction
    * Kinematic viscosities nu(T) with linear temperature correction
    * Levich base constant B_base for n = 1

Temperature model (linear approximation, valid ±30°C around 25°C):
    X(T) = X(25°C) × [1 + coeff × (T - 25)]

All values loaded from ``properties.json`` in the same directory.
Users may supply a custom JSON path or override individual values via
``D_custom`` / ``nu_custom`` keyword arguments.

Typical usage
-------------
    from eisforge.data.physical_properties import PhysicalPropertyDB

    db = PhysicalPropertyDB()           # uses bundled properties.json
    D  = db.get_diffusion_coeff("ethanol", temperature_C=40)
    B  = db.get_levich_base("ethanol", "KOH_1M", concentration_M=1.0, temperature_C=40)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Faraday constant (C/mol)
_FARADAY = 96485.0


class PhysicalPropertyDB:
    """
    Central database for physical properties with temperature correction.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the JSON configuration file.  Defaults to
        ``properties.json`` located in the same directory as this module.
    """

    def __init__(self, config_path: Optional[Union[Path, str]] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "properties.json"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Properties file not found: {self.config_path}. "
                "Ensure eisforge/data/properties.json is present."
            )

        with self.config_path.open("r", encoding="utf-8") as fh:
            self._data = json.load(fh)

        self._default_temp: float = float(self._data.get("default_temperature", 25.0))
        self._nu_temp_coeff: float = float(
            self._data.get("viscosity_temp_coeff_per_C", -0.021)
        )
        logger.debug(
            "PhysicalPropertyDB: loaded %d alcohol entries from %s",
            len(self._data.get("diffusion", {})),
            self.config_path,
        )

    # ------------------------------------------------------------------
    # Diffusion coefficient
    # ------------------------------------------------------------------

    def get_diffusion_coeff(
        self,
        alcohol: str,
        temperature_C: float,
        custom_D: Optional[float] = None,
    ) -> float:
        """
        Return diffusion coefficient D (cm²/s) at *temperature_C*.

        If *custom_D* is supplied it is returned as-is (no T-correction).
        Falls back to 1.0e-5 cm²/s with a warning when the alcohol is not
        in the database.

        Parameters
        ----------
        alcohol : str
            Alcohol name, e.g. ``'ethanol'``, ``'methanol'``.
        temperature_C : float
            Measurement temperature in degrees Celsius.
        custom_D : float, optional
            Override value (cm²/s).  Bypasses the database entirely.

        Returns
        -------
        float
            Diffusion coefficient in cm²/s (always > 0).
        """
        if custom_D is not None:
            return float(custom_D)

        entry = self._data["diffusion"].get(alcohol)
        if entry is None:
            logger.warning(
                "PhysicalPropertyDB: diffusion coefficient for '%s' not found. "
                "Using fallback 1.0e-5 cm²/s. "
                "Add an entry to properties.json to suppress this warning.",
                alcohol,
            )
            return 1.0e-5

        D25: float = float(entry["value_25C"])
        coeff: float = float(entry.get("temp_coeff_per_C", 0.020))
        D_T = D25 * (1.0 + coeff * (temperature_C - self._default_temp))
        # Safety floor: physically impossible to have D <= 0
        return max(D_T, 1.0e-8)

    # ------------------------------------------------------------------
    # Kinematic viscosity
    # ------------------------------------------------------------------

    def get_viscosity(
        self,
        electrolyte_key: str,
        temperature_C: float,
        custom_nu: Optional[float] = None,
    ) -> float:
        """
        Return kinematic viscosity nu (cm²/s) at *temperature_C*.

        If *custom_nu* is supplied it is returned as-is.
        Unknown electrolyte keys fall back to the ``'default'`` entry.

        Parameters
        ----------
        electrolyte_key : str
            Key from the ``viscosity`` section of ``properties.json``,
            e.g. ``'KOH_1M'``, ``'H2SO4_05M'``.
        temperature_C : float
            Measurement temperature in degrees Celsius.
        custom_nu : float, optional
            Override value (cm²/s).

        Returns
        -------
        float
            Kinematic viscosity in cm²/s (always > 0).
        """
        if custom_nu is not None:
            return float(custom_nu)

        visc_map = self._data.get("viscosity", {})
        nu25: float = float(visc_map.get(electrolyte_key, visc_map.get("default", 0.01007)))

        if electrolyte_key not in visc_map:
            logger.warning(
                "PhysicalPropertyDB: viscosity key '%s' not found; using default %.5f cm²/s.",
                electrolyte_key,
                nu25,
            )

        nu_T = nu25 * (1.0 + self._nu_temp_coeff * (temperature_C - self._default_temp))
        return max(nu_T, 1.0e-5)  # absolute floor: liquid water never goes below ~0.004 cm²/s

    # ------------------------------------------------------------------
    # Levich base constant
    # ------------------------------------------------------------------

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
        Calculate the Levich base constant B for n = 1 (mA·s^0.5/cm²).

        The full Levich constant for *n* electrons is ``n × B_base``.

        Formula::

            B_base = 0.62 × F × D^(2/3) × ν^(−1/6) × C   [A·s^0.5/cm²]

        Returned in **mA·s^0.5/cm²** (×1000 conversion applied).

        Parameters
        ----------
        alcohol : str
        electrolyte_key : str
        concentration_M : float
            Bulk alcohol concentration in mol/L.
        temperature_C : float
        D_custom : float, optional
        nu_custom : float, optional

        Returns
        -------
        float
            B_base in mA·s^0.5/cm².
        """
        D  = self.get_diffusion_coeff(alcohol, temperature_C, D_custom)
        nu = self.get_viscosity(electrolyte_key, temperature_C, nu_custom)
        C_mol_per_cm3 = float(concentration_M) * 1.0e-3  # mol/L → mol/cm³

        B_base_A = (
            0.62
            * _FARADAY
            * (D  ** (2.0 / 3.0))
            * (nu ** (-1.0 / 6.0))
            * C_mol_per_cm3
        )
        return B_base_A * 1000.0  # → mA·s^0.5/cm²

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def available_alcohols(self) -> list[str]:
        """Return list of alcohols in the database."""
        return list(self._data.get("diffusion", {}).keys())

    def available_electrolytes(self) -> list[str]:
        """Return list of electrolyte viscosity keys in the database."""
        return [k for k in self._data.get("viscosity", {}).keys() if k != "default"]

    def to_dict(self) -> dict:
        """Return a copy of the raw database for inspection."""
        import copy
        return copy.deepcopy(self._data)
