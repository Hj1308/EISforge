"""
Physical Property Database for Electrochemistry.
Author: Hoda Jafari | Updated: June 2026
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Boundaries for the linear temperature-correction approximation.
# Outside 15-45 °C the linear model (1 + coeff*ΔT) diverges from
# Stokes-Einstein / Arrhenius behaviour; a warning is issued once per instance.
_LINEAR_RANGE_LOW_C: float = 15.0
_LINEAR_RANGE_HIGH_C: float = 45.0

# Hard physical limits – values outside these are almost certainly user errors.
_PHYSICAL_TEMP_MIN_C: float = -10.0
_PHYSICAL_TEMP_MAX_C: float = 120.0


class PhysicalPropertyDB:
    """
    Provides temperature-corrected transport properties for
    electrochemical calculations (Levich, Cottrell, etc.).

    All diffusion coefficients are *infinite-dilution* values in pure water.
    In real electrolytes (1 M KOH, 0.5 M H₂SO₄ …) the effective D can be
    5-15× lower; use ``custom_D`` to override with measured values.
    """

    def __init__(self, config_path: Optional[Path | str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "properties.json"
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Properties file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._default_temp: float = self._data.get("default_temperature_C", 25.0)

        # Backward-compatible read: new JSON stores {"value": ...}, old JSON
        # stores the float directly.
        raw_coeff = self._data.get("viscosity_temp_coeff_per_C", -0.021)
        if isinstance(raw_coeff, dict):
            self._viscosity_temp_coeff: float = raw_coeff.get("value", -0.021)
        else:
            self._viscosity_temp_coeff = float(raw_coeff)

        # Flag: emit the linear-range warning at most once per instance.
        self._linear_range_warned: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_temperature(self, temperature_C: float) -> None:
        """
        Raise ValueError for non-finite or physically implausible temperatures.
        Emit a one-time logger.warning when outside the linear-approximation
        range (15-45 °C) so callers are aware of reduced accuracy.
        """
        if not math.isfinite(temperature_C):
            raise ValueError(
                f"temperature_C must be a finite number, got {temperature_C!r}"
            )
        if not (_PHYSICAL_TEMP_MIN_C <= temperature_C <= _PHYSICAL_TEMP_MAX_C):
            raise ValueError(
                f"temperature_C={temperature_C}°C is outside the physically "
                f"valid range [{_PHYSICAL_TEMP_MIN_C}, {_PHYSICAL_TEMP_MAX_C}]°C."
            )
        if (
            not self._linear_range_warned
            and not (_LINEAR_RANGE_LOW_C <= temperature_C <= _LINEAR_RANGE_HIGH_C)
        ):
            logger.warning(
                "Temperature %.1f°C is outside the linear-approximation range "
                "(%g–%g°C). The linear correction (1 + coeff*ΔT) may diverge "
                "from Stokes-Einstein / Arrhenius behaviour. Consider supplying "
                "custom_D / custom_nu, or applying an Arrhenius correction.",
                temperature_C,
                _LINEAR_RANGE_LOW_C,
                _LINEAR_RANGE_HIGH_C,
            )
            self._linear_range_warned = True

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
        Return the infinite-dilution diffusion coefficient in cm²/s.

        Args:
            alcohol: key in ``diffusion_infinite_dilution_cm2_per_s``
                     (e.g. ``"methanol"``, ``"ethanol"``).
            temperature_C: measurement temperature in °C.
            custom_D: if provided, bypasses the database entirely and
                      returns this value directly (useful for measured D
                      in real electrolyte).

        Returns:
            D(T) in cm²/s.

        Note:
            Infinite-dilution values are ~5-15× higher than effective D
            measured electrochemically in 1 M KOH or H₂SO₄ media.
            Use ``custom_D`` when an experimental value is available.
        """
        if custom_D is not None:
            return custom_D

        self._validate_temperature(temperature_C)

        alcohol_data = (
            self._data
            .get("diffusion_infinite_dilution_cm2_per_s", {})
            .get(alcohol)
        )
        if alcohol_data is None:
            logger.warning(
                "Diffusion coefficient for '%s' not found in database. "
                "Falling back to 1.0e-5 cm²/s.",
                alcohol,
            )
            return 1.0e-5

        if isinstance(alcohol_data, dict):
            D25 = float(alcohol_data.get("value_25C", 1.0e-5))
            coeff = float(alcohol_data.get("temp_coeff_per_C", 0.020))
            # Only debug-level: this note may appear in every call inside
            # a sweep loop; info-level would spam production logs.
            if "_electrochemical_note" in alcohol_data:
                logger.debug(
                    "Electrochemical note for %s: %s",
                    alcohol,
                    alcohol_data["_electrochemical_note"],
                )
        else:
            # Backward compatibility: plain float in old JSON format.
            D25 = float(alcohol_data)
            coeff = 0.020

        delta_T = temperature_C - self._default_temp
        D_T = D25 * (1.0 + coeff * delta_T)

        if D_T <= 0:
            raise ValueError(
                f"Computed D({temperature_C}°C) = {D_T:.3e} cm²/s for "
                f"'{alcohol}' is non-positive. Check temp_coeff_per_C "
                f"or supply custom_D."
            )
        return D_T

    def get_kinematic_viscosity(
        self,
        electrolyte_key: str,
        temperature_C: float,
        custom_nu: Optional[float] = None,
    ) -> float:
        """
        Return the kinematic viscosity in cm²/s.

        Args:
            electrolyte_key: key in ``kinematic_viscosity_cm2_per_s``
                             (e.g. ``"KOH_1M"``, ``"H2SO4_05M"``).
            temperature_C: measurement temperature in °C.
            custom_nu: if provided, bypasses the database entirely.

        Returns:
            ν(T) in cm²/s.
        """
        if custom_nu is not None:
            return custom_nu

        self._validate_temperature(temperature_C)

        visc_data = self._data.get("kinematic_viscosity_cm2_per_s", {})
        default_item = visc_data.get("default", {"value": 0.00893})

        item = visc_data.get(electrolyte_key)
        if item is None:
            logger.warning(
                "Electrolyte key '%s' not found in viscosity table. "
                "Using default (pure water at 25°C).",
                electrolyte_key,
            )
            item = default_item

        if isinstance(item, dict):
            nu25 = float(item.get("value", 0.00893))
            if "note" in item:
                logger.debug(
                    "Viscosity note for %s: %s", electrolyte_key, item["note"]
                )
        else:
            # Backward compatibility: plain float in old JSON format.
            nu25 = float(item)

        delta_T = temperature_C - self._default_temp
        nu_T = nu25 * (1.0 + self._viscosity_temp_coeff * delta_T)

        if nu_T <= 0:
            raise ValueError(
                f"Computed ν({temperature_C}°C) = {nu_T:.4e} cm²/s for "
                f"'{electrolyte_key}' is non-positive. "
                f"Temperature may be out of the model’s valid range."
            )
        return nu_T

    def get_levich_base(
        self,
        alcohol: str,
        electrolyte_key: str,
        concentration_M: float,
        temperature_C: float,
        n_electrons: int = 1,
        electrode_area_cm2: float = 1.0,
        D_custom: Optional[float] = None,
        nu_custom: Optional[float] = None,
    ) -> float:
        """
        Calculate the Levich B coefficient::

            B = 0.620 * n * F * A * D^(2/3) * nu^(-1/6) * C

        so that the diffusion-limited current is::

            I_L [mA] = B * omega^(1/2)   (omega in rad/s)

        Args:
            alcohol: fuel molecule key (see ``get_diffusion_coeff``).
            electrolyte_key: electrolyte key (see ``get_kinematic_viscosity``).
            concentration_M: bulk concentration in mol/L.
            temperature_C: temperature in °C.
            n_electrons: electrons transferred per molecule
                         (e.g. 6 for full methanol oxidation,
                          12 for full ethanol oxidation).
            electrode_area_cm2: geometric electrode area in cm².
            D_custom: override diffusion coefficient (cm²/s).
            nu_custom: override kinematic viscosity (cm²/s).

        Returns:
            B in mA·s^(1/2).
        """
        if concentration_M <= 0:
            raise ValueError(
                f"concentration_M must be positive, got {concentration_M}"
            )
        if n_electrons < 1:
            raise ValueError(
                f"n_electrons must be ≥ 1, got {n_electrons}"
            )
        if electrode_area_cm2 <= 0:
            raise ValueError(
                f"electrode_area_cm2 must be positive, got {electrode_area_cm2}"
            )

        D = self.get_diffusion_coeff(alcohol, temperature_C, D_custom)
        nu = self.get_kinematic_viscosity(electrolyte_key, temperature_C, nu_custom)

        C_mol_per_cm3 = concentration_M * 1.0e-3
        FARADAY = 96485.0  # C mol⁻¹

        B = (
            0.620
            * n_electrons
            * FARADAY
            * electrode_area_cm2
            * (D ** (2.0 / 3.0))
            * (nu ** (-1.0 / 6.0))
            * C_mol_per_cm3
        )
        return B * 1000.0  # A·s^(1/2) → mA·s^(1/2)
