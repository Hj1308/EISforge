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
# Outside 15-45 °C the linear model (1 + coeff*ΔT) diverges noticeably from
# Stokes-Einstein / Arrhenius behaviour.  Beyond this range the Arrhenius
# path (get_diffusion_arrhenius) should be preferred.
_LINEAR_RANGE_LOW_C: float = 15.0
_LINEAR_RANGE_HIGH_C: float = 45.0

# Hard physical limits – values outside these are almost certainly user errors.
_PHYSICAL_TEMP_MIN_C: float = -10.0
_PHYSICAL_TEMP_MAX_C: float = 120.0

# Concentration above which the electrolyte viscosity may deviate significantly
# from the pure-solvent values stored in the database.
_CONCENTRATION_WARNING_THRESHOLD: float = 2.0  # mol/L

# Universal gas constant (J mol⁻¹ K⁻¹)
_R_GAS: float = 8.31446

# Faraday constant (C mol⁻¹)
_FARADAY: float = 96485.0


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _celsius_to_kelvin(temperature_C: float) -> float:
    return temperature_C + 273.15


class PhysicalPropertyDB:
    """
    Provides temperature-corrected transport properties for
    electrochemical calculations (Levich, Cottrell, etc.).

    All diffusion coefficients are *infinite-dilution* values in pure water.
    In real electrolytes (1 M KOH, 0.5 M H₂SO₄ …) the effective D can be
    5-15× lower; use ``custom_D`` to override with measured values.

    Parameters
    ----------
    config_path : Path or str, optional
        Path to the ``properties.json`` data file.  Defaults to
        ``<package>/data/properties.json`` next to this module.

    Examples
    --------
    >>> db = PhysicalPropertyDB()
    >>> db.get_diffusion_coeff("methanol", 30.0)
    1.28e-05  # approximate
    >>> db.list_alcohols()
    ['methanol', 'ethanol', ...]
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
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        n_alcohols = len(
            self._data.get("diffusion_infinite_dilution_cm2_per_s", {})
        )
        n_electrolytes = len(
            self._data.get("kinematic_viscosity_cm2_per_s", {})
        )
        return (
            f"PhysicalPropertyDB("
            f"alcohols={n_alcohols}, "
            f"electrolytes={n_electrolytes}, "
            f"T_default={self._default_temp}°C)"
        )

    def __len__(self) -> int:
        """Return total number of entries (alcohols + electrolytes)."""
        return len(
            self._data.get("diffusion_infinite_dilution_cm2_per_s", {})
        ) + len(
            self._data.get("kinematic_viscosity_cm2_per_s", {})
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_temperature(self, temperature_C: float) -> None:
        """
        Raise ``ValueError`` for non-finite or physically implausible
        temperatures.  Emit a one-time ``logger.warning`` when outside the
        linear-approximation range (15-45 °C) so callers are aware of
        reduced accuracy.
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
                "from Stokes-Einstein / Arrhenius behaviour. Consider using "
                "get_diffusion_arrhenius(), or supplying custom_D / custom_nu.",
                temperature_C,
                _LINEAR_RANGE_LOW_C,
                _LINEAR_RANGE_HIGH_C,
            )
            self._linear_range_warned = True

    def _validate_concentration(
        self, concentration_M: float, context: str = "concentration_M"
    ) -> None:
        """
        Raise ``ValueError`` if *concentration_M* is non-positive.
        Emit a ``logger.warning`` if it exceeds the viscosity-deviation
        threshold.
        """
        if concentration_M <= 0:
            raise ValueError(
                f"{context} must be positive, got {concentration_M}"
            )
        if concentration_M > _CONCENTRATION_WARNING_THRESHOLD:
            logger.warning(
                "Concentration %.2f M exceeds %.1f M. The solution viscosity "
                "may deviate substantially from database values. "
                "Consider supplying custom_nu measured for your solution.",
                concentration_M,
                _CONCENTRATION_WARNING_THRESHOLD,
            )

    @staticmethod
    def _resolve_dict_or_float(
        entry, value_key: str, fallback: float
    ) -> tuple[float, dict]:
        """
        Return (scalar_value, full_dict) from a JSON entry that is either a
        plain float (old format) or a dict containing *value_key* (new format).
        """
        if isinstance(entry, dict):
            return float(entry.get(value_key, fallback)), entry
        return float(entry), {}

    # ------------------------------------------------------------------
    # Discoverability helpers
    # ------------------------------------------------------------------

    def list_alcohols(self) -> list[str]:
        """Return all alcohol keys present in the diffusion table."""
        return list(
            self._data.get("diffusion_infinite_dilution_cm2_per_s", {}).keys()
        )

    def list_electrolytes(self) -> list[str]:
        """Return all electrolyte keys present in the viscosity table."""
        return list(
            self._data.get("kinematic_viscosity_cm2_per_s", {}).keys()
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
        Return the infinite-dilution diffusion coefficient (cm²/s) using a
        **linear** temperature correction.

        For measurements outside 15-45 °C, prefer
        :meth:`get_diffusion_arrhenius` which uses an Arrhenius model and is
        more physically accurate across a wider temperature range.

        Parameters
        ----------
        alcohol : str
            Key in ``diffusion_infinite_dilution_cm2_per_s``
            (e.g. ``"methanol"``, ``"ethanol"``).
        temperature_C : float
            Measurement temperature in °C.
        custom_D : float, optional
            If provided, bypasses the database entirely and returns this value
            directly (useful for a measured D in a real electrolyte).

        Returns
        -------
        float
            D(T) in cm²/s.

        Raises
        ------
        ValueError
            If *temperature_C* is non-finite, outside the physical range, or
            if the computed D(T) is non-positive.

        Notes
        -----
        Infinite-dilution values are ~5-15× higher than effective D measured
        electrochemically in 1 M KOH or H₂SO₄ media.  Use ``custom_D`` when
        an experimental value is available.
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
            available = self.list_alcohols()
            logger.warning(
                "Diffusion coefficient for '%s' not found in database "
                "(available: %s). Falling back to 1.0e-5 cm²/s.",
                alcohol,
                available,
            )
            return 1.0e-5

        D25, meta = self._resolve_dict_or_float(
            alcohol_data, "value_25C", 1.0e-5
        )
        coeff = float(meta.get("temp_coeff_per_C", 0.020)) if meta else 0.020

        if meta and "_electrochemical_note" in meta:
            logger.debug(
                "Electrochemical note for %s: %s",
                alcohol,
                meta["_electrochemical_note"],
            )

        delta_T = temperature_C - self._default_temp
        D_T = D25 * (1.0 + coeff * delta_T)

        if D_T <= 0:
            raise ValueError(
                f"Computed D({temperature_C}°C) = {D_T:.3e} cm²/s for "
                f"'{alcohol}' is non-positive. Check temp_coeff_per_C "
                f"or supply custom_D."
            )
        return D_T

    def get_diffusion_arrhenius(
        self,
        alcohol: str,
        temperature_C: float,
        custom_D: Optional[float] = None,
    ) -> float:
        """
        Return the diffusion coefficient (cm²/s) using an **Arrhenius**
        temperature correction::

            D(T) = D_ref * exp(-Ea/R * (1/T - 1/T_ref))

        This model is more physically accurate than the linear correction
        outside the 15-45 °C range and is preferred for high-temperature
        or low-temperature experiments.

        Parameters
        ----------
        alcohol : str
            Key in ``diffusion_infinite_dilution_cm2_per_s``.  The entry
            should supply ``activation_energy_J_per_mol``; if absent,
            a typical aqueous-diffusion value of 17 000 J mol⁻¹ is used.
        temperature_C : float
            Measurement temperature in °C.
        custom_D : float, optional
            If provided, bypasses the database and returns this value directly.

        Returns
        -------
        float
            D(T) in cm²/s.

        Raises
        ------
        ValueError
            If *temperature_C* is non-finite, outside the physical range, or
            if the computed D(T) is non-positive.
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
            available = self.list_alcohols()
            logger.warning(
                "Diffusion coefficient for '%s' not found in database "
                "(available: %s). Falling back to 1.0e-5 cm²/s.",
                alcohol,
                available,
            )
            return 1.0e-5

        D_ref, meta = self._resolve_dict_or_float(
            alcohol_data, "value_25C", 1.0e-5
        )
        # Typical activation energy for small molecule diffusion in water
        Ea = float(meta.get("activation_energy_J_per_mol", 17000.0)) if meta else 17000.0

        T_ref_K = _celsius_to_kelvin(self._default_temp)
        T_K = _celsius_to_kelvin(temperature_C)

        D_T = D_ref * math.exp(-Ea / _R_GAS * (1.0 / T_K - 1.0 / T_ref_K))

        if D_T <= 0:
            raise ValueError(
                f"Arrhenius D({temperature_C}°C) = {D_T:.3e} cm²/s for "
                f"'{alcohol}' is non-positive. Verify activation_energy_J_per_mol."
            )
        return D_T

    def get_kinematic_viscosity(
        self,
        electrolyte_key: str,
        temperature_C: float,
        custom_nu: Optional[float] = None,
    ) -> float:
        """
        Return the kinematic viscosity (cm²/s).

        Parameters
        ----------
        electrolyte_key : str
            Key in ``kinematic_viscosity_cm2_per_s``
            (e.g. ``"KOH_1M"``, ``"H2SO4_05M"``).
        temperature_C : float
            Measurement temperature in °C.
        custom_nu : float, optional
            If provided, bypasses the database entirely.

        Returns
        -------
        float
            ν(T) in cm²/s.

        Raises
        ------
        ValueError
            If *temperature_C* is invalid or the computed ν(T) is
            non-positive (indicating the model has been pushed beyond its
            valid range).
        """
        if custom_nu is not None:
            return custom_nu

        self._validate_temperature(temperature_C)

        visc_data = self._data.get("kinematic_viscosity_cm2_per_s", {})
        default_item = visc_data.get("default", {"value": 0.00893})

        item = visc_data.get(electrolyte_key)
        if item is None:
            available = self.list_electrolytes()
            logger.warning(
                "Electrolyte key '%s' not found in viscosity table "
                "(available: %s). Using default (pure water at 25°C).",
                electrolyte_key,
                available,
            )
            item = default_item

        nu25, meta = self._resolve_dict_or_float(item, "value", 0.00893)
        if meta and "note" in meta:
            logger.debug(
                "Viscosity note for %s: %s", electrolyte_key, meta["note"]
            )

        delta_T = temperature_C - self._default_temp
        nu_T = nu25 * (1.0 + self._viscosity_temp_coeff * delta_T)

        if nu_T <= 0:
            raise ValueError(
                f"Computed ν({temperature_C}°C) = {nu_T:.4e} cm²/s for "
                f"'{electrolyte_key}' is non-positive. "
                f"Temperature may be outside the model's valid range. "
                f"Consider using custom_nu."
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
        use_arrhenius: bool = False,
    ) -> float:
        """
        Calculate the Levich B coefficient::

            B = 0.620 · n · F · A · D^(2/3) · ν^(-1/6) · C

        so that the diffusion-limited current is::

            I_L [mA] = B · ω^(1/2)   (ω in rad/s)

        Parameters
        ----------
        alcohol : str
            Fuel molecule key (see :meth:`get_diffusion_coeff`).
        electrolyte_key : str
            Electrolyte key (see :meth:`get_kinematic_viscosity`).
        concentration_M : float
            Bulk concentration in mol/L.
        temperature_C : float
            Temperature in °C.
        n_electrons : int
            Electrons transferred per molecule (e.g. 6 for full methanol
            oxidation, 12 for full ethanol oxidation).
        electrode_area_cm2 : float
            Geometric electrode area in cm².
        D_custom : float, optional
            Override diffusion coefficient (cm²/s).
        nu_custom : float, optional
            Override kinematic viscosity (cm²/s).
        use_arrhenius : bool
            If ``True``, use :meth:`get_diffusion_arrhenius` for D(T) instead
            of the linear model.  Recommended outside 15-45 °C.

        Returns
        -------
        float
            B in mA·s^(1/2).

        Raises
        ------
        ValueError
            If *concentration_M*, *n_electrons*, or *electrode_area_cm2*
            are non-positive.
        """
        self._validate_concentration(concentration_M)

        if n_electrons < 1:
            raise ValueError(
                f"n_electrons must be ≥ 1, got {n_electrons}"
            )
        if electrode_area_cm2 <= 0:
            raise ValueError(
                f"electrode_area_cm2 must be positive, got {electrode_area_cm2}"
            )

        if use_arrhenius:
            D = self.get_diffusion_arrhenius(alcohol, temperature_C, D_custom)
        else:
            D = self.get_diffusion_coeff(alcohol, temperature_C, D_custom)
        nu = self.get_kinematic_viscosity(electrolyte_key, temperature_C, nu_custom)

        C_mol_per_cm3 = concentration_M * 1.0e-3

        B = (
            0.620
            * n_electrons
            * _FARADAY
            * electrode_area_cm2
            * (D ** (2.0 / 3.0))
            * (nu ** (-1.0 / 6.0))
            * C_mol_per_cm3
        )
        return B * 1000.0  # A·s^(1/2) → mA·s^(1/2)

    def get_reynolds_number(
        self,
        angular_velocity_rad_s: float,
        electrode_radius_cm: float,
        temperature_C: float,
        electrolyte_key: str,
        custom_nu: Optional[float] = None,
    ) -> float:
        """
        Compute the Reynolds number for a rotating disk electrode (RDE)::

            Re = (ω · r²) / ν

        For a smooth rotating disk, the flow is laminar for Re < 200 000.
        This method helps verify that the Levich equation (valid for laminar
        flow only) is applicable to the chosen rotation speed.

        Parameters
        ----------
        angular_velocity_rad_s : float
            Rotation speed in rad/s.
        electrode_radius_cm : float
            Radius of the disk electrode in cm.
        temperature_C : float
            Temperature in °C.
        electrolyte_key : str
            Electrolyte key (see :meth:`get_kinematic_viscosity`).
        custom_nu : float, optional
            Optional override of kinematic viscosity.

        Returns
        -------
        float
            Reynolds number (dimensionless).

        Raises
        ------
        ValueError
            If *angular_velocity_rad_s* or *electrode_radius_cm* are
            non-positive.
        """
        if angular_velocity_rad_s <= 0:
            raise ValueError(
                f"angular_velocity_rad_s must be positive, "
                f"got {angular_velocity_rad_s}"
            )
        if electrode_radius_cm <= 0:
            raise ValueError(
                f"electrode_radius_cm must be positive, "
                f"got {electrode_radius_cm}"
            )

        nu = self.get_kinematic_viscosity(
            electrolyte_key, temperature_C, custom_nu
        )
        re = (angular_velocity_rad_s * (electrode_radius_cm ** 2)) / nu

        if re >= 2.0e5:
            logger.warning(
                "Reynolds number Re=%.0f ≥ 200 000: flow may be turbulent. "
                "The Levich equation assumes laminar flow (Re < 200 000).",
                re,
            )
        return re

    def to_dict(self) -> dict:
        """Return a shallow copy of the raw database dictionary."""
        return dict(self._data)
