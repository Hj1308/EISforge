"""
EISForge — patch20: Band Edge Calculator & Mott-Schottky Analysis
=================================================================
Calculates semiconductor band edge positions (Ecb, Evb) from the
Mulliken electronegativity method and performs Mott-Schottky analysis
to determine flat-band potential (Vfb) and carrier density (Nd).

Band alignment method (Butler & Ginley 1978)
--------------------------------------------
    E_cb (vs vacuum) = χ − E_c − 0.5 × E_g
    E_vb (vs vacuum) = E_cb + E_g

where χ is the geometric mean of constituent atom electronegativities
(Mulliken scale, in eV), and E_c = 4.5 eV (standard absolute scale).

Conversion to electrochemical scales
-------------------------------------
    E vs NHE  = E_vacuum − 4.44  eV
    E vs RHE  = E vs NHE − 0.0592 × pH

Mott-Schottky
-------------
    1/C² = (2 / (e ε₀ ε_r A² N_d)) × (V − V_fb − kT/e)

Usage
-----
>>> from eisforge.analysis.band_edge_calculator import BandEdgeCalculator, MottSchottky
>>> calc = BandEdgeCalculator(material='g-C3N4', Eg_eV=2.7, pH=7.0)
>>> edges = calc.calculate()
>>> print(edges)

>>> ms = MottSchottky(epsilon_r=15.0, electrode_area_cm2=0.0707, pH=7.0)
>>> result = ms.analyze(V, C_F)
>>> print(result.Vfb_vs_RHE)

References
----------
* Butler & Ginley (1978) J. Electrochem. Soc. 125, 228.
* Xu & Schoonen (2000) Am. Mineral. 85, 543.
* Gelderman et al. (2007) J. Chem. Educ. 84, 685.
Author: Hoda Jafari | EISForge 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
_E_CHARGE  = 1.602176634e-19   # C
_EPS_0     = 8.8541878128e-12  # F/m
_KB        = 1.380649e-23      # J/K
_EC_ABS    = 4.44              # eV — NHE vs vacuum  (IUPAC, Trasatti 1986)
_EC_SCALE  = 4.5               # eV — Butler & Ginley 1978 reference


# ─────────────────────────────────────────────────────────────────────────────
# Built-in materials database
# ─────────────────────────────────────────────────────────────────────────────
# Format: chi (eV), Eg (eV), epsilon_r, n/p type
# chi values from Xu & Schoonen (2000) unless noted.
_MATERIALS_DB: Dict[str, dict] = {
    'g-C3N4':  {'chi': 4.73,  'Eg': 2.70, 'epsilon_r': 10.0,  'type': 'n',
                'notes': 'Xu & Schoonen 2000; Eg varies 2.5-2.9 eV by synthesis'},
    'TiO2':    {'chi': 5.81,  'Eg': 3.20, 'epsilon_r': 55.0,  'type': 'n',
                'notes': 'anatase; Eg 3.0 eV for rutile'},
    'ZnO':     {'chi': 5.79,  'Eg': 3.37, 'epsilon_r': 8.5,   'type': 'n',
                'notes': 'wurtzite'},
    'WO3':     {'chi': 6.59,  'Eg': 2.70, 'epsilon_r': 20.0,  'type': 'n',
                'notes': 'monoclinic'},
    'BiVO4':   {'chi': 6.04,  'Eg': 2.40, 'epsilon_r': 68.0,  'type': 'n',
                'notes': 'scheelite monoclinic'},
    'Fe2O3':   {'chi': 5.88,  'Eg': 2.10, 'epsilon_r': 12.0,  'type': 'n',
                'notes': 'hematite'},
    'CdS':     {'chi': 4.50,  'Eg': 2.40, 'epsilon_r': 8.9,   'type': 'n',
                'notes': 'wurtzite'},
    'BCN':     {'chi': 4.85,  'Eg': None, 'epsilon_r': 9.0,   'type': 'n',
                'notes': 'boron-carbon-nitride; Eg from Tauc plot required'},
    'CuO':     {'chi': 5.42,  'Eg': 1.40, 'epsilon_r': 10.0,  'type': 'p',
                'notes': 'p-type; low Eg'},
    'Cu2O':    {'chi': 5.04,  'Eg': 2.17, 'epsilon_r': 7.1,   'type': 'p',
                'notes': 'p-type'},
    'NiO':     {'chi': 6.16,  'Eg': 3.60, 'epsilon_r': 11.0,  'type': 'p',
                'notes': 'p-type'},
    'Co3O4':   {'chi': 6.08,  'Eg': 1.60, 'epsilon_r': 12.5,  'type': 'p',
                'notes': 'spinel; p-type dominant'},
}


# ─────────────────────────────────────────────────────────────────────────────
# Band edge result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BandEdgeResult:
    material: str
    chi_eV: float
    Eg_eV: float
    pH: float
    # vs vacuum (Butler & Ginley scale, Ec = 4.5 eV)
    Ecb_vs_vacuum: float
    Evb_vs_vacuum: float
    # vs NHE (IUPAC: E_NHE = E_vacuum - 4.44 eV)
    Ecb_vs_NHE: float
    Evb_vs_NHE: float
    # vs RHE (pH-corrected)
    Ecb_vs_RHE: float
    Evb_vs_RHE: float
    semiconductor_type: str = 'n'
    notes: str = ''

    def summary(self) -> str:
        lines = [
            f"Material       : {self.material}",
            f"χ (Mulliken)   : {self.chi_eV:.3f} eV",
            f"Eg             : {self.Eg_eV:.3f} eV",
            f"pH             : {self.pH:.1f}",
            f"Ecb vs vacuum  : {self.Ecb_vs_vacuum:+.3f} eV",
            f"Evb vs vacuum  : {self.Evb_vs_vacuum:+.3f} eV",
            f"Ecb vs NHE     : {self.Ecb_vs_NHE:+.3f} V",
            f"Evb vs NHE     : {self.Evb_vs_NHE:+.3f} V",
            f"Ecb vs RHE     : {self.Ecb_vs_RHE:+.3f} V  (pH {self.pH:.1f})",
            f"Evb vs RHE     : {self.Evb_vs_RHE:+.3f} V  (pH {self.pH:.1f})",
            f"Type           : {self.semiconductor_type}-type",
        ]
        if self.notes:
            lines.append(f"Notes          : {self.notes}")
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BandEdgeCalculator
# ─────────────────────────────────────────────────────────────────────────────

class BandEdgeCalculator:
    """
    Calculate semiconductor band edge positions.

    Parameters
    ----------
    material : str, optional
        Name from the built-in database (case-insensitive).  If supplied,
        chi and Eg are read from the DB; manual values override DB values.
    chi_eV : float, optional
        Mulliken electronegativity of the compound (geometric mean of
        atomic electronegativities), in eV.  Required if material not in DB.
    Eg_eV : float, optional
        Optical bandgap from Tauc plot (eV).  Overrides DB value.
    pH : float
        Solution pH for RHE conversion (default 7.0).
    Ec : float
        Energy reference point in eV (default 4.5 — Butler & Ginley 1978).
    """

    MATERIALS = list(_MATERIALS_DB.keys())

    def __init__(
        self,
        material: str = '',
        chi_eV: Optional[float] = None,
        Eg_eV: Optional[float] = None,
        pH: float = 7.0,
        Ec: float = _EC_SCALE,
    ):
        self.material = material
        self.pH = pH
        self.Ec = Ec
        self._stype = 'n'
        self._notes = ''
        self._epsilon_r: Optional[float] = None

        # Load from DB
        db_entry = _MATERIALS_DB.get(material, _MATERIALS_DB.get(material.lower(), {}))
        self.chi_eV: float = chi_eV if chi_eV is not None else db_entry.get('chi', None)
        self.Eg_eV: float  = Eg_eV  if Eg_eV  is not None else db_entry.get('Eg',  None)
        self._stype = db_entry.get('type', 'n')
        self._notes = db_entry.get('notes', '')
        self._epsilon_r = db_entry.get('epsilon_r', None)

        if self.chi_eV is None:
            raise ValueError(
                f"Material '{material}' not in DB.  Please supply chi_eV manually.\n"
                f"Available: {', '.join(self.MATERIALS)}"
            )
        if self.Eg_eV is None:
            raise ValueError(
                f"Eg_eV is required for '{material}' (not in DB).  "
                f"Obtain from Tauc plot."
            )

    # ── calculation ───────────────────────────────────────────────────────

    def calculate(self) -> BandEdgeResult:
        """
        Compute Ecb and Evb on all scales.

        Returns a BandEdgeResult dataclass.
        """
        # Butler & Ginley formula
        Ecb_vac = self.chi_eV - self.Ec - 0.5 * self.Eg_eV
        Evb_vac = Ecb_vac + self.Eg_eV

        # vs NHE  (Trasatti 1986: E_NHE = 4.44 eV vs vacuum)
        Ecb_NHE = Ecb_vac - (_EC_ABS - self.Ec)   # adjust for scale difference
        Evb_NHE = Evb_vac - (_EC_ABS - self.Ec)
        # Simpler: since we used Ec=4.5, Ecb vs NHE = chi - 4.5 - 0.5Eg - (4.44-4.5)
        # = chi - 4.44 - 0.5Eg
        Ecb_NHE = self.chi_eV - _EC_ABS - 0.5 * self.Eg_eV
        Evb_NHE = Ecb_NHE + self.Eg_eV

        # vs RHE  (Nernst: E_RHE = E_NHE - 0.0592 × pH at 25°C)
        nernst = 0.05916 * self.pH  # at 298.15 K
        Ecb_RHE = Ecb_NHE - nernst
        Evb_RHE = Evb_NHE - nernst

        return BandEdgeResult(
            material=self.material,
            chi_eV=self.chi_eV,
            Eg_eV=self.Eg_eV,
            pH=self.pH,
            Ecb_vs_vacuum=Ecb_vac,
            Evb_vs_vacuum=Evb_vac,
            Ecb_vs_NHE=Ecb_NHE,
            Evb_vs_NHE=Evb_NHE,
            Ecb_vs_RHE=Ecb_RHE,
            Evb_vs_RHE=Evb_RHE,
            semiconductor_type=self._stype,
            notes=self._notes,
        )

    # ── convenience: multi-material comparison ────────────────────────────

    @classmethod
    def compare(
        cls,
        materials: list,
        pH: float = 7.0,
    ) -> list:
        """
        Calculate band edges for a list of material names.
        Returns list[BandEdgeResult].  Materials not in DB are skipped
        with a warning.
        """
        import warnings
        results = []
        for m in materials:
            try:
                calc = cls(material=m, pH=pH)
                results.append(calc.calculate())
            except ValueError as e:
                warnings.warn(str(e))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Mott-Schottky analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MottSchottkyResult:
    """Result of Mott-Schottky (1/C² vs V) analysis."""
    Vfb_vs_ref: float       # V vs reference electrode used during measurement
    Vfb_vs_RHE: float       # V vs RHE
    Vfb_vs_NHE: float       # V vs NHE
    Nd_cm3: float           # carrier density (cm⁻³)
    slope: float            # d(1/C²)/dV
    intercept: float        # 1/C² at V=0
    R2: float               # coefficient of determination
    semiconductor_type: str  # 'n' (slope>0) or 'p' (slope<0)
    Vfb_range: Tuple[float, float]  # 95% CI on Vfb
    diagnostics: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Vfb (vs ref)   : {self.Vfb_vs_ref:+.4f} V",
            f"Vfb (vs RHE)   : {self.Vfb_vs_RHE:+.4f} V",
            f"Vfb (vs NHE)   : {self.Vfb_vs_NHE:+.4f} V",
            f"Nd             : {self.Nd_cm3:.3e} cm⁻³",
            f"Type           : {self.semiconductor_type}-type (slope {'+' if self.slope > 0 else ''}{self.slope:.3e})",
            f"R²             : {self.R2:.5f}",
            f"Vfb 95% CI     : [{self.Vfb_range[0]:+.4f}, {self.Vfb_range[1]:+.4f}] V",
        ]
        for k, v in self.diagnostics.items():
            lines.append(f"{k:15s}: {v}")
        return '\n'.join(lines)


class MottSchottky:
    """
    Mott-Schottky analysis from capacitance-voltage data.

    Parameters
    ----------
    epsilon_r : float
        Relative permittivity of the semiconductor (dielectric constant).
        Use built-in DB value: ``_MATERIALS_DB['TiO2']['epsilon_r']``.
    electrode_area_cm2 : float
        Geometric electrode area in cm² (default 0.0707 — 3 mm GCE).
    pH : float
        Solution pH (for RHE conversion).
    e_ref_vs_NHE : float
        Reference electrode potential vs NHE (V).  Use E_REF_MAP in app.py.
    temperature_K : float
        Temperature in Kelvin (default 298.15).
    V_range : tuple (V_low, V_high) or None
        Restrict linear fit to this potential window.  None = auto.
    min_R2 : float
        Minimum acceptable R² for the linear fit (warning if below).
    """

    def __init__(
        self,
        epsilon_r: float = 10.0,
        electrode_area_cm2: float = 0.0707,
        pH: float = 7.0,
        e_ref_vs_NHE: float = 0.0,
        temperature_K: float = 298.15,
        V_range: Optional[Tuple[float, float]] = None,
        min_R2: float = 0.98,
    ):
        self.epsilon_r = epsilon_r
        self.area_m2 = electrode_area_cm2 * 1e-4   # cm² → m²
        self.pH = pH
        self.e_ref_vs_NHE = e_ref_vs_NHE
        self.T = temperature_K
        self.V_range = V_range
        self.min_R2 = min_R2

    # ── main analysis ─────────────────────────────────────────────────────

    def analyze(
        self,
        V: np.ndarray,
        C_F: np.ndarray,
    ) -> MottSchottkyResult:
        """
        Perform Mott-Schottky analysis.

        Parameters
        ----------
        V : ndarray
            Applied potential (V vs reference electrode).
        C_F : ndarray
            Capacitance (Farads).  If you measure from EIS: C = -1/(ω·Z_imag)
            at a fixed intermediate frequency (e.g., 1 kHz).

        Returns
        -------
        MottSchottkyResult
        """
        V = np.asarray(V, dtype=float)
        C_F = np.asarray(C_F, dtype=float)

        # Remove non-positive capacitances
        valid = C_F > 0
        V, C_F = V[valid], C_F[valid]
        if len(V) < 4:
            raise ValueError("Fewer than 4 valid (C > 0) data points after filtering.")

        C_inv2 = 1.0 / (C_F ** 2)

        # Restrict to V_range if specified
        if self.V_range is not None:
            mask = (V >= self.V_range[0]) & (V <= self.V_range[1])
            V_fit, y_fit = V[mask], C_inv2[mask]
            if len(V_fit) < 4:
                raise ValueError(
                    f"Fewer than 4 points in V_range {self.V_range}.  "
                    f"Broaden the range or set V_range=None for auto."
                )
        else:
            V_fit, y_fit = self._auto_linear_region(V, C_inv2)

        # Linear regression
        slope, intercept, R2, slope_err = self._linreg(V_fit, y_fit)

        # Flat-band potential: Vfb = -intercept / slope  (at 1/C² = 0)
        # Corrected for thermal voltage: Vfb → Vfb − kT/q (small, ~26 mV)
        kT_q = _KB * self.T / _E_CHARGE
        Vfb_ref = -intercept / slope - kT_q

        # 95% CI on Vfb using propagation of uncertainty
        # δVfb ≈ sqrt((δm/m)² + (δb/b)²) × |Vfb|
        Vfb_uncertainty = abs(Vfb_ref) * np.sqrt(
            (slope_err / abs(slope)) ** 2 if slope != 0 else 0
        )
        Vfb_range = (Vfb_ref - 1.96 * Vfb_uncertainty,
                     Vfb_ref + 1.96 * Vfb_uncertainty)

        # Carrier density Nd from slope
        # slope = 2 / (e × ε₀ × εr × A² × Nd)
        # → Nd = 2 / (e × ε₀ × εr × A² × slope)
        Nd_m3 = 2.0 / (
            _E_CHARGE * _EPS_0 * self.epsilon_r
            * (self.area_m2 ** 2) * abs(slope)
        )
        Nd_cm3 = Nd_m3 * 1e-6   # m⁻³ → cm⁻³

        # Semiconductor type
        stype = 'n' if slope > 0 else 'p'

        # Convert Vfb to other scales
        Vfb_NHE = Vfb_ref + self.e_ref_vs_NHE
        Vfb_RHE = Vfb_NHE - 0.05916 * self.pH

        # Diagnostics
        diag: Dict[str, str] = {}
        if R2 < self.min_R2:
            diag['R² warning'] = (
                f"R²={R2:.4f} < {self.min_R2} — non-ideal MS behaviour. "
                f"Check frequency dispersion or surface states."
            )
        if Nd_cm3 < 1e14:
            diag['Nd warning'] = (
                f"Nd={Nd_cm3:.2e} cm⁻³ is very low — check area or ε_r value."
            )
        if Nd_cm3 > 1e22:
            diag['Nd warning'] = (
                f"Nd={Nd_cm3:.2e} cm⁻³ is very high — possibly metallic or "
                f"degenerate semiconductor."
            )
        if abs(V_fit[-1] - V_fit[0]) < 0.1:
            diag['Range warning'] = (
                "Linear fit region < 100 mV — increase potential window for reliability."
            )

        return MottSchottkyResult(
            Vfb_vs_ref=Vfb_ref,
            Vfb_vs_RHE=Vfb_RHE,
            Vfb_vs_NHE=Vfb_NHE,
            Nd_cm3=Nd_cm3,
            slope=slope,
            intercept=intercept,
            R2=R2,
            semiconductor_type=stype,
            Vfb_range=Vfb_range,
            diagnostics=diag,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _linreg(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Ordinary least-squares linear regression.
        Returns (slope, intercept, R², slope_stderr).
        """
        n = len(x)
        Sx  = np.sum(x)
        Sy  = np.sum(y)
        Sxx = np.sum(x * x)
        Sxy = np.sum(x * y)
        denom = n * Sxx - Sx ** 2
        if abs(denom) < 1e-300:
            return 0.0, np.mean(y), 0.0, np.inf
        slope = (n * Sxy - Sx * Sy) / denom
        intercept = (Sy - slope * Sx) / n
        y_pred = slope * x + intercept
        SS_res = np.sum((y - y_pred) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0.0
        # Standard error of slope
        s2 = SS_res / max(n - 2, 1)
        slope_err = np.sqrt(s2 / max(Sxx - Sx**2 / n, 1e-300))
        return float(slope), float(intercept), float(R2), float(slope_err)

    def _auto_linear_region(
        self,
        V: np.ndarray,
        y: np.ndarray,
        min_pts: int = 6,
        step: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the best contiguous linear sub-region of the 1/C² vs V curve.
        Uses a sliding window: maximise R² × window_length².
        """
        n = len(V)
        best_score = -1.0
        best_mask = slice(0, n)

        for start in range(0, n - min_pts, step):
            for end in range(start + min_pts, n + 1, step):
                sl, ic, R2, _ = self._linreg(V[start:end], y[start:end])
                score = R2 * ((end - start) ** 2)   # reward long + linear
                if score > best_score:
                    best_score = score
                    best_mask = slice(start, end)

        return V[best_mask], y[best_mask]


# ─────────────────────────────────────────────────────────────────────────────
# Band edge energy diagram  (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

class BandEdgePlotter:
    """
    Generate a Plotly energy-level diagram for one or more semiconductors.

    Usage
    -----
    >>> plotter = BandEdgePlotter(scale='NHE', pH=7.0)
    >>> results = BandEdgeCalculator.compare(['g-C3N4', 'TiO2', 'WO3'], pH=7.0)
    >>> fig = plotter.draw(results)
    >>> import streamlit as st; st.plotly_chart(fig)
    """

    _COLORS = [
        '#6d28d9', '#2563eb', '#059669', '#d97706',
        '#dc2626', '#7c3aed', '#0891b2', '#65a30d',
    ]

    def __init__(self, scale: str = 'NHE', pH: float = 7.0):
        """
        Parameters
        ----------
        scale : 'NHE', 'RHE', or 'vacuum'
        pH : float  — used for RHE labels.
        """
        assert scale in ('NHE', 'RHE', 'vacuum'), \
            "scale must be 'NHE', 'RHE', or 'vacuum'"
        self.scale = scale
        self.pH = pH

    def draw(
        self,
        results: list,
        Vfb_results: Optional[list] = None,
        title: str = 'Semiconductor Band Edge Positions',
        redox_lines: Optional[Dict[str, float]] = None,
    ):
        """
        Draw the energy band diagram.

        Parameters
        ----------
        results : list[BandEdgeResult]
        Vfb_results : list[MottSchottkyResult] or None
            If provided, flat-band potentials are shown as dashed lines.
        title : str
        redox_lines : dict {label: E_NHE}, optional
            Add horizontal reference lines (e.g. O2/H2O, H+/H2).
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for BandEdgePlotter.")

        fig = go.Figure()
        bar_width = 0.5
        scale_attr = {
            'NHE':    ('Ecb_vs_NHE', 'Evb_vs_NHE',    'E vs NHE (V)'),
            'RHE':    ('Ecb_vs_RHE', 'Evb_vs_RHE',    f'E vs RHE (V, pH {self.pH:.1f})'),
            'vacuum': ('Ecb_vs_vacuum', 'Evb_vs_vacuum', 'E vs vacuum (eV)'),
        }[self.scale]
        ecb_attr, evb_attr, y_label = scale_attr

        # Default redox reference lines (NHE; adjust if RHE)
        if redox_lines is None:
            ph_corr = -0.05916 * self.pH if self.scale == 'RHE' else 0.0
            redox_lines = {
                'H⁺/H₂  (0 V NHE)':    0.00 + ph_corr,
                'O₂/H₂O (+1.23 V NHE)': 1.23 + ph_corr,
            }

        for i, res in enumerate(results):
            x0 = i - bar_width / 2
            x1 = i + bar_width / 2
            Ecb = getattr(res, ecb_attr)
            Evb = getattr(res, evb_attr)
            color = self._COLORS[i % len(self._COLORS)]

            # Valence band (filled bar)
            fig.add_shape(
                type='rect', x0=x0, x1=x1, y0=Evb, y1=Evb + 0.3,
                fillcolor=color, opacity=0.85,
                line=dict(color=color, width=1),
            )
            # Bandgap region
            fig.add_shape(
                type='rect', x0=x0, x1=x1, y0=Ecb, y1=Evb,
                fillcolor=color, opacity=0.15,
                line=dict(color=color, width=0.5, dash='dot'),
            )
            # Conduction band (filled bar)
            fig.add_shape(
                type='rect', x0=x0, x1=x1, y0=Ecb - 0.3, y1=Ecb,
                fillcolor=color, opacity=0.85,
                line=dict(color=color, width=1),
            )
            # Labels
            fig.add_annotation(
                x=i, y=Ecb - 0.15,
                text=f"Ecb = {Ecb:+.2f} V",
                showarrow=False, font=dict(size=10, color='white'),
                yanchor='middle',
            )
            fig.add_annotation(
                x=i, y=Evb + 0.15,
                text=f"Evb = {Evb:+.2f} V",
                showarrow=False, font=dict(size=10, color='white'),
                yanchor='middle',
            )
            fig.add_annotation(
                x=i, y=(Ecb + Evb) / 2,
                text=f"Eg = {res.Eg_eV:.2f} eV",
                showarrow=False, font=dict(size=9, color=color),
                yanchor='middle',
            )

            # Flat-band line
            if Vfb_results and i < len(Vfb_results):
                vfb_val = getattr(Vfb_results[i],
                                   'Vfb_vs_RHE' if self.scale == 'RHE'
                                   else 'Vfb_vs_NHE')
                fig.add_shape(
                    type='line', x0=x0, x1=x1, y0=vfb_val, y1=vfb_val,
                    line=dict(color=color, dash='dash', width=2),
                )
                fig.add_annotation(
                    x=i, y=vfb_val,
                    text=f" Vfb={vfb_val:+.2f}",
                    showarrow=False, xanchor='left',
                    font=dict(size=9, color=color),
                )

        # Redox reference lines
        for label, E in redox_lines.items():
            fig.add_hline(
                y=E, line_dash='dot', line_color='#6b7280',
                annotation_text=label,
                annotation_position='right',
                annotation_font=dict(size=9, color='#6b7280'),
            )

        fig.update_layout(
            title=title,
            xaxis=dict(
                tickvals=list(range(len(results))),
                ticktext=[r.material for r in results],
                showgrid=False,
            ),
            yaxis=dict(
                title=y_label,
                zeroline=True,
                zerolinecolor='#e5e7eb',
                zerolinewidth=1,
            ),
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f8f9fa',
            font=dict(family='Inter', color='#1e293b'),
            margin=dict(l=60, r=120, t=60, b=60),
            height=550,
            showlegend=False,
        )
        return fig
