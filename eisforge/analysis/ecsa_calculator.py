import numpy as np
from scipy import stats, integrate as sci_integrate
from typing import List, Tuple, Dict, Union, Optional
import warnings


class ECSACalculator:
    """
    Utility class for calculating Electrochemically Active Surface Area (ECSA).

    Supported methods:
        A — Hydrogen Underpotential Deposition (H-UPD)  → Pt, Pd
        B — CO Stripping                                 → PtRu, PtSn, Pd
        C — Double Layer Capacitance (Cdl)               → Carbon, metal-free

    Units convention (strictly enforced):
        potential  : V  (vs RHE)
        current    : A  (Amperes — divide by 1000 if your data is in mA)
        scan_rate  : V/s
        q_ref      : µC/cm²
        loading_mg : mg (total catalyst mass on electrode)
        area_cm2   : cm² (geometric electrode area)
    """

    # Reference charges (µC/cm²)
    Q_H_PT   = 210.0   # Pt  — H-UPD
    Q_H_PD   = 212.0   # Pd  — H-UPD
    Q_CO_PT  = 420.0   # Pt  — CO stripping
    Q_CO_PD  = 424.0   # Pd  — CO stripping

    # Specific double-layer capacitance (mF/cm²)
    CS_CARBON = 0.035   # porous carbon / CNT / graphene
    CS_RUO2   = 0.060   # RuO2 / metal oxides
    CS_GC     = 0.020   # flat glassy carbon

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(potential: np.ndarray, current: np.ndarray) -> None:
        """Basic sanity checks on input arrays."""
        if len(potential) != len(current):
            raise ValueError(
                f"potential ({len(potential)}) and current ({len(current)}) must have the same length."
            )
        if len(potential) < 10:
            raise ValueError("Minimum 10 data points required.")
        if not (np.all(np.isfinite(potential)) and np.all(np.isfinite(current))):
            raise ValueError("NaN or Inf values detected in input arrays.")

    @staticmethod
    def _split_scans(
        potential: np.ndarray, current: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split a full CV into forward (anodic) and backward (cathodic) scans.

        Detection is based on the true potential vertex (argmax), which is robust
        against unequal numbers of points in each half — unlike the naive len//2 split.

        Returns:
            (fwd_pot, fwd_cur), (bwd_pot, bwd_cur)
        """
        vertex_idx = int(np.argmax(potential))
        if vertex_idx == 0 or vertex_idx == len(potential) - 1:
            warnings.warn(
                "Vertex detected at array boundary — CV may be incomplete or reversed. "
                "Attempting argmin-based split.",
                UserWarning, stacklevel=3,
            )
            vertex_idx = int(np.argmin(potential))

        fwd = (potential[:vertex_idx], current[:vertex_idx])
        bwd = (potential[vertex_idx:], current[vertex_idx:])
        return fwd, bwd

    @staticmethod
    def _linear_baseline(
        potential: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Subtract a straight-line baseline connecting the first and last points."""
        baseline = np.interp(
            potential,
            [potential[0], potential[-1]],
            [current[0],   current[-1]],
        )
        return current - baseline

    # ──────────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def method_a_hupd(
        cls,
        potential    : np.ndarray,
        current      : np.ndarray,
        scan_rate    : float,
        loading_mg   : float,
        v_range      : Tuple[float, float] = (0.05, 0.40),
        q_ref        : Optional[float] = None,         # ✅ FIX: subclass-safe default
        area_cm2     : float = 1.0,
    ) -> Dict[str, Union[float, str]]:
        """
        Method A: H-UPD charge integration (cathodic scan only).

        Args:
            potential  : V vs RHE
            current    : A  (Amperes)
            scan_rate  : V/s
            loading_mg : total catalyst loading (mg)
            v_range    : (V_low, V_high) integration window vs RHE
            q_ref      : reference charge density (µC/cm²); defaults to Q_H_PT
            area_cm2   : geometric electrode area (cm²)

        Returns dict keys:
            method, charge_uC, q_ref_used, ecsa_cm2, specific_ecsa_cm2_mg
        """
        if q_ref is None:                   # ✅ FIX: resolved at call time, not class-def time
            q_ref = cls.Q_H_PT

        cls._validate(potential, current)

        # ✅ FIX: use only the cathodic (backward) scan for H-UPD
        _, (bwd_pot, bwd_cur) = cls._split_scans(potential, current)

        # Mask the H-UPD window
        mask = (bwd_pot >= v_range[0]) & (bwd_pot <= v_range[1])
        pot_w = bwd_pot[mask]
        cur_w = bwd_cur[mask]

        if len(pot_w) < 5:
            raise ValueError(
                f"Only {len(pot_w)} points found in H-UPD window {v_range}. "
                "Check v_range or data coverage."
            )

        # Sort by potential (cathodic scan runs high→low; trapz needs monotonic x)
        order   = np.argsort(pot_w)
        pot_s   = pot_w[order]
        cur_s   = cur_w[order]

        # Baseline correction
        cur_bc  = cls._linear_baseline(pot_s, cur_s)

        # Q (µC) = |∫ I dV| / scan_rate  ×  1e6
        charge_uC = abs(sci_integrate.trapezoid(cur_bc, pot_s)) / abs(scan_rate) * 1e6

        ecsa_cm2           = charge_uC / q_ref                    # cm²
        specific_ecsa      = ecsa_cm2  / loading_mg if loading_mg > 0 else 0.0

        return {
            "method"               : "H-UPD",
            "charge_uC"            : charge_uC,
            "q_ref_used"           : q_ref,
            "ecsa_cm2"             : ecsa_cm2,
            "specific_ecsa_cm2_mg" : specific_ecsa,
            "integration_window"   : v_range,
        }

    @classmethod
    def method_b_co(
        cls,
        potential    : np.ndarray,
        current      : np.ndarray,
        scan_rate    : float,
        loading_mg   : float,
        v_range      : Tuple[float, float],
        q_ref        : Optional[float] = None,         # ✅ FIX: subclass-safe
        area_cm2     : float = 1.0,
    ) -> Dict[str, Union[float, str]]:
        """
        Method B: CO stripping charge integration (anodic scan only).

        CO oxidation peak appears on the forward (anodic) scan.
        """
        if q_ref is None:
            q_ref = cls.Q_CO_PT

        cls._validate(potential, current)

        # ✅ FIX: use only the anodic (forward) scan for CO stripping
        (fwd_pot, fwd_cur), _ = cls._split_scans(potential, current)

        mask    = (fwd_pot >= v_range[0]) & (fwd_pot <= v_range[1])
        pot_w   = fwd_pot[mask]
        cur_w   = fwd_cur[mask]

        if len(pot_w) < 5:
            raise ValueError(
                f"Only {len(pot_w)} points found in CO stripping window {v_range}."
            )

        order   = np.argsort(pot_w)
        pot_s   = pot_w[order]
        cur_s   = cur_w[order]
        cur_bc  = cls._linear_baseline(pot_s, cur_s)

        charge_uC          = abs(sci_integrate.trapezoid(cur_bc, pot_s)) / abs(scan_rate) * 1e6
        ecsa_cm2           = charge_uC / q_ref
        specific_ecsa      = ecsa_cm2  / loading_mg if loading_mg > 0 else 0.0

        return {
            "method"               : "CO Stripping",
            "charge_uC"            : charge_uC,
            "q_ref_used"           : q_ref,
            "ecsa_cm2"             : ecsa_cm2,
            "specific_ecsa_cm2_mg" : specific_ecsa,
            "integration_window"   : v_range,
        }

    @classmethod
    def method_c_cdl(
        cls,
        potentials_list : List[np.ndarray],
        currents_list   : List[np.ndarray],
        scan_rates      : List[float],          # V/s (all the same unit as A/B)
        v_range         : Tuple[float, float],
        cs_mF_cm2       : Optional[float] = None,
        loading_mg      : float = 1.0,
        area_cm2        : float = 1.0,
    ) -> Dict[str, Union[float, list, str]]:
        """
        Method C: Cdl from scan-rate dependence.

        Δj (= j_anodic − j_cathodic) at v_mid is plotted vs scan rate.
        Slope = 2 × Cdl  →  Cdl = slope / 2.

        Args:
            potentials_list : one array per scan rate (V)
            currents_list   : one array per scan rate (A)
            scan_rates      : V/s — must match order of potentials/currents lists
            v_range         : non-Faradaic window; v_mid = mean(v_range)
            cs_mF_cm2       : specific capacitance (mF/cm²); defaults to CS_CARBON
            loading_mg      : catalyst loading (mg)
            area_cm2        : geometric electrode area (cm²)
        """
        if cs_mF_cm2 is None:
            cs_mF_cm2 = cls.CS_CARBON

        n = len(scan_rates)
        if not (len(potentials_list) == len(currents_list) == n):
            raise ValueError(
                f"potentials_list ({len(potentials_list)}), currents_list ({len(currents_list)}), "
                f"and scan_rates ({n}) must all have the same length."
            )
        if n < 3:
            warnings.warn("At least 3 scan rates recommended for reliable Cdl.", UserWarning)

        v_mid       = (v_range[0] + v_range[1]) / 2.0
        delta_j_list = []

        for i, (pot, cur) in enumerate(zip(potentials_list, currents_list)):
            cls._validate(pot, cur)

            # ✅ FIX: true vertex → separate forward & backward scans
            (fwd_pot, fwd_cur), (bwd_pot, bwd_cur) = cls._split_scans(pot, cur)

            if len(fwd_pot) < 2 or len(bwd_pot) < 2:
                raise ValueError(f"CV #{i+1}: scan split failed — too few points.")

            # Interpolate at v_mid (more robust than nearest-index lookup)
            j_a = float(np.interp(v_mid, fwd_pot, fwd_cur))   # anodic  current (A)
            j_c = float(np.interp(v_mid, bwd_pot[::-1], bwd_cur[::-1]))  # cathodic

            # Normalise to geometric area → A/cm²
            delta_j_list.append(abs(j_a - j_c) / area_cm2)

        # ✅ FIX: linregress gives slope AND intercept; both returned for UI trendline
        slope, intercept, r_value, _, std_err = stats.linregress(scan_rates, delta_j_list)

        cdl_F_cm2   = abs(slope) / 2.0                   # F/cm²
        cdl_mF_cm2  = cdl_F_cm2 * 1000.0                 # mF/cm²
        ecsa_cm2    = (cdl_mF_cm2 / cs_mF_cm2) * area_cm2
        specific    = ecsa_cm2 / loading_mg if loading_mg > 0 else 0.0

        return {
            "method"               : "Cdl",
            "cdl_mF_cm2"           : cdl_mF_cm2,
            "ecsa_cm2"             : ecsa_cm2,
            "specific_ecsa_cm2_mg" : specific,
            "r_squared"            : r_value ** 2,
            "fit_slope"            : slope,
            "fit_intercept"        : intercept,       # ✅ for UI trendline
            "fit_std_err"          : std_err,
            "cs_used_mF_cm2"       : cs_mF_cm2,
            "v_mid"                : v_mid,
            "scan_rates_V_s"       : list(scan_rates),
            "delta_j_A_cm2"        : delta_j_list,
        }
