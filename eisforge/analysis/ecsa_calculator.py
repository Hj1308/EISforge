import numpy as np
from scipy import stats, integrate as sci_integrate
from typing import List, Tuple, Dict, Union, Optional
import warnings


class ECSACalculator:
    """
    Utility class for calculating Electrochemically Active Surface Area (ECSA).

    Supported methods:
        A -- Hydrogen Underpotential Deposition (H-UPD)  -> Pt, Pd (with caveats)
        B -- CO Stripping                                -> Pt, PtRu, PtSn, Pd (with caveats)
        C -- Double Layer Capacitance (Cdl)              -> Carbon, glassy carbon only

    Units convention (strictly enforced):
        potential  : V  (vs RHE)
        current    : A  (Amperes -- divide by 1000 if your data is in mA)
        scan_rate  : V/s
        q_ref      : uC/cm2
        loading_mg : mg (total catalyst mass on electrode)
        area_cm2   : cm2 (geometric electrode area)

    Material-specific warnings
    --------------------------
    Pd (H-UPD):  Palladium absorbs hydrogen sub-surface (bulk absorption), so the
                 integrated H-UPD charge overestimates ECSA by up to 30%.  Use
                 CO-stripping for Pd, or substitute Q_H_PD with ~405 uC/cm2 (IUPAC
                 recommended effective value that partially accounts for bulk H).

    PtRu alloys (CO-stripping): Ru oxidises CO at lower potentials than Pt and
                 contributes additional charge. Using Q_CO_PT=420 uC/cm2 will
                 under-estimate ECSA. Adjust q_ref according to the Pt atomic
                 fraction (e.g. ~300 uC/cm2 for PtRu 1:1).

    RuO2 / metal oxides (Cdl): These are pseudo-capacitive materials; a large
                 fraction of the "non-Faradaic" current is actually Faradaic
                 (surface redox). Cdl-derived ECSA can be 10x too high.  Use this
                 method only for relative comparisons, never as absolute ECSA.
    """

    # Reference charges (uC/cm2)
    Q_H_PT   = 210.0   # Pt  -- H-UPD
    Q_H_PD   = 212.0   # Pd  -- H-UPD (see class docstring: bulk-H caveat)
    Q_CO_PT  = 420.0   # Pt  -- CO stripping
    Q_CO_PD  = 424.0   # Pd  -- CO stripping

    # Specific double-layer capacitance (mF/cm2)
    CS_CARBON = 0.035   # porous carbon / CNT / graphene
    CS_RUO2   = 0.060   # RuO2 / metal oxides  (pseudo-capacitive: see docstring)
    CS_GC     = 0.020   # flat glassy carbon

    # Catalyst type identifiers used by auto_ecsa
    _NOBLE   = ("noble_metal", "pt", "platinum")
    _CARBON  = ("carbon_material", "carbon", "mf", "metal_free")

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate(potential: np.ndarray, current: np.ndarray) -> None:
        """Basic sanity checks on input arrays."""
        if len(potential) != len(current):
            raise ValueError(
                f"potential ({len(potential)}) and current ({len(current)}) "
                "must have the same length."
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
        Split a full CV into forward and backward scans with full robustness.

        Handles both common starting directions:
            Low  -> High -> Low  (typical H-UPD; idx_min < idx_max)
            High -> Low  -> High (typical CO-stripping or oxide reduction;
                                  idx_min > idx_max)

        The vertex is defined as the *first* turning point encountered after the
        start, which is determined by comparing idx_min and idx_max positions.

        Edge-boundary detection: if either vertex falls on the first or last
        sample (incomplete CV), a UserWarning is raised and the array is split
        at the midpoint as a safe fallback.

        Returns:
            (fwd_pot, fwd_cur): first sweep (start -> first vertex)
            (bwd_pot, bwd_cur): return sweep (first vertex -> end)
        """
        if len(potential) < 3:
            raise ValueError(
                "Potential array must have at least 3 points to detect vertices."
            )

        idx_min = int(np.argmin(potential))
        idx_max = int(np.argmax(potential))

        # Boundary guard: vertex at edge => incomplete or single-direction data
        boundary = {0, len(potential) - 1}
        if idx_min in boundary or idx_max in boundary:
            warnings.warn(
                "Vertex detected at array boundary -- CV may be incomplete or "
                "reversed. Using midpoint fallback split.",
                UserWarning,
                stacklevel=3,
            )
            mid = len(potential) // 2
            return (
                (potential[:mid], current[:mid]),
                (potential[mid:], current[mid:]),
            )

        # Case 1: started at low potential -> went high -> returned low
        if idx_min < idx_max:
            vertex = idx_max
        # Case 2: started at high potential -> went low -> returned high
        else:
            vertex = idx_min

        fwd = (potential[:vertex + 1], current[:vertex + 1])
        bwd = (potential[vertex:],     current[vertex:])

        if len(bwd[0]) < 5:
            warnings.warn(
                "Backward scan has fewer than 5 points. Check CV data quality.",
                UserWarning,
                stacklevel=3,
            )

        return fwd, bwd

    @staticmethod
    def _subtract_baseline(
        potential  : np.ndarray,
        current    : np.ndarray,
        degree     : int = 1,
        mask_peak  : bool = True,
    ) -> np.ndarray:
        """
        Subtract a polynomial baseline from current data.

        More accurate than a simple linear baseline for broad asymmetric peaks
        (e.g. CO stripping) where the double-layer capacitance background is
        curved.

        Args:
            potential : potential array (V)
            current   : current array (A)
            degree    : polynomial degree
                            1  -- linear  (adequate for narrow H-UPD peaks)
                            2  -- quadratic (recommended for CO stripping)
                            3  -- cubic  (only if the window is very wide)
            mask_peak : if True, exclude the central 70% of the window so that
                        the peak itself does not distort the baseline fit.
                        Uses the outer 15% on each side for fitting.

        Returns:
            Baseline-corrected current array (same shape as input).
        """
        if len(potential) < 5:
            raise ValueError("At least 5 points required for baseline subtraction.")

        # Linear: fast path (identical to old _linear_baseline)
        if degree == 1:
            baseline = np.interp(
                potential,
                [potential[0], potential[-1]],
                [current[0],   current[-1]],
            )
            return current - baseline

        # Polynomial degree >= 2
        if mask_peak:
            n   = len(potential)
            n15 = max(int(n * 0.15), degree + 1)  # at least degree+1 points per side
            mask = np.zeros(n, dtype=bool)
            mask[:n15]  = True
            mask[-n15:] = True
            fit_pot = potential[mask]
            fit_cur = current[mask]
        else:
            fit_pot = potential
            fit_cur = current

        if len(fit_pot) < degree + 1:
            warnings.warn(
                f"Not enough edge points ({len(fit_pot)}) for degree-{degree} "
                "polynomial. Falling back to linear baseline.",
                UserWarning,
                stacklevel=3,
            )
            return ECSACalculator._subtract_baseline(
                potential, current, degree=1, mask_peak=False
            )

        coeffs   = np.polyfit(fit_pot, fit_cur, deg=degree)
        baseline = np.polyval(coeffs, potential)
        return current - baseline

    @staticmethod
    def _baseline_from_edges(
        potential  : np.ndarray,
        current    : np.ndarray,
        left_edge  : float,
        right_edge : float,
        degree     : int = 1,
    ) -> np.ndarray:
        """
        Fit a polynomial baseline using only the user-specified flanking regions.

        Useful when the auto mask_peak heuristic in _subtract_baseline is
        insufficient (e.g. very broad CO peaks that span >70% of the window).

        Args:
            left_edge  : upper bound of the left flanking region (V)
            right_edge : lower bound of the right flanking region (V)
            degree     : polynomial degree (1 or 2 recommended)

        Returns:
            Baseline-corrected current array.
        """
        mask = (potential <= left_edge) | (potential >= right_edge)
        if np.sum(mask) < degree + 1:
            raise ValueError(
                f"Only {np.sum(mask)} points outside edges; need at least "
                f"{degree+1} for a degree-{degree} fit. Widen edge regions."
            )
        coeffs   = np.polyfit(potential[mask], current[mask], deg=degree)
        baseline = np.polyval(coeffs, potential)
        return current - baseline

    @staticmethod
    def _windowed_delta_j(
        fwd_pot : np.ndarray,
        fwd_cur : np.ndarray,
        bwd_pot : np.ndarray,
        bwd_cur : np.ndarray,
        v_mid   : float,
        half_window : float = 0.020,
    ) -> float:
        """
        Average |j_anodic - j_cathodic| over [v_mid - half_window, v_mid + half_window].

        More robust than a single-point interpolation at v_mid because it
        averages out potentiostat lag and shot noise, especially at fast scan
        rates (>=100 mV/s) where the CV loop is significantly widened.

        Args:
            half_window : half-width of the averaging window in V (default 20 mV)

        Returns:
            Scalar mean |delta_j| in A (before area normalisation).
        """
        v_lo = v_mid - half_window
        v_hi = v_mid + half_window

        # Sample the window at 40 evenly spaced points for robust averaging
        v_sample = np.linspace(v_lo, v_hi, 40)

        j_a = np.interp(v_sample, fwd_pot,        fwd_cur)
        j_c = np.interp(v_sample, bwd_pot[::-1],  bwd_cur[::-1])
        return float(np.mean(np.abs(j_a - j_c)))

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------

    @classmethod
    def method_a_hupd(
        cls,
        potential       : np.ndarray,
        current         : np.ndarray,
        scan_rate       : float,
        loading_mg      : float,
        v_range         : Tuple[float, float] = (0.05, 0.40),
        q_ref           : Optional[float] = None,
        area_cm2        : float = 1.0,
        baseline_degree : int = 1,
        catalyst        : str = "Pt",
    ) -> Dict[str, Union[float, str]]:
        """
        Method A: H-UPD charge integration (cathodic scan only).

        Args:
            potential       : V vs RHE
            current         : A (Amperes)
            scan_rate       : V/s
            loading_mg      : total catalyst loading (mg)
            v_range         : (V_low, V_high) integration window vs RHE
            q_ref           : reference charge density (uC/cm2).
                              Default: Q_H_PT (210 uC/cm2 for Pt).
            area_cm2        : geometric electrode area (cm2)
            baseline_degree : 1 = linear (default, adequate for H-UPD);
                              2 = quadratic for wider, asymmetric peaks.
            catalyst        : material label used for material-specific
                              warnings (e.g. "Pd").

        Returns dict keys:
            method, charge_uC, q_ref_used, ecsa_cm2, specific_ecsa_cm2_mg,
            integration_window.

        WARNING -- Palladium:
            Pd absorbs H into its bulk lattice (sub-surface absorption), so
            the measured H-UPD charge is a mixture of surface adsorption and
            bulk uptake.  This overestimates ECSA by ~20-30%.  For Pd, use
            method_b_co() instead, or supply q_ref=405 (IUPAC effective value).
        """
        if q_ref is None:
            q_ref = cls.Q_H_PT

        # Pd bulk-absorption warning
        if catalyst.lower() in ("pd", "palladium"):
            warnings.warn(
                "H-UPD on Pd: bulk hydrogen absorption adds 20-30% to the "
                "measured charge, overestimating ECSA. Consider CO-stripping "
                "(method_b_co) or set q_ref=405 uC/cm2 (IUPAC effective value).",
                UserWarning,
                stacklevel=2,
            )

        cls._validate(potential, current)

        # Use only the cathodic (backward) scan for H-UPD
        _, (bwd_pot, bwd_cur) = cls._split_scans(potential, current)

        mask  = (bwd_pot >= v_range[0]) & (bwd_pot <= v_range[1])
        pot_w = bwd_pot[mask]
        cur_w = bwd_cur[mask]

        if len(pot_w) < 5:
            raise ValueError(
                f"Only {len(pot_w)} points found in H-UPD window {v_range}. "
                "Check v_range or data coverage."
            )

        # Sort by potential (cathodic scan runs high->low; trapz needs monotonic x)
        order  = np.argsort(pot_w)
        pot_s  = pot_w[order]
        cur_s  = cur_w[order]
        cur_bc = cls._subtract_baseline(pot_s, cur_s, degree=baseline_degree)

        # Q (uC) = |integral I dV| / scan_rate * 1e6
        charge_uC     = abs(sci_integrate.trapezoid(cur_bc, pot_s)) / abs(scan_rate) * 1e6
        ecsa_cm2      = charge_uC / q_ref
        specific_ecsa = ecsa_cm2 / loading_mg if loading_mg > 0 else 0.0

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
        potential       : np.ndarray,
        current         : np.ndarray,
        scan_rate       : float,
        loading_mg      : float,
        v_range         : Tuple[float, float],
        q_ref           : Optional[float] = None,
        area_cm2        : float = 1.0,
        baseline_degree : int = 2,
        catalyst        : str = "Pt",
    ) -> Dict[str, Union[float, str]]:
        """
        Method B: CO stripping charge integration (anodic scan only).

        CO oxidation peak appears on the forward (anodic) scan.

        Args:
            potential       : V vs RHE
            current         : A (Amperes)
            scan_rate       : V/s
            loading_mg      : catalyst loading (mg)
            v_range         : (V_low, V_high) integration window containing
                              the entire CO oxidation peak.
            q_ref           : reference charge density (uC/cm2).
                              Default: Q_CO_PT (420 uC/cm2 for pure Pt).

                              IMPORTANT for alloys (PtRu, PtSn, etc.):
                              Ru and Sn co-oxidise CO at lower potentials than
                              Pt, so the total charge reflects contributions
                              from both metals.  There is no universal constant;
                              supply a composition-corrected q_ref or treat the
                              result as a relative comparison only.
                              Approximate guides:
                                  PtRu 1:1  ->  ~300 uC/cm2
                                  PtSn 3:1  ->  ~350 uC/cm2
            area_cm2        : geometric electrode area (cm2)
            baseline_degree : 1 = linear; 2 = quadratic (default, recommended
                              for CO stripping where DL background is curved).
            catalyst        : material label for alloy warning.

        Returns dict keys:
            method, charge_uC, q_ref_used, ecsa_cm2, specific_ecsa_cm2_mg,
            integration_window.
        """
        if q_ref is None:
            q_ref = cls.Q_CO_PT

        # Alloy warning: Ru/Sn also oxidise CO
        _catalyst_low = catalyst.lower()
        if any(x in _catalyst_low for x in ("ru", "sn", "alloy")):
            warnings.warn(
                f"CO-stripping on alloy catalyst '{catalyst}': secondary metals "
                "(Ru, Sn) also oxidise CO, adding extra charge. "
                "Using Q_CO_PT=420 will underestimate ECSA for the Pt component. "
                "Supply a composition-corrected q_ref (e.g. ~300 uC/cm2 for "
                "PtRu 1:1, ~350 uC/cm2 for PtSn 3:1).",
                UserWarning,
                stacklevel=2,
            )

        cls._validate(potential, current)

        # Use only the anodic (forward) scan for CO stripping
        (fwd_pot, fwd_cur), _ = cls._split_scans(potential, current)

        mask  = (fwd_pot >= v_range[0]) & (fwd_pot <= v_range[1])
        pot_w = fwd_pot[mask]
        cur_w = fwd_cur[mask]

        if len(pot_w) < 5:
            raise ValueError(
                f"Only {len(pot_w)} points found in CO stripping window {v_range}."
            )

        order  = np.argsort(pot_w)
        pot_s  = pot_w[order]
        cur_s  = cur_w[order]
        cur_bc = cls._subtract_baseline(pot_s, cur_s, degree=baseline_degree)

        charge_uC     = abs(sci_integrate.trapezoid(cur_bc, pot_s)) / abs(scan_rate) * 1e6
        ecsa_cm2      = charge_uC / q_ref
        specific_ecsa = ecsa_cm2 / loading_mg if loading_mg > 0 else 0.0

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
        scan_rates      : List[float],
        v_range         : Tuple[float, float],
        cs_mF_cm2       : Optional[float] = None,
        loading_mg      : float = 1.0,
        area_cm2        : float = 1.0,
        half_window     : float = 0.020,
    ) -> Dict[str, Union[float, list, str]]:
        """
        Method C: Cdl from scan-rate dependence.

        delta_j (= |j_anodic - j_cathodic|) at v_mid is plotted vs scan rate.
        Slope = 2 * Cdl  ->  Cdl = slope / 2.

        Args:
            potentials_list : one array per scan rate (V)
            currents_list   : one array per scan rate (A)
            scan_rates      : V/s -- must match order of potentials/currents
            v_range         : non-Faradaic window; v_mid = mean(v_range)
            cs_mF_cm2       : specific capacitance (mF/cm2).
                              Defaults to CS_CARBON (0.035 mF/cm2).

                              IMPORTANT: This method is strictly valid only for
                              pure double-layer materials (glassy carbon, porous
                              carbon, CNTs, graphene).  For metal oxides (RuO2,
                              MnO2, Co3O4), a large fraction of the measured
                              current in the "non-Faradaic" region is actually
                              Faradaic (surface redox / pseudo-capacitance).
                              Using this method for metal oxides can overestimate
                              ECSA by an order of magnitude (10x).  Use it for
                              relative comparisons only.

            loading_mg      : catalyst loading (mg)
            area_cm2        : geometric electrode area (cm2)
            half_window     : half-width (V) of the averaging window around
                              v_mid used for delta_j calculation (default 20 mV).
                              Wider windows reduce noise and potentiostat lag
                              at fast scan rates (>= 100 mV/s).

        Returns dict keys:
            method, cdl_mF_cm2, ecsa_cm2, specific_ecsa_cm2_mg, r_squared,
            fit_slope, fit_intercept, fit_std_err, cs_used_mF_cm2, v_mid,
            scan_rates_V_s, delta_j_A_cm2.
        """
        if cs_mF_cm2 is None:
            cs_mF_cm2 = cls.CS_CARBON

        n = len(scan_rates)
        if not (len(potentials_list) == len(currents_list) == n):
            raise ValueError(
                f"potentials_list ({len(potentials_list)}), "
                f"currents_list ({len(currents_list)}), "
                f"and scan_rates ({n}) must all have the same length."
            )
        if n < 3:
            warnings.warn(
                "At least 3 scan rates recommended for reliable Cdl fit.",
                UserWarning,
            )

        v_mid        = (v_range[0] + v_range[1]) / 2.0
        delta_j_list = []

        for i, (pot, cur) in enumerate(zip(potentials_list, currents_list)):
            cls._validate(pot, cur)

            (fwd_pot, fwd_cur), (bwd_pot, bwd_cur) = cls._split_scans(pot, cur)

            if len(fwd_pot) < 2 or len(bwd_pot) < 2:
                raise ValueError(f"CV #{i+1}: scan split failed -- too few points.")

            # Windowed average instead of single-point interpolation:
            # averages over [v_mid - half_window, v_mid + half_window] to
            # reduce potentiostat lag and shot noise at fast scan rates.
            dj = cls._windowed_delta_j(
                fwd_pot, fwd_cur, bwd_pot, bwd_cur,
                v_mid=v_mid, half_window=half_window,
            )
            delta_j_list.append(dj / area_cm2)  # normalise to A/cm2

        slope, intercept, r_value, _, std_err = stats.linregress(scan_rates, delta_j_list)

        cdl_F_cm2  = abs(slope) / 2.0
        cdl_mF_cm2 = cdl_F_cm2 * 1000.0
        ecsa_cm2   = (cdl_mF_cm2 / cs_mF_cm2) * area_cm2
        specific   = ecsa_cm2 / loading_mg if loading_mg > 0 else 0.0

        return {
            "method"               : "Cdl",
            "cdl_mF_cm2"           : cdl_mF_cm2,
            "ecsa_cm2"             : ecsa_cm2,
            "specific_ecsa_cm2_mg" : specific,
            "r_squared"            : r_value ** 2,
            "fit_slope"            : slope,
            "fit_intercept"        : intercept,
            "fit_std_err"          : std_err,
            "cs_used_mF_cm2"       : cs_mF_cm2,
            "v_mid"                : v_mid,
            "scan_rates_V_s"       : list(scan_rates),
            "delta_j_A_cm2"        : delta_j_list,
        }

    # --------------------------------------------------------------------------
    # Convenience dispatcher
    # --------------------------------------------------------------------------

    @classmethod
    def auto_ecsa(
        cls,
        catalyst_type   : str,
        potential       : np.ndarray,
        current         : np.ndarray,
        scan_rate       : float,
        loading_mg      : float,
        area_cm2        : float = 1.0,
        # H-UPD / CO-stripping kwargs
        v_range         : Optional[Tuple[float, float]] = None,
        q_ref           : Optional[float] = None,
        baseline_degree : Optional[int]   = None,
        # Cdl-only kwargs (pass potentials_list, currents_list, scan_rates explicitly)
        potentials_list : Optional[List[np.ndarray]] = None,
        currents_list   : Optional[List[np.ndarray]] = None,
        scan_rates_list : Optional[List[float]]      = None,
        cs_mF_cm2       : Optional[float]            = None,
    ) -> Dict[str, Union[float, str]]:
        """
        Automatically dispatch to the appropriate ECSA method based on catalyst type.

        Routing logic
        -------------
        noble_metal / pt / platinum  -> method_a_hupd (H-UPD, v_range default 0.05-0.40 V)
        carbon_material / carbon / mf / metal_free -> method_c_cdl (must supply Cdl lists)
        pd / palladium               -> method_a_hupd with Pd warning
        alloy (contains 'ru'/'sn')   -> method_b_co with alloy warning (v_range required)

        For metal oxides (RuO2, MnO2, etc.), no automatic dispatch is provided
        because no method gives a reliable absolute ECSA for these materials.
        Call method_c_cdl directly and treat the result as relative.

        Args:
            catalyst_type   : string identifier (case-insensitive)
            potential       : V vs RHE  (ignored for Cdl -- use potentials_list)
            current         : A          (ignored for Cdl -- use currents_list)
            scan_rate       : V/s        (ignored for Cdl -- use scan_rates_list)
            loading_mg      : mg
            area_cm2        : cm2
            v_range         : integration window (required for CO stripping;
                              optional for H-UPD, defaults applied if None)
            q_ref           : override reference charge (uC/cm2)
            baseline_degree : override baseline polynomial degree
            potentials_list, currents_list, scan_rates_list, cs_mF_cm2:
                              passed through to method_c_cdl

        Returns:
            Result dict from the dispatched method.

        Raises:
            ValueError  : if catalyst_type is metal oxide or unrecognised.
        """
        ct = catalyst_type.lower().strip()

        # Metal oxides: no safe automatic dispatch
        if any(x in ct for x in ("ruo2", "mno2", "co3o4", "oxide")):
            raise ValueError(
                f"catalyst_type='{catalyst_type}' is a metal oxide / pseudo-"
                "capacitive material. No automatic ECSA dispatch is available "
                "because Cdl is not a valid absolute ECSA metric for these "
                "materials. Call method_c_cdl() directly and treat the result "
                "as a relative comparison only."
            )

        # Carbon / metal-free: Cdl
        if ct in cls._CARBON:
            if potentials_list is None or currents_list is None or scan_rates_list is None:
                raise ValueError(
                    "For carbon / metal-free catalysts, supply potentials_list, "
                    "currents_list, and scan_rates_list for Cdl calculation."
                )
            return cls.method_c_cdl(
                potentials_list=potentials_list,
                currents_list=currents_list,
                scan_rates=scan_rates_list,
                v_range=v_range or (0.10, 0.30),
                cs_mF_cm2=cs_mF_cm2,
                loading_mg=loading_mg,
                area_cm2=area_cm2,
            )

        # Alloys with CO-oxidising secondary metals -> CO stripping
        if any(x in ct for x in ("ru", "sn", "alloy")):
            if v_range is None:
                raise ValueError(
                    "v_range is required for CO stripping on alloy catalysts."
                )
            return cls.method_b_co(
                potential=potential,
                current=current,
                scan_rate=scan_rate,
                loading_mg=loading_mg,
                v_range=v_range,
                q_ref=q_ref,
                area_cm2=area_cm2,
                baseline_degree=baseline_degree if baseline_degree is not None else 2,
                catalyst=catalyst_type,
            )

        # Noble metals and Pd -> H-UPD
        if ct in cls._NOBLE or "pt" in ct or "pd" in ct or "palladium" in ct:
            return cls.method_a_hupd(
                potential=potential,
                current=current,
                scan_rate=scan_rate,
                loading_mg=loading_mg,
                v_range=v_range or (0.05, 0.40),
                q_ref=q_ref,
                area_cm2=area_cm2,
                baseline_degree=baseline_degree if baseline_degree is not None else 1,
                catalyst=catalyst_type,
            )

        raise ValueError(
            f"Unrecognised catalyst_type='{catalyst_type}'. "
            "Use one of: noble_metal, pt, pd, alloy, ptru, carbon_material, "
            "carbon, mf, metal_free -- or call the specific method directly."
        )
