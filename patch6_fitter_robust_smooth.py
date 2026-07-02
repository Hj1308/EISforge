# patch6_fitter_robust_smooth.py
# Adds to CNLSFitter:
#   1) robust=True option — IRLS with Huber weights: points with large
#      residuals are automatically down-weighted (ZView-like tolerance
#      to stray points, without hard deletion).
#   2) Smooth fitted curve: FitResult gains freq_smooth / z_fit_smooth —
#      model evaluated on 400 log-spaced frequencies, so the plotted fit
#      is a continuous line like ZView, not segments between data points.
import shutil, sys

PATH = r"eisforge/core/fitter.py"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# -- 1. FitResult: smooth-curve fields ---------------------------------------
(
"""    z_fit: Optional[np.ndarray]
    converged: bool
    n_outliers_removed: int = 0
    n_points_used: int = 0""",
"""    z_fit: Optional[np.ndarray]
    converged: bool
    n_outliers_removed: int = 0
    n_points_used: int = 0
    freq_smooth: Optional[np.ndarray] = None
    z_fit_smooth: Optional[np.ndarray] = None"""
),
# -- 2. __init__: robust flag -------------------------------------------------
(
"""        neighbor_ratio: float = 5.0,
        allow_negative_r: bool = False,
    ) -> None:""",
"""        neighbor_ratio: float = 5.0,
        allow_negative_r: bool = False,
        robust: bool = False,
    ) -> None:"""
),
(
"""        self.allow_negative_r  = bool(allow_negative_r)""",
"""        self.allow_negative_r  = bool(allow_negative_r)
        # Robust IRLS (Huber): after the first fit, points with residuals
        # beyond 1.345·MAD are down-weighted and the fit is repeated. This
        # gives ZView-like insensitivity to stray points without deleting data.
        self.robust            = bool(robust)"""
),
# -- 3. residuals: accept external IRLS weights -------------------------------
(
"""        def residuals(params):
            \"\"\"Returns weighted residuals [Re(\u0394Z)\u00b7w, Im(\u0394Z)\u00b7w].\"\"\"
            try:
                circuit.parameters_ = np.array(params)
                Z_pred = circuit.predict(freq)
                r_real = (Z_meas.real - Z_pred.real) * weights
                r_imag = (Z_meas.imag - Z_pred.imag) * weights
                return np.concatenate([r_real, r_imag])
            except Exception:
                return np.ones(2 * len(freq)) * 1e10""",
"""        irls_w = np.ones(len(freq))  # Huber weights (all 1.0 = plain CNLS)

        def residuals(params):
            \"\"\"Returns weighted residuals [Re(\u0394Z)\u00b7w, Im(\u0394Z)\u00b7w].\"\"\"
            try:
                circuit.parameters_ = np.array(params)
                Z_pred = circuit.predict(freq)
                r_real = (Z_meas.real - Z_pred.real) * weights * irls_w
                r_imag = (Z_meas.imag - Z_pred.imag) * weights * irls_w
                return np.concatenate([r_real, r_imag])
            except Exception:
                return np.ones(2 * len(freq)) * 1e10"""
),
# -- 4. IRLS loop after the optimizer ------------------------------------------
(
"""        # \u2500\u2500 Build parameters dict""",
"""        # \u2500\u2500 Robust IRLS re-fit (Huber weights): points with large residuals
        # are down-weighted and the fit repeated (ZView-like outlier tolerance)
        if self.robust and result is not None and np.all(np.isfinite(fitted)):
            try:
                for _ in range(3):
                    circuit.parameters_ = fitted
                    Z_pred = circuit.predict(freq)
                    r_c    = np.abs(Z_meas - Z_pred) * weights
                    mad    = np.median(np.abs(r_c - np.median(r_c)))
                    scale  = 1.4826 * max(mad, 1e-15)
                    k      = 1.345 * scale
                    new_w  = np.where(r_c <= k, 1.0, np.sqrt(k / r_c))
                    if np.allclose(new_w, irls_w, atol=1e-3):
                        break
                    irls_w[:] = new_w
                    result_r = least_squares(
                        residuals, x0=fitted, bounds=(lb, ub),
                        method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                        max_nfev=10000,
                    )
                    if np.all(np.isfinite(result_r.x)):
                        result, fitted = result_r, result_r.x
                        converged = converged or result_r.success
            except Exception:
                pass  # robust pass is best-effort; plain fit already valid

        # \u2500\u2500 Build parameters dict"""
),
# -- 5. smooth curve computation -----------------------------------------------
(
"""        z_fit = None
        chi2  = float("inf")
        try:
            circuit.parameters_ = fitted
            z_fit = circuit.predict(freq)""",
"""        z_fit = None
        chi2  = float("inf")
        freq_smooth  = None
        z_fit_smooth = None
        try:
            circuit.parameters_ = fitted
            z_fit = circuit.predict(freq)
            # Smooth ZView-style fitted line on dense log-spaced frequencies
            if len(freq) >= 2 and np.all(freq > 0):
                freq_smooth  = np.logspace(
                    np.log10(freq.min()), np.log10(freq.max()), 400
                )
                z_fit_smooth = circuit.predict(freq_smooth)"""
),
# -- 6. return smooth fields ----------------------------------------------------
(
"""            n_outliers_removed=n_removed,
            n_points_used=len(freq),
            _circuit_obj=circuit,
        )""",
"""            n_outliers_removed=n_removed,
            n_points_used=len(freq),
            freq_smooth=freq_smooth,
            z_fit_smooth=z_fit_smooth,
            _circuit_obj=circuit,
        )"""
),
]

for i, (old, new) in enumerate(PATCHES, 1):
    if new in s:
        print(f"[{i}/6] already applied, skipping")
        continue
    if old not in s:
        print(f"ERROR at step {i}: OLD block not found. Aborting, no changes written.")
        sys.exit(1)
    s = s.replace(old, new, 1)
    print(f"[{i}/6] OK")

shutil.copy(PATH, PATH + ".bak")
open(PATH, "w", encoding="utf-8").write(s)
print("Patched OK:", PATH, "(backup:", PATH + ".bak)")
