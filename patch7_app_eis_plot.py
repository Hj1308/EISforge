# patch7_app_eis_plot.py
# EIS tab plot fixes in app.py:
#   1) Nyquist orientation: parser convention stores z_imag = -Im(Z)
#      (positive for capacitive). The plot did y=-zi, flipping the curve
#      downward. Now plots y=zi so -Z'' points UP, like ZView.
#   2) mode="markers" instead of "markers+lines": no more jagged segments
#      between data points; the smooth red CNLS fit line (z_fit_smooth,
#      400 log-spaced frequencies) is overlaid instead — ZView style.
#   3) Robust Huber IRLS enabled in the CNLS fit (robust=True).
#   4) BioLogic loader sign fixed to match the project-wide convention.
import shutil, sys

PATH = r"app.py"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# -- 1+2. Nyquist: orientation, markers-only, smooth fit overlay --------------
(
'''        fig_eis = go.Figure()
        fig_eis.add_trace(go.Scatter(x=zr, y=-zi, mode="markers+lines",
                                     name="Nyquist", marker=dict(color="#2563eb", size=6)))''',
'''        fig_eis = go.Figure()
        # Convention: z_imag stores -Im(Z) (positive, capacitive) -> plot y=zi
        fig_eis.add_trace(go.Scatter(x=zr, y=zi, mode="markers",
                                     name="Data", marker=dict(color="#2563eb", size=6)))
        _fit_prev = st.session_state.get("eis_fit")
        if _fit_prev is not None and getattr(_fit_prev, "z_fit_smooth", None) is not None:
            _zs = _fit_prev.z_fit_smooth  # complex Z on 400 log-spaced freqs
            fig_eis.add_trace(go.Scatter(
                x=_zs.real, y=-_zs.imag, mode="lines",
                name="CNLS fit",
                line=dict(color="#dc2626", width=2.5),
            ))'''
),
# -- 3. robust Huber fit -------------------------------------------------------
(
'''                fitter = CNLSFitter(circuit_string=circ, initial_guess=p0_list,
                                    bounds=bounds, allow_negative_r=ndr_hint)''',
'''                fitter = CNLSFitter(circuit_string=circ, initial_guess=p0_list,
                                    bounds=bounds, allow_negative_r=ndr_hint,
                                    robust=True)'''
),
# -- 4. BioLogic sign convention -------------------------------------------------
(
'''                df["freq/Hz"].to_numpy(),
                df["Re(Z)/Ohm"].to_numpy(),
                -df["-Im(Z)/Ohm"].to_numpy(),''',
'''                df["freq/Hz"].to_numpy(),
                df["Re(Z)/Ohm"].to_numpy(),
                df["-Im(Z)/Ohm"].to_numpy(),'''
),
]

n_applied = 0
for i, (old, new) in enumerate(PATCHES, 1):
    if new in s:
        print(f"[{i}/{len(PATCHES)}] already applied, skipping")
        continue
    if old not in s:
        print(f"ERROR at step {i}: OLD block not found. Aborting, no changes written.")
        sys.exit(1)
    s = s.replace(old, new, 1)
    n_applied += 1
    print(f"[{i}/{len(PATCHES)}] OK")

if n_applied:
    shutil.copy(PATH, PATH + ".bak_patch7")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: app.py.bak_patch7)")
else:
    print("Nothing to do.")
