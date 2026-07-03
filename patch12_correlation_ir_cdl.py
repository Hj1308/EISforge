# patch12_correlation_ir_cdl.py
# Three fixes in app.py:
#
#   1) Correlation tab (tab5) crashed whenever BOTH a CV analysis and an EIS
#      fit were present: it accessed fit_r.param_names / fit_r.params, which
#      do not exist on FitResult (only .parameters does). Fixed, and R_ct is
#      now taken as the LARGEST faradaic resistance (dominant charge-transfer
#      arc) instead of blindly "R1".
#
#   2) iR compensation now auto-connects to the EIS fit: if an EIS CNLS fit
#      exists in the session, its R0 (high-frequency intercept) is offered as
#      the default R_s value with a caption. Manual override still possible.
#
#   3) C_dl from EIS: the Physical Interpretation block now also reports each
#      process's effective capacitance normalised by the electrode area
#      (uF/cm^2), so the double-layer arc can be compared directly with the
#      C_dl literature range used elsewhere in the app. A caution note states
#      this equals C_dl only for spectra measured at non-faradaic potential.
import shutil, sys

PATH = r"app.py"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# ── 1. Correlation tab: use .parameters dict ─────────────────────────────────
(
'''        corr_data = {
            "R_s (\u03a9)": [next((v for n, v in zip(fit_r.param_names, fit_r.params) if n == "R0"), "N/A")],
            "R_ct (\u03a9)": [next(
                (v for n, v in zip(fit_r.param_names, fit_r.params) if n in ("R1", "Rct")), "N/A")],
            "E_onset (V)": [cv_r.e_onset],
            "j_f (mA/cm\u00b2)": [cv_r.j_forward_peak],
        }''',
'''        import re as _re
        _params = fit_r.parameters
        _r_items = sorted(
            ((int(m.group(1)), v) for n, v in _params.items()
             if (m := _re.fullmatch(r"R(\\d+)", n))),
        )
        _r_s_val = _r_items[0][1] if _r_items else "N/A"
        _faradaic = [v for _, v in _r_items[1:]]
        # dominant charge-transfer resistance = largest |R| among faradaic arcs
        _r_ct_val = max(_faradaic, key=abs) if _faradaic else "N/A"
        corr_data = {
            "R_s (\u03a9)": [_r_s_val],
            "R_ct (\u03a9)": [_r_ct_val],
            "E_onset (V)": [cv_r.e_onset],
            "j_f (mA/cm\u00b2)": [cv_r.j_forward_peak],
        }''',
),
# ── 2. iR: auto-connect to EIS fit R0 ────────────────────────────────────────
(
'''    use_ir = st.checkbox(
        "Apply iR compensation", value=False,
        help="E_corrected = E_measured \u2212 I(A) \u00d7 R_s(\u03a9)",
    )
    r_s = st.number_input(
        "R_s (\u03a9) \u2014 from EIS fit", value=0.0, step=0.1, min_value=0.0,
        disabled=not use_ir,
        help="Use R0 from EIS CNLS fit (high-frequency intercept)",
    )''',
'''    use_ir = st.checkbox(
        "Apply iR compensation", value=False,
        help="E_corrected = E_measured \u2212 I(A) \u00d7 R_s(\u03a9)",
    )
    # Auto-connect: pull R0 (high-frequency intercept) from the EIS fit if one
    # exists in this session; manual override always possible.
    _rs_from_eis = 0.0
    if "eis_fit" in st.session_state:
        _rs_from_eis = float(st.session_state["eis_fit"].parameters.get("R0", 0.0))
    r_s = st.number_input(
        "R_s (\u03a9) \u2014 from EIS fit", value=float(_rs_from_eis), step=0.1, min_value=0.0,
        disabled=not use_ir,
        help="Auto-filled from R0 of the EIS CNLS fit when available; edit to override.",
    )
    if _rs_from_eis > 0:
        st.caption(f"\u21b3 auto-filled from EIS fit: R\u2080 = {_rs_from_eis:.3f} \u03a9")''',
),
# ── 3. C_dl per area in the interpretation block ─────────────────────────────
(
'''            st.markdown("#### Physical Interpretation")
            try:
                from eisforge.analysis.eis_interpreter import interpret_fit
                interp = interpret_fit(
                    fit_r.parameters, fit_r.circuit_string, fit_r.chi_squared
                )
                st.markdown(interp.as_markdown())
            except Exception as e:
                st.warning(f"Interpretation unavailable: {e}")''',
'''            st.markdown("#### Physical Interpretation")
            try:
                from eisforge.analysis.eis_interpreter import interpret_fit
                interp = interpret_fit(
                    fit_r.parameters, fit_r.circuit_string, fit_r.chi_squared
                )
                st.markdown(interp.as_markdown())
                # ── C_dl estimate from EIS (per geometric area) ─────────────
                if area and area > 0:
                    _cdl_lines = [
                        f"- {p.label}: C_eff/A \u2248 "
                        f"{p.c_eff / area * 1e6:.1f} \u03bcF/cm\u00b2"
                        for p in interp.processes if p.c_eff is not None
                    ]
                    if _cdl_lines:
                        st.markdown("**Specific capacitance (per geometric area):**")
                        st.markdown("\\n".join(_cdl_lines))
                        st.caption(
                            "The C_eff of an arc equals C_dl only if the spectrum "
                            "was measured at a NON-faradaic potential; at reaction "
                            "potentials it contains pseudocapacitive/adsorption "
                            "contributions."
                        )
            except Exception as e:
                st.warning(f"Interpretation unavailable: {e}")''',
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
    shutil.copy(PATH, PATH + ".bak_patch12")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: app.py.bak_patch12)")
else:
    print("Nothing to do.")
