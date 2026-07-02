# patch10_11_eis_excel_and_interpretation.py
# Two features in one app.py patch (same EIS region, applied atomically):
#
#   Patch 10 — Excel export of EIS results:
#     After the fit-parameter table, a "Download Excel" button exports a
#     multi-sheet .xlsx: Summary (circuit, chi2, R_s...), Fit_Parameters,
#     Data (f, Z', -Z''), Fit_Curve (400-pt smooth model curve).
#
#   Patch 11 — Interpretation:
#     (a) EIS tab: rule-based physical interpretation (eis_interpreter)
#         rendered right below the fit parameters.
#     (b) EIS-GPT tab: the old code imported a non-existent module
#         (eisforge.models.eis_gpt) and accessed FitResult attributes that
#         don't exist (.param_names/.params) — it errored on every click.
#         Replaced with the rule-based interpreter + an honest note that
#         the physics-informed transformer is implemented but untrained
#         (planned v0.4).
import shutil, sys

PATH = r"app.py"
s = open(PATH, encoding="utf-8").read()

PATCHES = [
# ── Patch 10 + 11a: after the fit parameter table ────────────────────────────
(
'''        if "eis_fit" in st.session_state:
            fit_r = st.session_state["eis_fit"]
            st.markdown("#### Fit Parameters")
            param_df = pd.DataFrame({
                "Parameter": list(fit_r.parameters.keys()),
                "Value": list(fit_r.parameters.values()),
                "Std Error": [fit_r.parameter_errors.get(k, float("nan"))
                             for k in fit_r.parameters.keys()],
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)''',
'''        if "eis_fit" in st.session_state:
            fit_r = st.session_state["eis_fit"]
            st.markdown("#### Fit Parameters")
            param_df = pd.DataFrame({
                "Parameter": list(fit_r.parameters.keys()),
                "Value": list(fit_r.parameters.values()),
                "Std Error": [fit_r.parameter_errors.get(k, float("nan"))
                             for k in fit_r.parameters.keys()],
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            # ── Physical interpretation (rule-based, deterministic) ─────────
            st.markdown("#### Physical Interpretation")
            try:
                from eisforge.analysis.eis_interpreter import interpret_fit
                interp = interpret_fit(
                    fit_r.parameters, fit_r.circuit_string, fit_r.chi_squared
                )
                st.markdown(interp.as_markdown())
            except Exception as e:
                st.warning(f"Interpretation unavailable: {e}")

            # ── Excel export ────────────────────────────────────────────────
            try:
                import io as _io
                _buf = _io.BytesIO()
                with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
                    pd.DataFrame({
                        "Item": ["Circuit", "Reduced chi2 (modulus-weighted)",
                                 "Converged", "Points used", "Outliers removed",
                                 "Source file"],
                        "Value": [fit_r.circuit_string, fit_r.chi_squared,
                                  fit_r.converged, fit_r.n_points_used,
                                  fit_r.n_outliers_removed,
                                  st.session_state.get("eis_filename", "")],
                    }).to_excel(_xw, sheet_name="Summary", index=False)
                    param_df.to_excel(_xw, sheet_name="Fit_Parameters", index=False)
                    pd.DataFrame({
                        "Frequency_Hz": fr,
                        "Z_real_Ohm": zr,
                        "minus_Z_imag_Ohm": zi,
                    }).to_excel(_xw, sheet_name="Data", index=False)
                    if getattr(fit_r, "z_fit_smooth", None) is not None:
                        pd.DataFrame({
                            "Frequency_Hz": fit_r.freq_smooth,
                            "Z_real_Ohm": fit_r.z_fit_smooth.real,
                            "minus_Z_imag_Ohm": -fit_r.z_fit_smooth.imag,
                        }).to_excel(_xw, sheet_name="Fit_Curve", index=False)
                st.download_button(
                    "\\U0001F4E5 Download EIS results (Excel)",
                    data=_buf.getvalue(),
                    file_name="eisforge_eis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")''',
),
# ── remember uploaded filename for the Excel Summary sheet ───────────────────
(
'''                st.session_state.update({"eis_fr": fr, "eis_zr": zr, "eis_zi": zi})''',
'''                st.session_state.update({"eis_fr": fr, "eis_zr": zr, "eis_zi": zi,
                                         "eis_filename": eis_file.name})''',
),
# ── Patch 11b: fix the broken EIS-GPT tab ─────────────────────────────────────
(
'''    st.info("AI-powered EIS spectrum interpretation using physics-informed language model.")
    if st.button("\\U0001F916 Interpret EIS Spectrum"):
        if "eis_fit" in st.session_state:
            try:
                from eisforge.models.eis_gpt import EISInterpreter
                interp = EISInterpreter()
                result = interp.interpret(
                    circuit=circ,
                    params=dict(zip(st.session_state["eis_fit"].param_names,
                                    st.session_state["eis_fit"].params)),
                    system_type=system_type, catalyst=catalyst,
                    electrolyte=ekey, potential=eis_pot,
                )
                st.markdown(result)
            except Exception as e:
                st.error(f"EIS-GPT error: {e}")
        else:
            st.warning("Run CNLS fit first (EIS Analysis tab).")''',
'''    st.info(
        "Rule-based physical interpretation of the CNLS fit (deterministic, "
        "reviewable). The physics-informed transformer (EIS-GPT) is implemented "
        "but not yet trained \\u2014 ML-based interpretation is planned for v0.4."
    )
    if st.button("\\U0001F50D Interpret EIS Spectrum"):
        if "eis_fit" in st.session_state:
            try:
                from eisforge.analysis.eis_interpreter import interpret_fit
                _fit = st.session_state["eis_fit"]
                _interp = interpret_fit(
                    _fit.parameters, _fit.circuit_string, _fit.chi_squared
                )
                st.markdown(f"**Circuit:** `{_fit.circuit_string}`")
                st.markdown(_interp.as_markdown())
            except Exception as e:
                st.error(f"Interpretation error: {e}")
        else:
            st.warning("Run CNLS fit first (EIS Analysis tab).")''',
),
]

n_applied = 0
for i, (old, new) in enumerate(PATCHES, 1):
    old = old.replace("\\U0001F916", "\U0001F916").replace("\\U0001F4E5", "\U0001F4E5").replace("\\U0001F50D", "\U0001F50D").replace("\\u2014", "\u2014")
    new = new.replace("\\U0001F916", "\U0001F916").replace("\\U0001F4E5", "\U0001F4E5").replace("\\U0001F50D", "\U0001F50D").replace("\\u2014", "\u2014")
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
    shutil.copy(PATH, PATH + ".bak_patch10_11")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: app.py.bak_patch10_11)")
else:
    print("Nothing to do.")
