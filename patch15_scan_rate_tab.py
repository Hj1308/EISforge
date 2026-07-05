# patch15_scan_rate_tab.py
# Adds a 7th tab "Scan-Rate Kinetics" to app.py.
#   - Loads a multi-column Excel (paired E,I columns; row0 = rate labels)
#   - Overlay of all CVs
#   - log(Ipa)-log(nu) b-value with mechanism interpretation
#   - Randles-Sevcik (Ipa vs sqrt(nu)); apparent D behind an opt-in checkbox
#   - Excel export of the per-rate table + regression summary
import shutil, sys

PATH = r"app.py"
s = open(PATH, encoding="utf-8").read()

# ── 1. add the 7th tab to the tabs() call ────────────────────────────────────
OLD_TABS = '''tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001F4C8 CV Analysis", "\U0001F4C9 LSV Analysis", "\U0001F52C EIS Analysis",
    "\U0001F916 EIS-GPT", "\U0001F517 Correlation", "\u2697\ufe0f K-L Analysis"
])'''
NEW_TABS = '''tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "\U0001F4C8 CV Analysis", "\U0001F4C9 LSV Analysis", "\U0001F52C EIS Analysis",
    "\U0001F916 EIS-GPT", "\U0001F517 Correlation", "\u2697\ufe0f K-L Analysis",
    "\U0001F4CA Scan-Rate Kinetics"
])'''

# ── 2. append the tab body at the end of the file ────────────────────────────
TAB_BODY = '''

# ══════════════════ SCAN-RATE KINETICS ════════════════════════════════════════
with tab7:
    st.markdown('<h3>Scan-Rate Kinetics</h3>', unsafe_allow_html=True)
    st.caption(
        "Upload one Excel file with paired columns per scan rate. "
        "Row 1 = scan-rate labels (e.g. '50mV/s'); row 2 = 'E /V', 'I /mA'; "
        "data from row 3. Each scan rate uses two adjacent columns (E, then I)."
    )

    sr_file = st.file_uploader(
        "Upload scan-rate Excel (.xlsx)", type=["xlsx", "xls"], key="sr_up"
    )

    colA, colB = st.columns(2)
    win_on = colA.checkbox(
        "Restrict anodic-peak search to a potential window", value=True,
        help="Recommended: set this to the AOR peak region so the edge of the "
             "scan (OER onset) is not mistaken for the peak.",
    )
    wlo = colB.number_input("Window E_low (V)", value=0.30, step=0.05,
                            disabled=not win_on)
    whi = colB.number_input("Window E_high (V)", value=0.70, step=0.05,
                            disabled=not win_on)

    show_D = st.checkbox(
        "Compute apparent diffusion coefficient D (advanced, use with caution)",
        value=False,
        help="Randles-Sevcik D assumes pure diffusion + reversibility + a "
             "well-defined planar area. For mesoporous/high-area carbon with "
             "mixed control (b<1) this is apparent only.",
    )
    if show_D:
        dc1, dc2, dc3 = st.columns(3)
        d_n = dc1.number_input("n (electrons)", value=4, min_value=1, step=1)
        d_C = dc2.number_input("C (mol/cm³)", value=1.0e-3, format="%.2e")
        d_A = dc3.number_input("Area (cm²)", value=0.07068583, format="%.5f")

    if sr_file is not None:
        try:
            import re as _re
            import numpy as _np
            import pandas as _pd
            from eisforge.analysis.scan_rate_analyzer import analyze_scan_rates

            raw = _pd.read_excel(sr_file, header=None)
            labels = raw.iloc[0].tolist()
            data = {}
            for c in range(0, raw.shape[1], 2):
                lab = str(labels[c])
                m = _re.search(r"([\\d.]+)\\s*mV", lab)
                if not m:
                    continue
                rate = float(m.group(1))
                E = _pd.to_numeric(raw.iloc[2:, c], errors="coerce").to_numpy()
                I = _pd.to_numeric(raw.iloc[2:, c + 1], errors="coerce").to_numpy()
                keep = ~(_np.isnan(E) | _np.isnan(I))
                if keep.sum() >= 5:
                    data[rate] = (E[keep], I[keep])

            if len(data) < 3:
                st.error("Found fewer than 3 valid scan-rate column pairs. "
                         "Check the file layout.")
            else:
                st.success(f"\u2705 Loaded {len(data)} scan rates: "
                           + ", ".join(f"{r:.0f}" for r in sorted(data)) + " mV/s")

                window = (wlo, whi) if win_on else None
                kw = dict(peak_window=window)
                if show_D:
                    kw.update(compute_D=True, n_electrons=int(d_n),
                              area_cm2=float(d_A), conc_mol_cm3=float(d_C))
                res = analyze_scan_rates(data, **kw)

                import plotly.graph_objects as go

                # (a) overlay of all CVs
                fig_ov = go.Figure()
                for r in sorted(data):
                    E, I = data[r]
                    fig_ov.add_trace(go.Scatter(x=E, y=I, mode="lines",
                                                name=f"{r:.0f} mV/s"))
                fig_ov.update_layout(**PLOTLY_LAYOUT, title="CV overlay",
                                     xaxis_title="E (V)", yaxis_title="I")
                st.plotly_chart(fig_ov, use_container_width=True)

                # (b) log-log
                nu = res.rates_mV / 1000.0
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=_np.log10(nu),
                                           y=_np.log10(res.ipa),
                                           mode="markers", name="data",
                                           marker=dict(size=9, color="#2563eb")))
                xfit = _np.array([_np.log10(nu).min(), _np.log10(nu).max()])
                fig_b.add_trace(go.Scatter(
                    x=xfit, y=res.b_value * xfit + res.b_intercept,
                    mode="lines", name=f"slope b={res.b_value:.3f}",
                    line=dict(color="#dc2626", width=2)))
                fig_b.update_layout(**PLOTLY_LAYOUT,
                                    title="log(Ipa) vs log(\u03bd)",
                                    xaxis_title="log \u03bd (V/s)",
                                    yaxis_title="log Ipa")
                st.plotly_chart(fig_b, use_container_width=True)

                # (c) Randles-Sevcik
                fig_rs = go.Figure()
                fig_rs.add_trace(go.Scatter(x=_np.sqrt(nu), y=res.ipa,
                                            mode="markers", name="data",
                                            marker=dict(size=9, color="#2563eb")))
                xr = _np.array([_np.sqrt(nu).min(), _np.sqrt(nu).max()])
                fig_rs.add_trace(go.Scatter(
                    x=xr, y=res.rs_slope * xr + res.rs_intercept,
                    mode="lines", name=f"R\u00b2={res.rs_r2:.4f}",
                    line=dict(color="#dc2626", width=2)))
                fig_rs.update_layout(**PLOTLY_LAYOUT,
                                     title="Randles\u2013\u0160ev\u010d\u00edk: Ipa vs \u221a\u03bd",
                                     xaxis_title="\u221a\u03bd (V/s)^\u00bd",
                                     yaxis_title="Ipa")
                st.plotly_chart(fig_rs, use_container_width=True)

                # findings + warnings
                st.markdown("#### Interpretation")
                for f_ in res.findings:
                    st.markdown(f"- {f_}")
                for w in res.warnings:
                    st.markdown(f"- \u26a0 {w}")
                if res.diffusion_coeff is not None:
                    st.metric("Apparent D (cm\u00b2/s)",
                              f"{res.diffusion_coeff:.3e}")

                # per-rate table
                table = _pd.DataFrame({
                    "Scan rate (mV/s)": res.rates_mV,
                    "sqrt(nu) (V/s)^0.5": _np.sqrt(nu),
                    "Ipa": res.ipa,
                    "E at peak (V)": res.ipa_potential,
                })
                st.dataframe(table, use_container_width=True, hide_index=True)

                # Excel export
                try:
                    import io as _io
                    buf = _io.BytesIO()
                    with _pd.ExcelWriter(buf, engine="openpyxl") as xw:
                        table.to_excel(xw, sheet_name="Peaks", index=False)
                        _pd.DataFrame({
                            "Item": ["b-value (log-log slope)", "b R2",
                                     "Randles-Sevcik slope", "Randles-Sevcik R2",
                                     "Mechanism", "Apparent D (cm2/s)"],
                            "Value": [res.b_value, res.b_r2, res.rs_slope,
                                      res.rs_r2, res.mechanism_label(),
                                      res.diffusion_coeff
                                      if res.diffusion_coeff is not None else "not computed"],
                        }).to_excel(xw, sheet_name="Summary", index=False)
                    st.download_button(
                        "\U0001F4E5 Download scan-rate results (Excel)",
                        data=buf.getvalue(),
                        file_name="eisforge_scan_rate_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.warning(f"Excel export unavailable: {e}")

        except Exception as e:
            st.error(f"Scan-rate analysis error: {e}")
'''

n = 0
if NEW_TABS in s:
    print("[1/2] tabs already updated, skipping")
else:
    if OLD_TABS not in s:
        print("ERROR step 1: OLD tabs block not found. Aborting.")
        sys.exit(1)
    s = s.replace(OLD_TABS, NEW_TABS, 1)
    n += 1
    print("[1/2] OK — added tab7 to st.tabs()")

if "with tab7:" in s:
    print("[2/2] tab body already present, skipping")
else:
    s = s.rstrip() + "\n" + TAB_BODY
    n += 1
    print("[2/2] OK — appended tab7 body")

if n:
    shutil.copy(PATH, PATH + ".bak_patch15")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: app.py.bak_patch15)")
else:
    print("Nothing to do.")
