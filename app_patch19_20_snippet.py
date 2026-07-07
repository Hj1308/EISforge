"""
patch19+20 — Streamlit Integration Snippet
==========================================
Copy the relevant sections into app.py to add:
  1. "⚡ Batch EIS" tab  (patch19 - warm-start fitter)
  2. "🔭 Band Edge"  tab  (patch20 - band edge + Mott-Schottky)

This file is a *reference snippet*, NOT a standalone app.
Author: Hoda Jafari | July 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Add two tabs to the existing st.tabs() call in app.py
# ─────────────────────────────────────────────────────────────────────────────
# Change the existing line from:
#
#   tab1, ..., tab8 = st.tabs([...])
#
# to:
#
#   tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
#       "📈 CV Analysis", "📉 LSV Analysis", "🔬 EIS Analysis",
#       "🤖 EIS-GPT", "🔗 Correlation", "⚗️ K-L Analysis",
#       "📊 Scan-Rate Kinetics", "⏱️ Chronoamperometry",
#       "⚡ Batch EIS",          # <-- patch19
#       "🔭 Band Edge",          # <-- patch20
#   ])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Paste the Batch EIS tab block (patch19)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np

# ══ TAB 9 — BATCH EIS (patch19) ══════════════════════════════════════════════
_BATCH_EIS_TAB_CODE = '''
with tab9:
    st.markdown("<h3>⚡ Batch EIS — Warm-Start Series Fitting</h3>",
                unsafe_allow_html=True)
    st.caption(
        "Upload EIS files from a concentration / potential / time series. "
        "Each fit warm-starts from the previous result — 60–80 % faster convergence."
    )

    # ── file upload ───────────────────────────────────────────────────────
    batch_eis_files = st.file_uploader(
        "Upload EIS files (ordered, e.g. 0.1 M → 2.0 M)",
        type=["idf", "dta", "mpt", "csv", "txt"],
        accept_multiple_files=True,
        key="batch_eis_up",
    )

    batch_x_label = st.text_input(
        "X-axis label for trend plot", value="[Alcohol] / M", key="beis_xlabel"
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_circuit = st.text_input(
            "Equivalent circuit", value="R0-p(R1,CPE1)",
            help="e.g. R0-p(R1,CPE1) or R0-p(R1,C1)-p(R2,C2)",
            key="beis_circuit",
        )
        x_vals_str = st.text_input(
            "X-axis values (comma-separated, same order as files)",
            placeholder="0.1, 0.25, 0.5, 1.0, 2.0",
            key="beis_xvals",
        )
    with col_b2:
        p0_str = st.text_area(
            "Initial parameters (name=value, one per line)",
            value="R0=10\nR1=100\nCPE1_Q=1e-5\nCPE1_n=0.85",
            height=120,
            key="beis_p0",
        )

    if batch_eis_files and len(batch_eis_files) >= 2:
        if st.button(f"▶ Run Batch Fit ({len(batch_eis_files)} spectra)",
                     type="primary", key="run_batch_eis"):
            import importlib, re

            # ── parse p0 ──────────────────────────────────────────────────
            p0_dict = {}
            for line in p0_str.strip().splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    try:
                        p0_dict[k.strip()] = float(v.strip())
                    except ValueError:
                        pass

            # ── parse x values ────────────────────────────────────────────
            x_vals = None
            if x_vals_str.strip():
                try:
                    x_vals = [float(v.strip()) for v in x_vals_str.split(",")]
                except ValueError:
                    st.warning("Could not parse X-axis values — trend plot skipped.")

            # ── load datasets ─────────────────────────────────────────────
            datasets = []
            labels   = []
            with st.spinner("Loading files..."):
                for f in batch_eis_files:
                    try:
                        fr, zr, zi, _ = load_eis(f)
                        Z = zr - 1j * zi
                        datasets.append((fr, Z))
                        labels.append(f.name)
                    except Exception as exc:
                        st.warning(f"{f.name}: {exc}")

            if len(datasets) < 2:
                st.error("Need at least 2 valid EIS files.")
            else:
                # ── build circuit function (uses existing EISforge circuit engine) ──
                try:
                    from eisforge.core.circuit_parser import CircuitParser
                    parser = CircuitParser()
                    circuit_func, param_names_auto = parser.build(batch_circuit)
                    # prefer user-defined p0 names if they match
                    param_names = [p for p in param_names_auto]
                    # fill missing with 1.0 default
                    p0_full = {k: p0_dict.get(k, 1.0) for k in param_names}
                except Exception as exc:
                    st.error(f"Circuit parse error: {exc}")
                    st.stop()

                # ── run warm-start batch fit ──────────────────────────────
                from eisforge.core.batch_fitter import WarmStartBatchFitter

                bounds = [(1e-6, 1e8)] * len(param_names)
                fitter = WarmStartBatchFitter(
                    circuit_func=circuit_func,
                    param_names=param_names,
                    bounds=bounds,
                    weighting="modulus",
                )

                progress_bar = st.progress(0)
                status_text  = st.empty()

                def _cb(i, n, lbl):
                    progress_bar.progress(i / n)
                    status_text.text(f"Fitting {lbl} ({i}/{n})...")

                summary = fitter.fit_series(
                    datasets,
                    initial_params=p0_full,
                    labels=labels,
                    x_values=x_vals,
                    x_label=batch_x_label,
                    progress_callback=_cb,
                )
                progress_bar.empty()
                status_text.empty()
                st.session_state["batch_eis_summary"] = summary

                n_ok = sum(r.success for r in summary.results)
                st.success(f"✅ {n_ok}/{len(datasets)} spectra fitted successfully")

    if "batch_eis_summary" in st.session_state:
        summary = st.session_state["batch_eis_summary"]
        st.divider()

        # ── results table ─────────────────────────────────────────────────
        st.markdown("#### Fit Results")
        st.dataframe(summary.to_dataframe(), use_container_width=True, hide_index=True)

        # ── trend plot ────────────────────────────────────────────────────
        if summary.x_values is not None:
            import plotly.graph_objects as go

            param_keys = list(summary.results[0].params.keys())
            param_to_plot = st.selectbox(
                "Parameter to plot vs condition", param_keys, key="beis_param_plot"
            )

            vals = summary.param_array(param_to_plot)
            errs = summary.error_array(param_to_plot)
            xs   = np.array(summary.x_values[: len(vals)])
            mask = summary.successful_mask()

            fig_trend = go.Figure()
            # error bars
            fig_trend.add_trace(go.Scatter(
                x=xs[mask], y=vals[mask],
                error_y=dict(type="data", array=errs[mask],
                             visible=True, color="rgba(109,40,217,0.5)"),
                mode="markers+lines",
                name=param_to_plot,
                line=dict(color="#6d28d9", width=2),
                marker=dict(size=8, color="#6d28d9"),
            ))
            if (~mask).any():
                fig_trend.add_trace(go.Scatter(
                    x=xs[~mask], y=vals[~mask],
                    mode="markers",
                    name="Failed fits",
                    marker=dict(size=8, color="#dc2626", symbol="x"),
                ))
            fig_trend.update_layout(
                title=f"{param_to_plot} vs {summary.x_label}",
                xaxis_title=summary.x_label,
                yaxis_title=f"{param_to_plot} (Ω or F)",
                template="plotly_white",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── Nyquist overlay ───────────────────────────────────────────────
        if st.checkbox("Show Nyquist overlay (experimental vs fit)", key="beis_nyquist"):
            import plotly.graph_objects as go

            fig_ny = go.Figure()
            colors = [
                "#6d28d9", "#2563eb", "#059669", "#d97706",
                "#dc2626", "#0891b2", "#7c3aed", "#db2777",
            ]
            for idx, (ds, r) in enumerate(zip(
                    st.session_state.get("_beis_datasets", []), summary.results)):
                c = colors[idx % len(colors)]
                fr, Z_exp = ds
                fig_ny.add_trace(go.Scatter(
                    x=Z_exp.real, y=-Z_exp.imag,
                    mode="markers", name=f"{r.label} (exp)",
                    marker=dict(color=c, size=5),
                ))
                if r.z_fit_real is not None:
                    fig_ny.add_trace(go.Scatter(
                        x=r.z_fit_real, y=-r.z_fit_imag,
                        mode="lines", name=f"{r.label} (fit)",
                        line=dict(color=c, width=1.5),
                    ))
            fig_ny.update_layout(
                title="Nyquist Overlay — Experimental vs Fit",
                xaxis_title="Z\' (Ω)", yaxis_title="−Z\'\'(Ω)",
                template="plotly_white",
            )
            st.plotly_chart(fig_ny, use_container_width=True)
'''

# ══ TAB 10 — BAND EDGE (patch20) ═════════════════════════════════════════════
_BAND_EDGE_TAB_CODE = '''
with tab10:
    st.markdown("<h3>🔭 Band Edge & Mott-Schottky Analysis</h3>",
                unsafe_allow_html=True)

    sub_be, sub_ms = st.tabs(["Band Edge Calculator", "Mott-Schottky Analysis"])

    # ── Sub-tab A: Band Edge ──────────────────────────────────────────────
    with sub_be:
        from eisforge.analysis.band_edge_calculator import (
            BandEdgeCalculator, MATERIALS_DB,
        )
        import plotly.graph_objects as go

        st.caption(
            "Computes E_cb and E_vb from Mulliken electronegativity X and "
            "optical band gap Eg (Butler & Ginley, 1978)."
        )

        col_be1, col_be2 = st.columns([1, 1.6])
        with col_be1:
            mat_keys = list(MATERIALS_DB.keys())
            mat_sel = st.multiselect(
                "Select materials (or add custom below)",
                mat_keys, default=["g-C3N4", "TiO2"],
                key="be_mat_sel",
            )
            ph_be = st.number_input(
                "pH (for RHE conversion)", value=float(ph_value),
                min_value=0.0, max_value=14.0, step=0.1, key="be_ph",
            )

            # custom material
            with st.expander("➕ Add custom material"):
                cust_name = st.text_input("Name", value="My Material", key="be_cust_name")
                cust_X    = st.number_input("X (eV)", value=5.0, step=0.01, key="be_cust_X")
                cust_Eg   = st.number_input("Eg (eV)", value=2.5, step=0.01, key="be_cust_Eg")
                add_custom = st.button("Add", key="be_add_cust")

            # Eg overrides
            with st.expander("Override Eg from your Tauc plot"):
                eg_overrides = {}
                for k in mat_sel:
                    db_eg = MATERIALS_DB[k].get("Eg_eV") or 2.5
                    eg_overrides[k] = st.number_input(
                        f"{k} Eg (eV)", value=float(db_eg),
                        step=0.01, key=f"be_eg_{k}",
                    )

        with col_be2:
            calc = BandEdgeCalculator(pH=ph_be)
            results_be = calc.compare(mat_sel, Eg_overrides=eg_overrides)

            # custom material
            if add_custom:
                results_be.append(calc.calculate(cust_X, cust_Eg, material=cust_name))

            if results_be:
                # metrics table
                import pandas as pd
                rows = []
                for r in results_be:
                    rows.append({
                        "Material": r.material,
                        "Eg (eV)": f"{r.Eg_eV:.2f}",
                        "E_cb (NHE)": f"{r.Ecb_NHE:+.3f}",
                        "E_vb (NHE)": f"{r.Evb_NHE:+.3f}",
                        "E_cb (RHE)": f"{r.Ecb_RHE:+.3f}",
                        "E_vb (RHE)": f"{r.Evb_RHE:+.3f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Band diagram
                fig_bd = go.Figure()
                water_Ecb = 0.0     # O2/H2O at NHE = 1.23 V, H+/H2 = 0 V
                water_Evb = 1.23

                fig_bd.add_hline(y=water_Ecb, line_dash="dot", line_color="#2563eb",
                                 annotation_text="H⁺/H₂ (0 V vs NHE)",
                                 annotation_position="right")
                fig_bd.add_hline(y=water_Evb, line_dash="dot", line_color="#dc2626",
                                 annotation_text="O₂/H₂O (+1.23 V vs NHE)",
                                 annotation_position="right")

                colors_bd = ["#6d28d9","#2563eb","#059669","#d97706",
                             "#dc2626","#0891b2","#7c3aed","#db2777"]
                for idx, r in enumerate(results_be):
                    c = colors_bd[idx % len(colors_bd)]
                    x0, x1 = idx + 0.15, idx + 0.85
                    # CB bar
                    fig_bd.add_shape(type="rect",
                        x0=x0, x1=x1, y0=r.Ecb_NHE, y1=r.Ecb_NHE - 0.15,
                        fillcolor=c, opacity=0.85, line_width=0)
                    # VB bar
                    fig_bd.add_shape(type="rect",
                        x0=x0, x1=x1, y0=r.Evb_NHE, y1=r.Evb_NHE + 0.15,
                        fillcolor=c, opacity=0.5, line_width=0)
                    # band gap fill
                    fig_bd.add_shape(type="rect",
                        x0=x0, x1=x1,
                        y0=r.Ecb_NHE, y1=r.Evb_NHE,
                        fillcolor=c, opacity=0.08, line_width=0)
                    # Eg label
                    fig_bd.add_annotation(
                        x=(x0 + x1) / 2,
                        y=(r.Ecb_NHE + r.Evb_NHE) / 2,
                        text=f"{r.material}<br>Eg={r.Eg_eV:.2f} eV",
                        showarrow=False,
                        font=dict(size=11, color=c),
                    )

                fig_bd.update_layout(
                    title="Band Edge Diagram (vs NHE)",
                    yaxis_title="E / V vs NHE",
                    xaxis=dict(visible=False),
                    template="plotly_white",
                    height=500,
                )
                # invert y-axis so more negative (higher in energy) is up
                fig_bd.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_bd, use_container_width=True)

    # ── Sub-tab B: Mott-Schottky ──────────────────────────────────────────
    with sub_ms:
        from eisforge.analysis.band_edge_calculator import MottSchottkyAnalyzer
        import plotly.graph_objects as go

        st.caption(
            "Upload a 2-column CSV/TXT: E (V) | C (F or μF). "
            "The tool finds the most linear region automatically and returns "
            "V_fb, N_d, and semiconductor type."
        )

        col_ms1, col_ms2 = st.columns([1, 1.5])
        with col_ms1:
            ms_file = st.file_uploader(
                "Upload Mott-Schottky data (E, C)",
                type=["csv", "txt"], key="ms_up",
            )
            ms_eps = st.number_input(
                "Relative permittivity ε_r", value=30.0, step=1.0, key="ms_eps",
                help="e.g. TiO2 ≈ 30–80, g-C3N4 ≈ 15–25, ZnO ≈ 8",
            )
            ms_area = st.number_input(
                "Electrode area (cm²)", value=float(area), step=0.01, key="ms_area"
            )
            ms_ph = st.number_input(
                "pH", value=float(ph_value), min_value=0.0, max_value=14.0,
                step=0.1, key="ms_ph",
            )
            ms_C_unit = st.selectbox(
                "Capacitance unit in file", ["F", "μF", "mF"], key="ms_cunit"
            )
            _cunit_map = {"F": 1.0, "μF": 1e-6, "mF": 1e-3}
            ms_auto = st.checkbox(
                "Auto-detect linear region", value=True, key="ms_auto"
            )

        with col_ms2:
            if ms_file:
                try:
                    df_ms = read_csv_safe(ms_file)
                    E_ms = df_ms.iloc[:, 0].to_numpy(float)
                    C_ms = df_ms.iloc[:, 1].to_numpy(float) * _cunit_map[ms_C_unit]

                    ms_ana = MottSchottkyAnalyzer(
                        epsilon_r=ms_eps,
                        area_cm2=ms_area,
                        e_ref_vs_nhe=e_ref_val,
                        pH=ms_ph,
                    )
                    ms_result = ms_ana.analyze(
                        E_ms, C_ms, auto_region=ms_auto
                    )
                    st.session_state["ms_result"] = ms_result
                    st.session_state["ms_data"]   = (E_ms, C_ms)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("V_fb (vs ref)",
                              f"{ms_result.Vfb_V:+.4f} V")
                    c2.metric("V_fb (vs RHE)",
                              f"{ms_result.Vfb_vs_RHE:+.4f} V")
                    c3.metric("N_d (cm⁻³)",
                              f"{ms_result.Nd_cm3:.3e}")
                    st.metric("Semiconductor type", ms_result.semiconductor_type)
                    st.metric("R² (linear region)", f"{ms_result.R2:.5f}")

                    if ms_result.warning:
                        st.warning(ms_result.warning)

                except Exception as exc:
                    st.error(f"Mott-Schottky error: {exc}")

        if "ms_result" in st.session_state:
            E_ms, C_ms = st.session_state["ms_data"]
            ms_r = st.session_state["ms_result"]

            C_inv2_ms = 1.0 / (C_ms ** 2)
            E_fit = np.linspace(E_ms.min(), E_ms.max(), 200)
            C_fit = ms_r.slope * E_fit + ms_r.intercept

            fig_ms = go.Figure()
            fig_ms.add_trace(go.Scatter(
                x=E_ms, y=C_inv2_ms,
                mode="markers", name="1/C² (exp)",
                marker=dict(color="#6d28d9", size=6),
            ))
            fig_ms.add_trace(go.Scatter(
                x=E_fit, y=C_fit,
                mode="lines", name=f"Linear fit (R²={ms_r.R2:.4f})",
                line=dict(color="#2563eb", width=2, dash="dash"),
            ))
            fig_ms.add_vline(
                x=ms_r.Vfb_V, line_dash="dot", line_color="#d97706",
                annotation_text=f"V_fb = {ms_r.Vfb_V:+.3f} V",
                annotation_font=dict(color="#d97706"),
            )
            fig_ms.update_layout(
                title=f"Mott-Schottky Plot — {ms_r.semiconductor_type}",
                xaxis_title="E (V vs ref)",
                yaxis_title="1/C² (F⁻²)",
                template="plotly_white",
            )
            st.plotly_chart(fig_ms, use_container_width=True)
'''

if __name__ == "__main__":
    print("patch19+20 snippet loaded.")
    print("Paste _BATCH_EIS_TAB_CODE into app.py after tab8 block.")
    print("Paste _BAND_EDGE_TAB_CODE  into app.py after Batch EIS block.")
