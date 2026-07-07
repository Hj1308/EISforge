"""
EISForge — Band Edge & Mott-Schottky Page (patch20)
=====================================================
Computes semiconductor band positions and runs Mott-Schottky analysis.

Features
--------
- Band edge calculator: χ + Eg → Ecb/Evb vs vacuum, NHE, RHE
- Built-in material DB (g-C3N4, TiO2, ZnO, BCN, WO3, Fe2O3, BiVO4)
- Mott-Schottky: upload Z data → compute C → 1/C² vs V → Vfb, Nd, type
- Interactive band diagram (Plotly)
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eisforge.analysis.band_edge_calculator import (
    MATERIALS_DB,
    BandEdgeCalculator,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Band Edge — EISForge", page_icon="⚡", layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
body,[data-testid="stAppViewContainer"]{font-family:'Plus Jakarta Sans',sans-serif;background:#fff;}
.title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#18162a;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🔬 Band Edge & Mott-Schottky Calculator</h1>',
            unsafe_allow_html=True)
st.caption(
    "Butler-Ginley formula (1978) for Ecb/Evb. "
    "Mott-Schottky analysis for flat-band potential (Vfb) and carrier density (Nd)."
)
st.divider()

tab_be, tab_ms = st.tabs(["📐 Band Edge Calculator", "📊 Mott-Schottky Analysis"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Band Edge
# ══════════════════════════════════════════════════════════════════════════════
with tab_be:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("#### Material & Conditions")
        mat_choice = st.selectbox("Material", list(MATERIALS_DB.keys()))
        db = MATERIALS_DB[mat_choice]

        chi_default = db["chi"] if db["chi"] is not None else 5.0
        Eg_default  = db["Eg"]  if db["Eg"]  is not None else 2.7

        chi_val = st.number_input(
            "Electronegativity χ (eV)",
            value=float(chi_default), step=0.01, format="%.3f",
            help="Absolute electronegativity of the semiconductor (Butler-Ginley).",
        )
        Eg_val = st.number_input(
            "Band gap Eg (eV)",
            value=float(Eg_default), step=0.01, format="%.3f",
            help="Optical band gap from Tauc plot or literature.",
        )
        ph_val = st.number_input(
            "Solution pH", value=7.0, min_value=0.0, max_value=14.0, step=0.1
        )
        T_val = st.number_input(
            "Temperature (°C)", value=25.0, min_value=0.0, max_value=200.0, step=1.0
        )

        if db["notes"]:
            st.caption(f"ℹ️ {db['notes']}")
        if db["Eg"] is None:
            st.warning("Eg not in DB for this material — enter your Tauc plot value above.")

        calc_btn = st.button("Calculate Band Edges", type="primary")

    with col_right:
        if calc_btn:
            calc = BandEdgeCalculator(pH=ph_val, T_celsius=T_val)
            result = calc.band_edges(chi=chi_val, Eg=Eg_val, material=mat_choice)
            st.session_state["be_result"] = result

        if "be_result" in st.session_state:
            r = st.session_state["be_result"]

            st.markdown(f"#### Results — {r.material}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Ecb vs NHE", f"{r.Ecb_NHE:+.4f} V")
            c1.metric("Evb vs NHE", f"{r.Evb_NHE:+.4f} V")
            c2.metric("Ecb vs RHE", f"{r.Ecb_RHE:+.4f} V",
                      help=f"pH = {r.pH:.1f}")
            c2.metric("Evb vs RHE", f"{r.Evb_RHE:+.4f} V")
            c3.metric("Ecb vs vacuum", f"{r.Ecb_vac:+.4f} eV")
            c3.metric("Evb vs vacuum", f"{r.Evb_vac:+.4f} eV")

            # reference lines
            H2_NHE   =  0.00
            O2_NHE   =  1.23
            CO2_NHE  = -0.53
            MeOH_NHE = -0.02

            fig_bd = go.Figure()

            # semiconductor band block
            fig_bd.add_shape(
                type="rect",
                x0=0.3, x1=0.7,
                y0=r.Evb_NHE, y1=r.Ecb_NHE,
                fillcolor="rgba(109,40,217,0.15)",
                line=dict(color="#6d28d9", width=2),
            )
            # Ecb label
            fig_bd.add_annotation(
                x=0.5, y=r.Ecb_NHE, text=f"Ecb = {r.Ecb_NHE:+.3f} V",
                showarrow=False, yanchor="bottom",
                font=dict(color="#6d28d9", size=12),
            )
            # Evb label
            fig_bd.add_annotation(
                x=0.5, y=r.Evb_NHE, text=f"Evb = {r.Evb_NHE:+.3f} V",
                showarrow=False, yanchor="top",
                font=dict(color="#6d28d9", size=12),
            )

            # redox reference lines
            ref_lines = [
                (H2_NHE,   "H⁺/H₂  (0.00 V)",   "#1d4ed8", "dot"),
                (O2_NHE,   "O₂/H₂O (1.23 V)",   "#b91c1c", "dot"),
                (CO2_NHE,  "CO₂/CH₃OH (−0.53 V)", "#047857", "dash"),
                (MeOH_NHE, "MeOH/CO₂ (−0.02 V)", "#d97706", "dash"),
            ]
            for y_val, lbl, col, dash in ref_lines:
                fig_bd.add_hline(
                    y=y_val,
                    line=dict(color=col, dash=dash, width=1.2),
                    annotation_text=lbl,
                    annotation_position="right",
                    annotation_font=dict(color=col, size=10),
                )

            fig_bd.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8f9fa",
                font=dict(family="Inter"),
                title=f"Band Diagram — {r.material} (vs NHE, pH {r.pH:.1f})",
                xaxis=dict(visible=False, range=[0, 1]),
                yaxis=dict(
                    title="Potential vs NHE (V)",
                    autorange="reversed",  # negative = more reducing = top
                ),
                height=480,
                margin=dict(l=60, r=140, t=60, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_bd, use_container_width=True)
            st.code(r.summary(), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Mott-Schottky
# ══════════════════════════════════════════════════════════════════════════════
with tab_ms:
    st.markdown("#### Upload EIS Data for Mott-Schottky Analysis")
    st.caption(
        "Upload impedance spectra measured at multiple DC potentials. "
        "The tool extracts capacitance at the chosen frequency and plots 1/C² vs V."
    )

    col_ms1, col_ms2 = st.columns([1, 1.5])

    with col_ms1:
        ms_files = st.file_uploader(
            "Upload EIS files (one per DC potential)",
            type=["idf", "dta", "mpt", "csv", "txt"],
            accept_multiple_files=True,
            key="ms_upload",
        )
        target_freq = st.number_input(
            "Frequency for C extraction (Hz)",
            value=1000.0, min_value=0.01, step=100.0,
            help="Capacitance is computed at this frequency: C = 1/(ω·|Z''|)",
        )
        epsilon_r = st.number_input(
            "Relative permittivity εᵣ",
            value=float(MATERIALS_DB["g-C3N4"]["epsilon_r"]),
            min_value=1.0, step=0.5,
            help="From the material DB or literature.",
        )
        area_cm2 = st.number_input(
            "Electrode area (cm²)",
            value=1.0, min_value=0.0001, step=0.01, format="%.4f",
        )
        ph_ms = st.number_input(
            "pH (for RHE conversion)",
            value=7.0, min_value=0.0, max_value=14.0, step=0.1,
            key="ms_ph",
        )
        e_ref_ms = st.number_input(
            "Reference electrode offset vs NHE (V)",
            value=0.197, step=0.001, format="%.3f",
            help="Ag/AgCl sat. = 0.197 V, SCE = 0.241 V, RHE = 0.0",
        )
        use_potential_window = st.checkbox("Restrict fit to potential window")
        if use_potential_window:
            v_low  = st.number_input("V_low  (V vs ref)", value=-0.5, step=0.05, format="%.3f")
            v_high = st.number_input("V_high (V vs ref)",  value= 0.2, step=0.05, format="%.3f")
            v_range = (v_low, v_high)
        else:
            v_range = None

        # manual V entry (fallback)
        st.markdown("##### OR — manual V + C entry")
        manual_vc = st.text_area(
            "Paste V, C (F) pairs — one per line",
            value="",
            placeholder="-0.8, 1.2e-5\n-0.6, 1.4e-5\n-0.4, 1.7e-5",
            height=100,
        )

    with col_ms2:
        # parse data
        V_ms, C_ms = None, None

        if ms_files:
            import tempfile, os, re
            V_arr, C_arr = [], []
            for f in ms_files:
                suffix = f.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as t:
                    t.write(f.getvalue()); tmp = t.name
                try:
                    if suffix == "idf":
                        from eisforge.parsers.autolab_parser import AutolabIDFParser
                        ds = AutolabIDFParser().parse(tmp)
                        freq, zr, zi = ds.frequency, ds.z_real, ds.z_imag
                    elif suffix == "dta":
                        from eisforge.parsers.gamry_parser import GamryParser
                        ds = GamryParser().parse(tmp)
                        freq, zr, zi = ds.frequency, ds.z_real, ds.z_imag
                    else:
                        import pandas as _pd
                        df = _pd.read_csv(tmp, sep=None, engine="python",
                                          encoding="latin-1", comment="#")
                        c = df.columns.tolist()
                        freq = df[c[0]].to_numpy(float)
                        zr   = df[c[1]].to_numpy(float)
                        zi   = df[c[2]].to_numpy(float)

                    C_val = BandEdgeCalculator.capacitance_from_eis(
                        freq, zr, zi, target_freq=target_freq
                    )
                    # try to parse potential from filename (e.g. "0.2V.idf")
                    pot_m = re.search(r"([+-]?[\d.]+)V", f.name)
                    V_val = float(pot_m.group(1)) if pot_m else float(len(V_arr)) * 0.1
                    V_arr.append(V_val)
                    C_arr.append(C_val)
                except Exception as exc:
                    st.warning(f"Could not load {f.name}: {exc}")
                finally:
                    os.unlink(tmp)

            if V_arr:
                sort_idx = np.argsort(V_arr)
                V_ms = np.array(V_arr)[sort_idx]
                C_ms = np.array(C_arr)[sort_idx]

        elif manual_vc.strip():
            V_list, C_list = [], []
            for line in manual_vc.strip().splitlines():
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        V_list.append(float(parts[0]))
                        C_list.append(float(parts[1]))
                    except ValueError:
                        pass
            if V_list:
                sort_idx = np.argsort(V_list)
                V_ms = np.array(V_list)[sort_idx]
                C_ms = np.array(C_list)[sort_idx]

        if V_ms is not None and C_ms is not None and len(V_ms) >= 3:
            if st.button("▶ Run Mott-Schottky Analysis", type="primary"):
                try:
                    calc_ms = BandEdgeCalculator(pH=ph_ms)
                    ms_res = calc_ms.mott_schottky(
                        V_ms, C_ms, epsilon_r, area_cm2, v_range=v_range
                    )
                    st.session_state["ms_result"] = ms_res
                    st.session_state["ms_V"] = V_ms
                    st.session_state["ms_C"] = C_ms
                except Exception as exc:
                    st.error(f"Mott-Schottky failed: {exc}")

        if "ms_result" in st.session_state:
            ms = st.session_state["ms_result"]
            V_data = st.session_state["ms_V"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("V_fb (vs ref)", f"{ms.V_fb:+.4f} V")
            _vfb_rhe = ms.V_fb + e_ref_ms + 0.0592 * ph_ms
            c2.metric("V_fb (vs RHE)", f"{_vfb_rhe:+.4f} V",
                      help="V_fb + E_ref_offset + 0.0592·pH")
            c3.metric("Nd", f"{ms.Nd:.3e} cm⁻³")
            c4.metric("Type", ms.semiconductor_type)

            inv_C2_all = 1.0 / (st.session_state["ms_C"] ** 2)

            fig_ms = go.Figure()
            fig_ms.add_trace(go.Scatter(
                x=V_data, y=inv_C2_all,
                mode="markers",
                name="1/C² data",
                marker=dict(color="#6d28d9", size=7),
            ))
            # linear fit line
            V_line = np.linspace(ms.V_fit.min(), ms.V_fit.max(), 100)
            invC2_line = ms.slope * V_line + ms.intercept
            fig_ms.add_trace(go.Scatter(
                x=V_line, y=invC2_line,
                mode="lines",
                name=f"Linear fit (R²={ms.R2:.4f})",
                line=dict(color="#dc2626", dash="dash", width=2),
            ))
            fig_ms.add_vline(
                x=ms.V_fb,
                line=dict(color="#059669", dash="dot", width=1.5),
                annotation_text=f"V_fb = {ms.V_fb:+.4f} V",
                annotation_font=dict(color="#059669"),
            )
            fig_ms.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8f9fa",
                font=dict(family="Inter"),
                title=f"Mott-Schottky — {ms.semiconductor_type} | εᵣ={ms.epsilon_r}",
                xaxis_title="Potential (V vs ref)",
                yaxis_title="1/C² (F⁻²)",
                margin=dict(l=70, r=20, t=60, b=50),
            )
            st.plotly_chart(fig_ms, use_container_width=True)
            st.code(ms.summary(), language=None)
        elif V_ms is not None and len(V_ms) < 3:
            st.info("Upload at least 3 EIS files (or 3 V,C pairs) to run Mott-Schottky.")
        elif V_ms is None and not manual_vc.strip():
            st.info("Upload EIS files or enter V, C pairs to start.")

st.caption(
    "Band Edge Calculator by EISForge · patch20 · "
    "[GitHub](https://github.com/Hj1308/EISforge)"
)
