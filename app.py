"""
EISForge — Advanced EIS Analysis with Physics-Informed ML
Author: Hoda Jafari | May 2026
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="EISForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg:#ffffff; --surface:#f8f9fa; --border:#e2e8f0; --accent:#2563eb;
    --accent2:#7c3aed; --success:#16a34a; --warning:#d97706; --danger:#dc2626;
    --text:#1e293b; --muted:#64748b;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important; color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important; border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.title {
    font-size: 2.4rem; font-weight: 700; color: var(--accent);
    text-align: center; margin: 0; letter-spacing: -1px;
}
.subtitle {
    color: var(--muted); font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace; text-align: center; margin-top: 0.3rem;
}
.section-title {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px;
    color: var(--muted); margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
.stButton > button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important; padding: 0.4rem 1.2rem !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.8rem;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; }
.fmt {
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
    background: #eff6ff; color: var(--accent); border: 1px solid #bfdbfe; margin: 0.1rem;
}
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] {
    color: var(--accent) !important; border-bottom-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="title">⚡ EISForge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Physics-Informed ML for EIS Analysis · by Hoda Jafari · 2026 · '
    '<a href="https://github.com/Hj1308/EISforge-" style="color:#2563eb">GitHub</a></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center;margin:.4rem 0 .8rem 0;">'
    '<span class="fmt">.idf Autolab</span>'
    '<span class="fmt">.dta Gamry</span>'
    '<span class="fmt">.mpt BioLogic</span>'
    '<span class="fmt">.csv</span>'
    '<span class="fmt">.txt</span></div>',
    unsafe_allow_html=True,
)
st.divider()

EIS_FORMATS = ["idf", "dta", "mpt", "mpr", "csv", "txt"]
CV_FORMATS  = ["idf", "csv", "txt", "dta"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
    font=dict(family="Inter", color="#1e293b"),
    margin=dict(l=60, r=20, t=50, b=50),
)

E_REF_MAP = {
    "RHE": 0.000, "Ag/AgCl (sat.)": 0.197, "Ag/AgCl (3M KCl)": 0.210,
    "SCE": 0.241, "Hg/HgO (1M KOH)": 0.098, "NHE/SHE": 0.000,
}
UNIT_MAP = {"A": 1000.0, "mA": 1.0, "μA": 1e-3, "nA": 1e-6}


# ── File utilities ────────────────────────────────────────────────────────────

def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ["latin-1", "cp1252", "utf-8", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python",
                               comment="#", skip_blank_lines=True)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1", errors="replace",
                       sep=None, engine="python", comment="#")


def save_upload(f) -> str:
    suffix = Path(f.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(f.read())
        return t.name


def load_eis(f):
    suffix = Path(f.name).suffix.lower()
    tmp = save_upload(f)
    try:
        if suffix == ".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            ds = AutolabIDFParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            ds = GamryParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix in (".mpt", ".mpr"):
            from galvani import BioLogic
            mpr = BioLogic.MPRfile(tmp)
            df = mpr.DF
            return df["freq/Hz"].to_numpy(), df["Re(Z)/Ohm"].to_numpy(), \
                   -df["-Im(Z)/Ohm"].to_numpy(), {"source": "BioLogic"}
        else:
            df = read_csv_safe(tmp)
            c = df.columns.tolist()
            fr = df[c[0]].to_numpy(float)
            zr = df[c[1]].to_numpy(float)
            zi = df[c[2]].to_numpy(float)
            if zi.mean() < 0:
                zi = -zi
            return fr, zr, zi, {}
    finally:
        os.unlink(tmp)


def load_cv_lsv(f, unit_factor=1.0):
    suffix = Path(f.name).suffix.lower()
    tmp = save_upload(f)
    try:
        if suffix == ".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            ds = AutolabIDFParser().parse(tmp)
            return ds.z_real, ds.z_imag * unit_factor
        else:
            df = read_csv_safe(tmp)
            c = df.columns.tolist()
            pot = df[c[0]].to_numpy(float)
            cur = df[c[1]].to_numpy(float) * unit_factor
            return pot, cur
    finally:
        os.unlink(tmp)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">System Settings</p>', unsafe_allow_html=True)
    system_type = st.selectbox("System type", ["AOR", "Battery", "Corrosion", "Fuel Cell", "Biosensor"])
    catalyst    = st.text_input("Catalyst", placeholder="e.g. Pt/C, PdAu/C, PtRu/C")
    electrolyte = st.selectbox("Electrolyte", ["Acidic (H₂SO₄)", "Alkaline (KOH)", "NaCl", "PBS", "Other"])
    ekey        = "acidic" if "Acidic" in electrolyte else "alkaline" if "Alkaline" in electrolyte else "acidic"
    alcohol     = st.selectbox("Alcohol", ["ethanol", "methanol", "ethylene glycol", "glycerol", "N/A"],
                               disabled=(system_type != "AOR"))
    eis_pot     = st.number_input("EIS potential (V)", value=0.5, step=0.01)

    st.divider()
    st.markdown('<p class="section-title">Electrode Parameters</p>', unsafe_allow_html=True)
    area    = st.number_input("Geometric area (cm²)", value=1.0, step=0.01, min_value=0.001)
    ecsa    = st.number_input("ECSA (cm²_metal)",     value=0.0, step=0.1,  min_value=0.0)
    loading = st.number_input("Loading (mg/cm²)",     value=0.0, step=0.01, min_value=0.0)

    st.divider()
    st.markdown('<p class="section-title">Experimental Conditions</p>', unsafe_allow_html=True)
    temperature  = st.number_input("Temperature (°C)", value=25, min_value=0, max_value=200)
    current_unit = st.selectbox("Current unit", ["mA", "A", "μA", "nA"])
    e_ref_type   = st.selectbox("Reference electrode", list(E_REF_MAP.keys()))
    e_ref_val    = E_REF_MAP[e_ref_type]
    elec_conc    = st.number_input("Electrolyte conc. (M)", value=0.5, step=0.1)
    sub_conc     = st.number_input("Substrate conc. (M)",   value=1.0, step=0.1)
    unit_factor  = UNIT_MAP.get(current_unit, 1.0)
    if e_ref_val != 0:
        st.info(f"RHE conversion: +{e_ref_val:.3f} V")

    st.divider()
    if st.button("📚 Literature Guide"):
        try:
            from eisforge.knowledge.literature_engine import LiteratureEngine
            g = LiteratureEngine().query(
                system_type=system_type, catalyst=catalyst,
                electrolyte=ekey,
                alcohol=alcohol if system_type == "AOR" else "",
                potential=eis_pot,
            )
            st.session_state["lit"] = g
        except Exception as e:
            st.error(str(e))

    if "lit" in st.session_state and st.session_state["lit"].system_found:
        g = st.session_state["lit"]
        st.success(f"✅ {g.system_name}")
        st.code(f"Circuit: {g.recommended_circuit}")
        for w in g.warnings:
            st.warning(w)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 CV Analysis", "📉 LSV Analysis", "🔬 EIS Analysis",
    "🤖 EIS-GPT", "🔗 Correlation",
])


# ══════════════════ CV ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Cyclic Voltammetry Analysis</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        cv_file = st.file_uploader("Upload CV file", type=CV_FORMATS, key="cv_up")
        sr_cv   = st.number_input("Scan rate (mV/s)", value=50, min_value=1)
        om      = st.radio("E_onset method", ["tangent", "threshold", "derivative"], horizontal=True)

    with col2:
        if cv_file:
            try:
                if Path(cv_file.name).suffix.lower() == ".idf":
                    pot, cur = load_cv_lsv(cv_file, unit_factor=1.0)
                    st.info("Current auto-converted from A → mA (Autolab)")
                else:
                    pot, cur = load_cv_lsv(cv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot)} points loaded | {cv_file.name}")

                from eisforge.analysis.cv_analyzer import CVAnalyzer
                ana = CVAnalyzer(
                    scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                    catalyst_loading=loading, onset_method=om, electrolyte=ekey,
                )
                r = ana.analyze(pot, cur)
                st.session_state.update({"cv_r": r, "cv_pot": pot, "cv_cur": cur})

            except Exception as e:
                st.error(f"Error: {e}")

    if "cv_r" in st.session_state:
        r = st.session_state["cv_r"]
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E_onset",  f"{r.e_onset:.4f} V")
        c2.metric("I_f",      f"{r.i_forward_peak:.4f} mA")
        c3.metric("I_b",      f"{r.i_backward_peak:.4f} mA")
        c4.metric("I_f/I_b",  f"{r.if_ib_ratio:.3f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("j_f (geometric)", f"{r.j_forward_peak:.4f} mA/cm²")
        c6.metric("j_b (geometric)", f"{r.j_backward_peak:.4f} mA/cm²")
        if r.ecsa > 0:
            c7.metric("j_f (ECSA)", f"{r.j_specific_forward:.4f} mA/cm²_Pt")

        st.info(f"**Interpretation:** {r.interpretation}")

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state["cv_pot"], y=st.session_state["cv_cur"],
            mode="lines", name="CV", line=dict(color="#2563eb", width=2),
        ))
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset = {r.e_onset:.3f} V")
        fig.add_trace(go.Scatter(
            x=[r.e_forward_peak], y=[r.i_forward_peak], mode="markers",
            name=f"I_f = {r.i_forward_peak:.3f} mA",
            marker=dict(color="#16a34a", size=12, symbol="star"),
        ))
        fig.add_trace(go.Scatter(
            x=[r.e_backward_peak], y=[r.i_backward_peak], mode="markers",
            name=f"I_b = {r.i_backward_peak:.3f} mA",
            marker=dict(color="#dc2626", size=12, symbol="star"),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"CV — {sr_cv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}",
            xaxis_title=f"Potential (V vs {e_ref_type})",
            yaxis_title="Current (mA)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════ LSV ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Linear Sweep Voltammetry Analysis</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        lsv_file = st.file_uploader("Upload LSV file", type=CV_FORMATS, key="lsv_up")
        sr_lsv   = st.number_input("Scan rate (mV/s)", value=5, min_value=1, key="sr_lsv")
        tj_min   = st.number_input("Tafel j_min (mA/cm²)", value=0.1, step=0.05)
        tj_max   = st.number_input("Tafel j_max (mA/cm²)", value=2.0, step=0.5)

    with col2:
        if lsv_file:
            try:
                if Path(lsv_file.name).suffix.lower() == ".idf":
                    pot_lsv, cur_lsv = load_cv_lsv(lsv_file, unit_factor=1.0)
                else:
                    pot_lsv, cur_lsv = load_cv_lsv(lsv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot_lsv)} points | {lsv_file.name}")

                from eisforge.analysis.lsv_analyzer import LSVAnalyzer
                la = LSVAnalyzer(
                    scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                    catalyst_loading=loading, electrolyte=ekey,
                    e_ref_vs_rhe=e_ref_val, tafel_current_range=(tj_min, tj_max),
                )
                lr = la.analyze(pot_lsv, cur_lsv)
                st.session_state.update({"lsv_r": lr, "lsv_pot": pot_lsv, "lsv_cur": cur_lsv})

            except Exception as e:
                st.error(f"Error: {e}")

    if "lsv_r" in st.session_state:
        import math
        r = st.session_state["lsv_r"]
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("E_onset",     f"{r.e_onset:.4f} V")
        c2.metric("Tafel slope", f"{r.tafel_slope:.1f} mV/dec")
        c3.metric("j₀",          f"{r.exchange_current_density:.3e} mA/cm²")

        c4, c5, c6 = st.columns(3)
        c4.metric("η @ 10 mA/cm²",  f"{r.overpotential_10*1000:.1f} mV"  if not math.isnan(r.overpotential_10)  else "N/A")
        c5.metric("η @ 50 mA/cm²",  f"{r.overpotential_50*1000:.1f} mV"  if not math.isnan(r.overpotential_50)  else "N/A")
        c6.metric("η @ 100 mA/cm²", f"{r.overpotential_100*1000:.1f} mV" if not math.isnan(r.overpotential_100) else "N/A")

        if loading > 0: st.metric("Mass activity",     f"{r.mass_activity:.3f} mA/mg_cat")
        if ecsa > 0:    st.metric("Specific activity", f"{r.specific_activity:.4f} mA/cm²_Pt")

        st.info(f"**Mechanism:** {r.mechanism_interpretation}")
        st.success(f"**Performance:** {r.performance_rating}")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("LSV Curve", "Tafel Plot"))
        j_lsv = st.session_state["lsv_cur"] / area
        p_lsv = st.session_state["lsv_pot"] + e_ref_val

        fig.add_trace(go.Scatter(
            x=p_lsv, y=j_lsv, mode="lines", name="LSV",
            line=dict(color="#2563eb", width=2),
        ), row=1, col=1)
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706", row=1, col=1)

        mask = (j_lsv > 0) & (j_lsv >= tj_min) & (j_lsv <= tj_max)
        if np.sum(mask) > 3:
            fig.add_trace(go.Scatter(
                x=np.log10(j_lsv[mask]), y=p_lsv[mask], mode="markers", name="Tafel region",
                marker=dict(color="#7c3aed", size=6),
            ), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=f"LSV — {sr_lsv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}")
        fig.update_xaxes(title_text=f"Potential (V vs {e_ref_type})", row=1, col=1)
        fig.update_yaxes(title_text="j (mA/cm²)", row=1, col=1)
        fig.update_xaxes(title_text="log(j) [mA/cm²]", row=1, col=2)
        fig.update_yaxes(title_text="E (V)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════ EIS ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Electrochemical Impedance Spectroscopy</p>', unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        eis_file = st.file_uploader("Upload EIS file", type=EIS_FORMATS, key="eis_up")
        lit_c = "R0-p(R1,CPE1)"
        lit_g = "30, 31000, 2e-7, 0.78"
        if "lit" in st.session_state and st.session_state["lit"].system_found:
            g = st.session_state["lit"]
            lit_c = g.recommended_circuit
            lit_g = ", ".join(f"{v:.3e}" for v in g.initial_guess.values())

        circ = st.text_input("Equivalent circuit", value=lit_c,
                             help="e.g. R0-p(R1,CPE1) or R0-p(R1,CPE1)-W1")
        p0s  = st.text_input("Initial guess (comma-separated)", value=lit_g)

    with col2:
        if eis_file:
            try:
                fr, zr, zi, meta = load_eis(eis_file)
                st.success(f"✅ {len(fr)} raw points | {eis_file.name}")

                if meta:
                    with st.expander("📋 File Metadata"):
                        for k, v in meta.items():
                            st.text(f"{k}: {v}")

                st.session_state.update({
                    "eis_fr_raw": fr, "eis_zr_raw": zr, "eis_zi_raw": zi,
                    "eis_fr": fr, "eis_zr": zr, "eis_zi": zi,
                })
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Preprocessing section ─────────────────────────────────────────────────
    if "eis_fr_raw" in st.session_state:
        st.divider()
        st.markdown('<p class="section-title">🧹 Data Preprocessing</p>', unsafe_allow_html=True)

        prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)

        with prep_col1:
            remove_inductive = st.checkbox(
                "Remove inductive artifacts",
                value=True,
                help="Remove points where -Z'' < 0 (cable inductance)",
            )

        with prep_col2:
            remove_jumps = st.checkbox(
                "Remove |Z| jumps",
                value=True,
                help="Remove points with sudden jumps in Z' or Z''",
            )
            jump_threshold = st.slider(
                "Jump threshold (%)", min_value=10, max_value=100, value=20, step=5,
                disabled=not remove_jumps,
            )

        with prep_col3:
            crop_freq = st.checkbox("Crop frequency range", value=False)
            f_min_in = st.number_input("f_min (Hz)", value=0.01, format="%.4g",
                                       disabled=not crop_freq)
            f_max_in = st.number_input("f_max (Hz)", value=100000.0, format="%.4g",
                                       disabled=not crop_freq)

        with prep_col4:
            drop_mains = st.checkbox(
                "Drop mains noise",
                value=False,
                help="Remove 50/60 Hz mains and harmonics",
            )
            mains_freq = st.selectbox(
                "Mains frequency", [50.0, 60.0], index=0, disabled=not drop_mains,
                help="50 Hz: EU/Asia | 60 Hz: USA",
            )
            mains_tol = st.number_input(
                "Tolerance (Hz)", value=0.5, step=0.1, disabled=not drop_mains,
            )

        if st.button("🧹 Apply Preprocessing", type="primary"):
            try:
                from eisforge.core.preprocessor import DataPreprocessor
                from eisforge.parsers.base_parser import EISDataset

                ds = EISDataset(
                    frequency=st.session_state["eis_fr_raw"],
                    z_real=st.session_state["eis_zr_raw"],
                    z_imag=st.session_state["eis_zi_raw"],
                )
                n_before = len(ds.frequency)

                if remove_inductive:
                    ds = DataPreprocessor.remove_inductive_artifacts(ds, verbose=False)
                if crop_freq:
                    ds = DataPreprocessor.crop_frequencies(
                        ds, f_min=f_min_in, f_max=f_max_in, verbose=False,
                    )
                if remove_jumps:
                    ds = DataPreprocessor.remove_z_jumps(
                        ds, threshold_pct=float(jump_threshold), verbose=False,
                    )
                if drop_mains:
                    ds = DataPreprocessor.drop_specific_frequency(
                        ds, target_freq=float(mains_freq),
                        tolerance_hz=float(mains_tol), verbose=False,
                    )

                n_after = len(ds.frequency)
                n_removed = n_before - n_after

                st.session_state["eis_fr"] = ds.frequency
                st.session_state["eis_zr"] = ds.z_real
                st.session_state["eis_zi"] = ds.z_imag

                if n_removed > 0:
                    st.success(
                        f"✅ Preprocessing complete: {n_before} → {n_after} points "
                        f"({n_removed} removed)"
                    )
                else:
                    st.info(f"✅ Preprocessing complete: no points removed ({n_after} points kept)")

            except Exception as e:
                st.error(f"Preprocessing error: {e}")

    # ── K-K + Fit section ─────────────────────────────────────────────────────
    if "eis_fr" in st.session_state:
        st.divider()
        col_kk, col_fit = st.columns([1, 2])

        with col_kk:
            if st.button("🔍 Run K-K Validation"):
                try:
                    from eisforge.parsers.base_parser import EISDataset
                    from eisforge.core.validators import KramersKronigValidator
                    ds = EISDataset(
                        frequency=st.session_state["eis_fr"],
                        z_real=st.session_state["eis_zr"],
                        z_imag=st.session_state["eis_zi"],
                    )
                    kk = KramersKronigValidator().validate(ds)
                    if kk.passed:
                        st.success(f"K-K: PASSED | {kk.summary()}")
                    else:
                        st.warning(f"K-K: FAILED | {kk.summary()}")
                except Exception as e:
                    st.error(f"K-K error: {e}")

        with col_fit:
            if st.button("▶ Run CNLS Fit"):
                with st.spinner("Fitting in progress..."):
                    try:
                        from eisforge.parsers.base_parser import EISDataset
                        from eisforge.core.fitter import CNLSFitter
                        p0 = [float(x.strip()) for x in p0s.split(",")]
                        ds = EISDataset(
                            frequency=st.session_state["eis_fr"],
                            z_real=st.session_state["eis_zr"],
                            z_imag=st.session_state["eis_zi"],
                        )
                        # Auto-bounds: CPE_n params (values in 0-1) constrained to [0,1]
                        lower, upper = [], []
                        for v in p0:
                            lower.append(0.0)
                            upper.append(1.0 if 0 < v <= 1 else np.inf)

                        fit = CNLSFitter(circ, p0, bounds=(lower, upper),
                                         remove_outliers=False).fit(ds)
                        st.session_state["fit"] = fit

                        msg = f"χ² = {fit.chi_squared:.4f}"
                        if fit.converged:
                            st.success(f"✅ Fit converged | {msg}")
                        else:
                            st.warning(f"⚠ Fit did not fully converge | {msg}")
                    except Exception as e:
                        st.error(f"Fit error: {e}")

        if "fit" in st.session_state:
            fit = st.session_state["fit"]
            st.dataframe(pd.DataFrame([
                {"Parameter": n,
                 "Value": f"{v:.4e}",
                 "±Error": f"{fit.parameter_errors.get(n, float('nan')):.2e}",
                 "Rel. Error (%)": f"{abs(fit.parameter_errors.get(n, float('nan'))/v)*100:.2f}"
                                   if v != 0 and not np.isnan(v) else "N/A"}
                for n, v in fit.parameters.items()
            ]), use_container_width=True, hide_index=True)

            # Plot
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fr = st.session_state["eis_fr"]
            zr = st.session_state["eis_zr"]
            zi = st.session_state["eis_zi"]

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Nyquist Plot", "Bode Plot (|Z| vs f)"))
            fig.add_trace(go.Scatter(
                x=zr, y=zi, mode="markers", name="Measured",
                marker=dict(color="#2563eb", size=8),
            ), row=1, col=1)
            if fit.z_fit is not None:
                fig.add_trace(go.Scatter(
                    x=fit.z_fit.real, y=-fit.z_fit.imag, mode="lines",
                    name=f"Fit (χ²={fit.chi_squared:.4f})",
                    line=dict(color="#dc2626", width=2, dash="dash"),
                ), row=1, col=1)
            zm = np.sqrt(zr**2 + zi**2)
            fig.add_trace(go.Scatter(
                x=fr, y=zm, mode="markers+lines", name="|Z|",
                marker=dict(color="#16a34a", size=5), line=dict(color="#16a34a", width=1),
            ), row=1, col=2)
            fig.update_xaxes(title_text="Z' (Ω)",       row=1, col=1)
            fig.update_yaxes(title_text="-Z'' (Ω)",     row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
            fig.update_yaxes(title_text="|Z| (Ω)",      type="log", row=1, col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════ EIS-GPT ══════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">EIS-GPT — AI Circuit Prediction</p>', unsafe_allow_html=True)
    st.info(
        "**EIS-GPT** predicts the equivalent circuit directly from your EIS spectrum "
        "using a Physics-Informed Transformer."
    )

    if "eis_fr" in st.session_state:
        if st.button("▶ Predict with EIS-GPT"):
            with st.spinner("EIS-GPT analyzing spectrum..."):
                try:
                    import torch
                    from eisforge.ml.eis_gpt.transformer import EISForgeModel
                    m = EISForgeModel(d_model=128, n_heads=8, n_layers=6)
                    fr = torch.tensor(st.session_state["eis_fr"]).float().unsqueeze(0)
                    zr = torch.tensor(st.session_state["eis_zr"]).float().unsqueeze(0)
                    zi = torch.tensor(st.session_state["eis_zi"]).float().unsqueeze(0)
                    res = m.predict(fr, zr, zi)
                    st.session_state["gpt"] = res
                except Exception as e:
                    st.error(f"Error: {e}")

        if "gpt" in st.session_state:
            res = st.session_state["gpt"]
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Predicted circuit", res["predicted_circuit"])
            c2.metric("Confidence",        f"{res['confidence']*100:.1f}%")
            st.markdown("**Top 3 candidates:**")
            for c in res["top3"]:
                st.progress(c["probability"],
                            text=f"{c['circuit']} — {c['probability']*100:.1f}%")
            st.caption(
                "Note: Model not yet trained on real data. "
                "Train using scripts/train_ml_models.py for accurate predictions."
            )
    else:
        st.warning("Please upload an EIS file in the 'EIS Analysis' tab first.")


# ══════════════════ Correlation ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">EIS — CV — LSV Correlation</p>', unsafe_allow_html=True)

    has_cv  = "cv_r"  in st.session_state
    has_lsv = "lsv_r" in st.session_state
    has_fit = "fit"   in st.session_state

    if has_cv and has_lsv:
        st.markdown("#### E_onset Comparison")
        c1, c2, c3 = st.columns(3)
        c1.metric("E_onset (CV)",  f"{st.session_state['cv_r'].e_onset:.4f} V")
        c2.metric("E_onset (LSV)", f"{st.session_state['lsv_r'].e_onset:.4f} V")
        diff = abs(st.session_state["cv_r"].e_onset - st.session_state["lsv_r"].e_onset)
        c3.metric("Difference", f"{diff*1000:.1f} mV",
                  delta="Good agreement" if diff < 0.05 else "Large discrepancy",
                  delta_color="normal" if diff < 0.05 else "inverse")
        st.divider()

    if not (has_cv or has_lsv):
        st.warning("Please analyze a CV or LSV spectrum first.")
    if not has_fit:
        st.warning("Please fit an EIS spectrum first.")

    if (has_cv or has_lsv) and has_fit:
        cv_res = st.session_state.get("cv_r") or st.session_state.get("lsv_r")
        if st.button("▶ Run EIS-CV Correlation"):
            try:
                from eisforge.analysis.eis_cv_correlator import EISCVCorrelator
                corr = EISCVCorrelator(electrolyte=ekey).correlate(
                    cv_result=cv_res,
                    eis_fit_result=st.session_state["fit"],
                    eis_potential=eis_pot,
                )
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("E_onset",     f"{corr.e_onset:.4f} V")
                c2.metric("EIS region",  corr.eis_region)
                c3.metric("Consistency", f"{corr.consistency_score:.0%}")

                for w in corr.warnings:
                    st.warning(w)
                for r in corr.recommendations:
                    st.success(r)

                st.markdown("#### Combined Results Summary")
                summary = {"Parameter": [], "CV/LSV": [], "EIS": []}
                summary["Parameter"].append("E_onset (V)")
                summary["CV/LSV"].append(f"{cv_res.e_onset:.4f}")
                summary["EIS"].append(f"Measured at {eis_pot:.4f} V")
                if has_cv:
                    summary["Parameter"].append("I_f/I_b")
                    summary["CV/LSV"].append(f"{st.session_state['cv_r'].if_ib_ratio:.3f}")
                    summary["EIS"].append("—")
                summary["Parameter"].append("R_ct (Ω)")
                summary["CV/LSV"].append("—")
                summary["EIS"].append(f"{corr.r_ct:.2f}")
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")
