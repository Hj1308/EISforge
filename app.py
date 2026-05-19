"""
EISForge — Advanced EIS Analysis with Physics-Informed ML
Author: Hoda Jafari | May 2026
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="EISForge", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#fff;--surface:#f8f9fa;--border:#e2e8f0;--accent:#2563eb;--accent2:#7c3aed;--success:#16a34a;--warning:#d97706;--danger:#dc2626;--text:#1e293b;--muted:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.title{font-size:2.4rem;font-weight:700;color:var(--accent);text-align:center;margin:0;letter-spacing:-1px;}
.subtitle{color:var(--muted);font-size:.9rem;font-family:'JetBrains Mono',monospace;text-align:center;margin-top:.3rem;}
.section-title{font-size:.65rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:.6rem;font-family:'JetBrains Mono',monospace;font-weight:600;}
.stButton>button{background:var(--accent)!important;color:white!important;border:none!important;border-radius:6px!important;font-weight:600!important;padding:.4rem 1.2rem!important;}
.stButton>button:hover{background:#1d4ed8!important;}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem;}
div[data-testid="stMetric"] label{color:var(--muted)!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--text)!important;}
.fmt{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.7rem;font-family:'JetBrains Mono',monospace;background:#eff6ff;color:var(--accent);border:1px solid #bfdbfe;margin:.1rem;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
.ir-box{background:#fefce8;border:1px solid #fde047;border-radius:8px;padding:.8rem;margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">⚡ EISForge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Physics-Informed ML for EIS Analysis · by Hoda Jafari · 2026 · '
    '<a href="https://github.com/Hj1308/EISforge-" style="color:#2563eb">GitHub</a></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center;margin:.4rem 0 .8rem 0;">'
    '<span class="fmt">.idf Autolab</span><span class="fmt">.dta Gamry</span>'
    '<span class="fmt">.mpt BioLogic</span><span class="fmt">.csv</span>'
    '<span class="fmt">.txt</span></div>',
    unsafe_allow_html=True,
)
st.divider()

EIS_FORMATS = ["idf","dta","mpt","mpr","csv","txt"]
CV_FORMATS  = ["idf","csv","txt","dta"]

PLOTLY_LAYOUT = dict(
    template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
    font=dict(family="Inter", color="#1e293b"), margin=dict(l=60,r=20,t=50,b=50),
)
E_REF_MAP = {
    "RHE":0.000,"Ag/AgCl (sat.)":0.197,"Ag/AgCl (3M KCl)":0.210,
    "SCE":0.241,"Hg/HgO (1M KOH)":0.098,"NHE/SHE":0.000,
}
UNIT_MAP = {"A":1000.0,"mA":1.0,"μA":1e-3,"nA":1e-6}


def smart_bounds(circuit_str, p0):
    n = len(p0)
    lower = [0.0]*n
    upper = [np.inf]*n
    tokens = re.findall(r"[A-Za-z]+\d+", circuit_str)
    p_idx = 0
    for tok in tokens:
        if p_idx >= n: break
        if tok.upper().startswith("CPE"):
            p_idx += 1
            if p_idx < n: upper[p_idx] = 1.0
            p_idx += 1
        elif tok.upper().startswith(("WO","WS","G")):
            p_idx += 2
        else:
            p_idx += 1
    return (lower, upper)


def read_csv_safe(path):
    for enc in ["latin-1","cp1252","utf-8","utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python",
                               comment="#", skip_blank_lines=True)
        except (UnicodeDecodeError,UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1", errors="replace",
                       sep=None, engine="python", comment="#")


def save_upload(f):
    suffix = Path(f.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(f.read()); return t.name


def load_eis(f):
    suffix = Path(f.name).suffix.lower()
    tmp = save_upload(f)
    try:
        if suffix==".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            ds = AutolabIDFParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix==".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            ds = GamryParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix in (".mpt",".mpr"):
            from galvani import BioLogic
            mpr=BioLogic.MPRfile(tmp); df=mpr.DF
            return df["freq/Hz"].to_numpy(), df["Re(Z)/Ohm"].to_numpy(), \
                   -df["-Im(Z)/Ohm"].to_numpy(), {"source":"BioLogic"}
        else:
            df=read_csv_safe(tmp); c=df.columns.tolist()
            fr,zr,zi = df[c[0]].to_numpy(float),df[c[1]].to_numpy(float),df[c[2]].to_numpy(float)
            if zi.mean()<0: zi=-zi
            return fr,zr,zi,{}
    finally:
        os.unlink(tmp)


def load_cv_lsv(f, unit_factor=1.0):
    suffix = Path(f.name).suffix.lower()
    tmp = save_upload(f)
    try:
        if suffix==".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            ds=AutolabIDFParser().parse(tmp)
            return ds.z_real, ds.z_imag*unit_factor
        else:
            df=read_csv_safe(tmp); c=df.columns.tolist()
            return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)*unit_factor
    finally:
        os.unlink(tmp)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">System Settings</p>', unsafe_allow_html=True)
    system_type = st.selectbox("System type",["AOR","Battery","Corrosion","Fuel Cell","Biosensor"])
    catalyst    = st.text_input("Catalyst", placeholder="e.g. Pt/C, PdAu/C, PtRu/C")
    electrolyte = st.selectbox("Electrolyte",["Acidic (H₂SO₄)","Alkaline (KOH)","NaCl","PBS","Other"])
    ekey = "acidic" if "Acidic" in electrolyte else "alkaline" if "Alkaline" in electrolyte else "acidic"
    alcohol = st.selectbox("Alcohol",["ethanol","methanol","ethylene glycol","glycerol","N/A"],
                           disabled=(system_type!="AOR"))
    eis_pot = st.number_input("EIS potential (V)", value=0.5, step=0.01)

    st.divider()
    st.markdown('<p class="section-title">Electrode Parameters</p>', unsafe_allow_html=True)
    area    = st.number_input("Geometric area (cm²)", value=1.0, step=0.01, min_value=0.001)
    ecsa    = st.number_input("ECSA (cm²_metal)",     value=0.0, step=0.1,  min_value=0.0)
    loading = st.number_input("Loading (mg/cm²)",     value=0.0, step=0.01, min_value=0.0)

    st.divider()
    st.markdown('<p class="section-title">Experimental Conditions</p>', unsafe_allow_html=True)
    temperature  = st.number_input("Temperature (°C)", value=25, min_value=0, max_value=200)
    current_unit = st.selectbox("Current unit",["mA","A","μA","nA"])
    e_ref_type   = st.selectbox("Reference electrode", list(E_REF_MAP.keys()))
    e_ref_val    = E_REF_MAP[e_ref_type]
    elec_conc    = st.number_input("Electrolyte conc. (M)", value=0.5, step=0.1)
    sub_conc     = st.number_input("Substrate conc. (M)",   value=1.0, step=0.1)
    unit_factor  = UNIT_MAP.get(current_unit, 1.0)
    if e_ref_val!=0:
        st.info(f"RHE conversion: +{e_ref_val:.3f} V")

    st.divider()
    # iR Compensation — shared across CV and LSV tabs
    st.markdown('<p class="section-title">⚡ iR Compensation</p>', unsafe_allow_html=True)
    use_ir = st.checkbox(
        "Apply iR compensation",
        value=False,
        help="E_corrected = E_measured − I(A) × R_s(Ω)",
    )
    r_s = st.number_input(
        "R_s (Ω) — from EIS fit",
        value=0.0, step=0.1, min_value=0.0,
        disabled=not use_ir,
        help="Use R0 from EIS CNLS fit (high-frequency intercept)",
    )
    if use_ir and r_s > 0:
        st.markdown(
            f'<div class="ir-box">✅ iR compensation active<br>'
            f'R_s = {r_s:.3f} Ω<br>'
            f'iR drop @ 1 mA ≈ {r_s*1e-3*1000:.2f} mV</div>',
            unsafe_allow_html=True,
        )
    elif use_ir and r_s == 0:
        st.warning("Enter R_s from your EIS fit.")

    actual_rs = r_s if use_ir else 0.0

    st.divider()
    if st.button("📚 Literature Guide"):
        try:
            from eisforge.knowledge.literature_engine import LiteratureEngine
            g = LiteratureEngine().query(
                system_type=system_type, catalyst=catalyst,
                electrolyte=ekey,
                alcohol=alcohol if system_type=="AOR" else "",
                potential=eis_pot,
            )
            st.session_state["lit"] = g
        except Exception as e:
            st.error(str(e))

    if "lit" in st.session_state and st.session_state["lit"].system_found:
        g = st.session_state["lit"]
        st.success(f"✅ {g.system_name}")
        st.code(f"Circuit: {g.recommended_circuit}")
        for w in g.warnings: st.warning(w)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 CV Analysis","📉 LSV Analysis","🔬 EIS Analysis","🤖 EIS-GPT","🔗 Correlation"
])


# ══════════════════ CV ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Cyclic Voltammetry Analysis</p>', unsafe_allow_html=True)

    if actual_rs > 0:
        st.info(f"⚡ iR compensation will be applied — R_s = {actual_rs:.3f} Ω "
                f"(from sidebar). E_onset and peaks will be on iR-corrected potential.")

    col1,col2 = st.columns([1,1])
    with col1:
        cv_file = st.file_uploader("Upload CV file", type=CV_FORMATS, key="cv_up")
        sr_cv   = st.number_input("Scan rate (mV/s)", value=50, min_value=1)
        om      = st.radio("E_onset method",["tangent","threshold","derivative"], horizontal=True)

    with col2:
        if cv_file:
            try:
                if Path(cv_file.name).suffix.lower()==".idf":
                    pot,cur = load_cv_lsv(cv_file, unit_factor=1.0)
                    st.info("Current auto-converted from A → mA (Autolab)")
                else:
                    pot,cur = load_cv_lsv(cv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot)} points | {cv_file.name}")

                from eisforge.analysis.cv_analyzer import CVAnalyzer
                ana = CVAnalyzer(scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, onset_method=om, electrolyte=ekey)
                r = ana.analyze(pot, cur, r_s_ohms=actual_rs)
                st.session_state.update({"cv_r":r,"cv_pot":pot,"cv_cur":cur,"cv_pot_corr":
                    CVAnalyzer.apply_ir_compensation(pot, cur, actual_rs) if actual_rs>0 else pot})
            except Exception as e:
                st.error(f"Error: {e}")

    if "cv_r" in st.session_state:
        r = st.session_state["cv_r"]
        st.divider()

        if r.ir_compensated:
            st.success(f"✅ iR-corrected | R_s = {r.r_s_used:.3f} Ω")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("E_onset",  f"{r.e_onset:.4f} V")
        c2.metric("I_f",      f"{r.i_forward_peak:.4f} mA")
        c3.metric("I_b",      f"{r.i_backward_peak:.4f} mA")
        c4.metric("I_f/I_b",  f"{r.if_ib_ratio:.3f}")

        c5,c6,c7 = st.columns(3)
        c5.metric("j_f (geometric)", f"{r.j_forward_peak:.4f} mA/cm²")
        c6.metric("j_b (geometric)", f"{r.j_backward_peak:.4f} mA/cm²")
        if r.ecsa>0: c7.metric("j_f (ECSA)", f"{r.j_specific_forward:.4f} mA/cm²_Pt")

        st.info(f"**Interpretation:** {r.interpretation}")

        import plotly.graph_objects as go
        fig = go.Figure()
        x_plot = st.session_state.get("cv_pot_corr", st.session_state["cv_pot"])
        fig.add_trace(go.Scatter(
            x=x_plot, y=st.session_state["cv_cur"],
            mode="lines", name="CV" + (" (iR-corrected)" if actual_rs>0 else ""),
            line=dict(color="#2563eb", width=2),
        ))
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset = {r.e_onset:.3f} V",
                      annotation_font=dict(color="#d97706"))
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
        title = f"CV — {sr_cv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}"
        if actual_rs>0: title += f" | iR-corrected (R_s={actual_rs:.1f}Ω)"
        fig.update_layout(**PLOTLY_LAYOUT, title=title,
                          xaxis_title=f"Potential (V vs {e_ref_type})",
                          yaxis_title="Current (mA)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════ LSV ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Linear Sweep Voltammetry Analysis</p>', unsafe_allow_html=True)

    if actual_rs > 0:
        st.info(f"⚡ iR compensation will be applied — R_s = {actual_rs:.3f} Ω. "
                f"Tafel slope and E_onset computed on iR-corrected potential.")

    col1,col2 = st.columns([1,1])
    with col1:
        lsv_file = st.file_uploader("Upload LSV file", type=CV_FORMATS, key="lsv_up")
        sr_lsv   = st.number_input("Scan rate (mV/s)", value=5, min_value=1, key="sr_lsv")
        tj_min   = st.number_input("Tafel j_min (mA/cm²)", value=0.1, step=0.05)
        tj_max   = st.number_input("Tafel j_max (mA/cm²)", value=2.0, step=0.5)

    with col2:
        if lsv_file:
            try:
                if Path(lsv_file.name).suffix.lower()==".idf":
                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=1.0)
                else:
                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot_lsv)} points | {lsv_file.name}")

                from eisforge.analysis.lsv_analyzer import LSVAnalyzer
                la = LSVAnalyzer(scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, electrolyte=ekey,
                                 e_ref_vs_rhe=e_ref_val, tafel_current_range=(tj_min,tj_max))
                lr = la.analyze(pot_lsv, cur_lsv, r_s_ohms=actual_rs)
                st.session_state.update({"lsv_r":lr,"lsv_pot":pot_lsv,"lsv_cur":cur_lsv})
            except Exception as e:
                st.error(f"Error: {e}")

    if "lsv_r" in st.session_state:
        import math
        r = st.session_state["lsv_r"]
        st.divider()

        if r.ir_compensated:
            st.success(f"✅ iR-corrected | R_s = {r.r_s_used:.3f} Ω")

        c1,c2,c3 = st.columns(3)
        c1.metric("E_onset",     f"{r.e_onset:.4f} V")
        c2.metric("Tafel slope", f"{r.tafel_slope:.1f} mV/dec")
        c3.metric("j₀",          f"{r.exchange_current_density:.3e} mA/cm²")

        c4,c5,c6 = st.columns(3)
        c4.metric("η @ 10 mA/cm²",  f"{r.overpotential_10*1000:.1f} mV"  if not math.isnan(r.overpotential_10)  else "N/A")
        c5.metric("η @ 50 mA/cm²",  f"{r.overpotential_50*1000:.1f} mV"  if not math.isnan(r.overpotential_50)  else "N/A")
        c6.metric("η @ 100 mA/cm²", f"{r.overpotential_100*1000:.1f} mV" if not math.isnan(r.overpotential_100) else "N/A")

        if loading>0: st.metric("Mass activity",     f"{r.mass_activity:.3f} mA/mg_cat")
        if ecsa>0:    st.metric("Specific activity", f"{r.specific_activity:.4f} mA/cm²_Pt")

        st.info(f"**Mechanism:** {r.mechanism_interpretation}")
        st.success(f"**Performance:** {r.performance_rating}")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1,cols=2,subplot_titles=("LSV Curve","Tafel Plot"))

        j_lsv = st.session_state["lsv_cur"]/area
        # Use iR-corrected potential if applicable
        if actual_rs>0:
            from eisforge.analysis.lsv_analyzer import LSVAnalyzer
            cur_ma_lsv = st.session_state["lsv_cur"]*unit_factor
            p_lsv = LSVAnalyzer.apply_ir_compensation(
                st.session_state["lsv_pot"]+e_ref_val, cur_ma_lsv, actual_rs
            )
        else:
            p_lsv = st.session_state["lsv_pot"]+e_ref_val

        fig.add_trace(go.Scatter(x=p_lsv, y=j_lsv, mode="lines",
                                 name="LSV"+" (iR-corr.)" if actual_rs>0 else "LSV",
                                 line=dict(color="#2563eb",width=2)), row=1,col=1)
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset={r.e_onset:.3f}V",
                      annotation_font=dict(color="#d97706"), row=1,col=1)

        mask = (j_lsv>0)&(j_lsv>=tj_min)&(j_lsv<=tj_max)
        if np.sum(mask)>3:
            fig.add_trace(go.Scatter(x=np.log10(j_lsv[mask]),y=p_lsv[mask],
                                     mode="markers",name="Tafel region",
                                     marker=dict(color="#7c3aed",size=6)), row=1,col=2)

        title = f"LSV — {sr_lsv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}"
        if actual_rs>0: title += f" | iR-corrected"
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title=title)
        fig.update_xaxes(title_text=f"Potential (V vs {e_ref_type})", row=1,col=1)
        fig.update_yaxes(title_text="j (mA/cm²)", row=1,col=1)
        fig.update_xaxes(title_text="log(j) [mA/cm²]", row=1,col=2)
        fig.update_yaxes(title_text="E (V)", row=1,col=2)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════ EIS ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Electrochemical Impedance Spectroscopy</p>', unsafe_allow_html=True)

    col1,col2 = st.columns([1,1])
    with col1:
        eis_file = st.file_uploader("Upload EIS file", type=EIS_FORMATS, key="eis_up")
        lit_c = "R0-p(R1,CPE1)"
        lit_g = "30, 31000, 2e-7, 0.78"
        if "lit" in st.session_state and st.session_state["lit"].system_found:
            g=st.session_state["lit"]; lit_c=g.recommended_circuit
            lit_g=", ".join(f"{v:.3e}" for v in g.initial_guess.values())
        circ = st.text_input("Equivalent circuit", value=lit_c)
        p0s  = st.text_input("Initial guess (comma-separated)", value=lit_g)
        use_bounds = st.checkbox("Use smart bounds", value=False)
        st.caption("💡 Tip: After fit, use R0 value as R_s for iR compensation in CV/LSV tabs.")

    with col2:
        if eis_file:
            current_file_id = f"{eis_file.name}_{eis_file.size}"
            already_loaded  = st.session_state.get("eis_loaded_id")==current_file_id
            if not already_loaded:
                try:
                    fr,zr,zi,meta = load_eis(eis_file)
                    st.session_state.update({
                        "eis_loaded_id":current_file_id,
                        "eis_fr_raw":fr,"eis_zr_raw":zr,"eis_zi_raw":zi,"eis_meta":meta,
                        "eis_fr":fr.copy(),"eis_zr":zr.copy(),"eis_zi":zi.copy(),
                        "preprocessed":False,
                    })
                    if "fit" in st.session_state: del st.session_state["fit"]
                    st.success(f"✅ {len(fr)} raw points | {eis_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                n = len(st.session_state["eis_fr"])
                label = "(preprocessed ✨)" if st.session_state.get("preprocessed") else "(raw)"
                st.success(f"✅ {n} points {label} | {eis_file.name}")

            if st.session_state.get("eis_meta"):
                with st.expander("📋 File Metadata"):
                    for k,v in st.session_state["eis_meta"].items():
                        st.text(f"{k}: {v}")

    # Preprocessing
    if "eis_fr_raw" in st.session_state:
        st.divider()
        st.markdown('<p class="section-title">🧹 Data Preprocessing</p>', unsafe_allow_html=True)
        pc1,pc2,pc3,pc4 = st.columns(4)
        with pc1:
            rm_ind = st.checkbox("Remove inductive artifacts", value=True)
        with pc2:
            rm_jmp = st.checkbox("Remove |Z| jumps", value=True)
            jmp_thr = st.slider("Jump threshold (%)", 10, 100, 20, 5, disabled=not rm_jmp)
        with pc3:
            crop = st.checkbox("Crop frequency range", value=False)
            f_min_in = st.number_input("f_min (Hz)", value=0.01, format="%.4g", disabled=not crop)
            f_max_in = st.number_input("f_max (Hz)", value=100000.0, format="%.4g", disabled=not crop)
        with pc4:
            drop_mains = st.checkbox("Drop mains noise", value=False)
            mains_freq = st.selectbox("Mains (Hz)", [50.0, 60.0], disabled=not drop_mains)
            mains_tol  = st.number_input("Tolerance (Hz)", value=0.5, step=0.1, disabled=not drop_mains)

        ca,cb = st.columns([1,1])
        with ca:
            if st.button("🧹 Apply Preprocessing", type="primary"):
                try:
                    from eisforge.core.preprocessor import DataPreprocessor
                    from eisforge.parsers.base_parser import EISDataset
                    ds = EISDataset(frequency=st.session_state["eis_fr_raw"].copy(),
                                    z_real=st.session_state["eis_zr_raw"].copy(),
                                    z_imag=st.session_state["eis_zi_raw"].copy())
                    n0 = len(ds.frequency)
                    if rm_ind:  ds = DataPreprocessor.remove_inductive_artifacts(ds, verbose=False)
                    if crop:    ds = DataPreprocessor.crop_frequencies(ds, f_min_in, f_max_in, verbose=False)
                    if rm_jmp:  ds = DataPreprocessor.remove_z_jumps(ds, float(jmp_thr), verbose=False)
                    if drop_mains: ds = DataPreprocessor.drop_specific_frequency(ds, float(mains_freq), float(mains_tol), verbose=False)
                    st.session_state.update({"eis_fr":ds.frequency,"eis_zr":ds.z_real,
                                             "eis_zi":ds.z_imag,"preprocessed":True})
                    if "fit" in st.session_state: del st.session_state["fit"]
                    n1 = len(ds.frequency)
                    if n0-n1>0: st.success(f"✅ {n0} → {n1} points ({n0-n1} removed)")
                    else:       st.info(f"✅ No points removed ({n1} points)")
                except Exception as e:
                    st.error(f"Error: {e}")
        with cb:
            if st.button("↩ Reset to Raw"):
                st.session_state.update({
                    "eis_fr":st.session_state["eis_fr_raw"].copy(),
                    "eis_zr":st.session_state["eis_zr_raw"].copy(),
                    "eis_zi":st.session_state["eis_zi_raw"].copy(),
                    "preprocessed":False,
                })
                if "fit" in st.session_state: del st.session_state["fit"]
                st.info("Reset to raw data.")

    if "eis_fr" in st.session_state:
        st.divider()
        n_cur = len(st.session_state["eis_fr"])
        label = "(preprocessed ✨)" if st.session_state.get("preprocessed") else "(raw)"
        st.caption(f"Working with {n_cur} points {label}")

        ck_col, fit_col = st.columns([1,2])
        with ck_col:
            if st.button("🔍 K-K Validation"):
                try:
                    from eisforge.parsers.base_parser import EISDataset
                    from eisforge.core.validators import KramersKronigValidator
                    ds=EISDataset(frequency=st.session_state["eis_fr"],
                                  z_real=st.session_state["eis_zr"],
                                  z_imag=st.session_state["eis_zi"])
                    kk=KramersKronigValidator().validate(ds)
                    (st.success if kk.passed else st.warning)(f"K-K: {'PASSED' if kk.passed else 'FAILED'} | {kk.summary()}")
                except Exception as e:
                    st.error(f"K-K error: {e}")

        with fit_col:
            if st.button("▶ Run CNLS Fit"):
                with st.spinner("Fitting..."):
                    try:
                        from eisforge.parsers.base_parser import EISDataset
                        from eisforge.core.fitter import CNLSFitter
                        p0 = [float(x.strip()) for x in p0s.split(",")]
                        ds = EISDataset(frequency=st.session_state["eis_fr"],
                                        z_real=st.session_state["eis_zr"],
                                        z_imag=st.session_state["eis_zi"])
                        bounds = smart_bounds(circ, p0) if use_bounds else None
                        fit = CNLSFitter(circ, p0, bounds=bounds, remove_outliers=False).fit(ds)
                        st.session_state["fit"] = fit
                        msg = f"χ² = {fit.chi_squared:.4f}"
                        (st.success if fit.converged else st.warning)(
                            f"{'✅ Fit converged' if fit.converged else '⚠ Did not fully converge'} | {msg}"
                        )
                        # Show R0 hint for iR compensation
                        if "R0" in fit.parameters and not np.isnan(fit.parameters["R0"]):
                            r0_val = fit.parameters["R0"]
                            st.info(f"💡 R0 = {r0_val:.3f} Ω — use this as R_s in the sidebar for iR compensation")
                    except Exception as e:
                        st.error(f"Fit error: {e}")

        if "fit" in st.session_state:
            fit = st.session_state["fit"]
            rows=[]
            for name,value in fit.parameters.items():
                err=fit.parameter_errors.get(name,float("nan"))
                val_str=f"{value:.4e}" if not np.isnan(value) else "NaN"
                err_str=f"{err:.2e}" if not np.isnan(err) else "—"
                rel_str=f"{abs(err/value)*100:.2f}" if not np.isnan(value) and value!=0 and not np.isnan(err) else "N/A"
                rows.append({"Parameter":name,"Value":val_str,"±Error":err_str,"Rel. Error (%)":rel_str})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fr=st.session_state["eis_fr"]; zr=st.session_state["eis_zr"]; zi=st.session_state["eis_zi"]
            fig=make_subplots(rows=1,cols=2,subplot_titles=("Nyquist Plot","Bode Plot"))
            fig.add_trace(go.Scatter(x=zr,y=zi,mode="markers",name="Measured",
                                     marker=dict(color="#2563eb",size=8)), row=1,col=1)
            if fit.z_fit is not None:
                fig.add_trace(go.Scatter(x=fit.z_fit.real,y=-fit.z_fit.imag,mode="lines",
                                         name=f"Fit (χ²={fit.chi_squared:.4f})",
                                         line=dict(color="#dc2626",width=2,dash="dash")), row=1,col=1)
            zm=np.sqrt(zr**2+zi**2)
            fig.add_trace(go.Scatter(x=fr,y=zm,mode="markers+lines",name="|Z|",
                                     marker=dict(color="#16a34a",size=5),
                                     line=dict(color="#16a34a",width=1)), row=1,col=2)
            fig.update_xaxes(title_text="Z' (Ω)",row=1,col=1)
            fig.update_yaxes(title_text="-Z'' (Ω)",row=1,col=1)
            fig.update_xaxes(title_text="Frequency (Hz)",type="log",row=1,col=2)
            fig.update_yaxes(title_text="|Z| (Ω)",type="log",row=1,col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════ EIS-GPT ══════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">EIS-GPT — AI Circuit Prediction</p>', unsafe_allow_html=True)
    st.info("**EIS-GPT** predicts the equivalent circuit from your EIS spectrum using a Physics-Informed Transformer.")
    if "eis_fr" in st.session_state:
        if st.button("▶ Predict with EIS-GPT"):
            with st.spinner("EIS-GPT analyzing..."):
                try:
                    import torch
                    from eisforge.ml.eis_gpt.transformer import EISForgeModel
                    m=EISForgeModel(d_model=128,n_heads=8,n_layers=6)
                    fr=torch.tensor(st.session_state["eis_fr"]).float().unsqueeze(0)
                    zr=torch.tensor(st.session_state["eis_zr"]).float().unsqueeze(0)
                    zi=torch.tensor(st.session_state["eis_zi"]).float().unsqueeze(0)
                    res=m.predict(fr,zr,zi); st.session_state["gpt"]=res
                except Exception as e:
                    st.error(f"Error: {e}")
        if "gpt" in st.session_state:
            res=st.session_state["gpt"]
            st.divider()
            c1,c2=st.columns(2)
            c1.metric("Predicted circuit",res["predicted_circuit"])
            c2.metric("Confidence",f"{res['confidence']*100:.1f}%")
            for c in res["top3"]:
                st.progress(c["probability"],text=f"{c['circuit']} — {c['probability']*100:.1f}%")
    else:
        st.warning("Please upload an EIS file in the EIS Analysis tab first.")


# ══════════════════ Correlation ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">EIS — CV — LSV Correlation</p>', unsafe_allow_html=True)
    has_cv=("cv_r" in st.session_state); has_lsv=("lsv_r" in st.session_state); has_fit=("fit" in st.session_state)

    if has_cv and has_lsv:
        st.markdown("#### E_onset Comparison")
        c1,c2,c3=st.columns(3)
        cv_onset=st.session_state["cv_r"].e_onset
        lsv_onset=st.session_state["lsv_r"].e_onset
        ir_note = " (iR-corr.)" if actual_rs>0 else ""
        c1.metric(f"E_onset CV{ir_note}",  f"{cv_onset:.4f} V")
        c2.metric(f"E_onset LSV{ir_note}", f"{lsv_onset:.4f} V")
        diff=abs(cv_onset-lsv_onset)
        c3.metric("Difference",f"{diff*1000:.1f} mV",
                  delta="Good agreement" if diff<0.05 else "Large discrepancy",
                  delta_color="normal" if diff<0.05 else "inverse")
        st.divider()

    if not (has_cv or has_lsv): st.warning("Please analyze a CV or LSV spectrum first.")
    if not has_fit: st.warning("Please fit an EIS spectrum first.")

    if (has_cv or has_lsv) and has_fit:
        cv_res=st.session_state.get("cv_r") or st.session_state.get("lsv_r")
        if st.button("▶ Run EIS-CV Correlation"):
            try:
                from eisforge.analysis.eis_cv_correlator import EISCVCorrelator
                corr=EISCVCorrelator(electrolyte=ekey).correlate(
                    cv_result=cv_res, eis_fit_result=st.session_state["fit"], eis_potential=eis_pot)
                st.divider()
                c1,c2,c3=st.columns(3)
                c1.metric("E_onset",f"{corr.e_onset:.4f} V")
                c2.metric("EIS region",corr.eis_region)
                c3.metric("Consistency",f"{corr.consistency_score:.0%}")
                for w in corr.warnings:      st.warning(w)
                for r in corr.recommendations: st.success(r)

                st.markdown("#### Combined Results Summary")
                summary={"Parameter":[],"CV/LSV":[],"EIS":[]}
                summary["Parameter"].append("E_onset (V)" + (" iR-corr." if actual_rs>0 else ""))
                summary["CV/LSV"].append(f"{cv_res.e_onset:.4f}")
                summary["EIS"].append(f"Measured at {eis_pot:.4f} V")
                if has_cv:
                    summary["Parameter"].append("I_f/I_b")
                    summary["CV/LSV"].append(f"{st.session_state['cv_r'].if_ib_ratio:.3f}")
                    summary["EIS"].append("—")
                summary["Parameter"].append("R_ct (Ω)")
                summary["CV/LSV"].append("—")
                summary["EIS"].append(f"{corr.r_ct:.2f}")
                if actual_rs>0:
                    summary["Parameter"].append("R_s used for iR (Ω)")
                    summary["CV/LSV"].append(f"{actual_rs:.3f}")
                    summary["EIS"].append(f"{actual_rs:.3f}")
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")
