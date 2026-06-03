"""
EISForge — Advanced EIS Analysis with Physics-Informed ML
Author: Hoda Jafari | May 2026
Run: streamlit run app.py
"""

import streamlit as st
from eisforge.analysis.ecsa_calculator import ECSACalculator
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
.val-ok{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:.6rem .8rem;margin:.3rem 0;}
.val-warn{background:#fffbeb;border:1px solid #fde047;border-radius:8px;padding:.6rem .8rem;margin:.3rem 0;}
.val-err{background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:.6rem .8rem;margin:.3rem 0;}
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

# ── E_onset auto-select map ───────────────────────────────────────────────────
_ONSET_METHOD_MAP = {
    "noble_metal":     "tangent",
    "alloy":           "tangent",
    "metal_oxide":     "threshold",
    "carbon_material": "derivative",
}

# ── Try loading carbon_standards (graceful fallback if not yet installed) ─────
try:
    from eisforge.standards.carbon_standards import (
        CarbonValidator, suggest_eec,
        CDL_RANGES, CARBON_SUBTYPE_MAP, RECOMMENDED_EEC,
    )
    _STANDARDS_AVAILABLE = True
except ImportError:
    _STANDARDS_AVAILABLE = False


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


def _show_validation(result):
    """Render a ValidationResult in the correct coloured box."""
    css = {"ok":"val-ok","warning":"val-warn","error":"val-err"}.get(result.severity,"val-warn")
    html = f'<div class="{css}">{result.message}'
    if result.suggested_action:
        html += f'<br><small>💡 {result.suggested_action}</small>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-title">System Settings</p>', unsafe_allow_html=True)
    system_type = st.selectbox("System type",["AOR","Battery","Corrosion","Fuel Cell","Biosensor"])
    catalyst    = st.text_input("Catalyst", placeholder="e.g. Pt/C, carbon_material, PtRu/C")

    # ── Catalyst type ──────────────────────────────────────────────────────
    catalyst_type_ui = st.selectbox(
        "Catalyst type",
        ["Noble Metal (Pt, Pd, Au, Rh)",
         "Alloy (PtRu, PtSn, PdAu, PtCu)",
         "Metal Oxide (NiO, Co₃O₄, MnO₂)",
         "Carbon Material (N-doped C, CNT, Graphene)"],
    )
    _CTYPE_MAP = {
        "Noble Metal (Pt, Pd, Au, Rh)"    : "noble_metal",
        "Alloy (PtRu, PtSn, PdAu, PtCu)" : "alloy",
        "Metal Oxide (NiO, Co₃O₄, MnO₂)" : "metal_oxide",
        "Carbon Material (N-doped C, CNT, Graphene)": "carbon_material",
    }
    catalyst_type = _CTYPE_MAP[catalyst_type_ui]

    # ── Carbon subtype (only for carbon_material) ──────────────────────────
    carbon_subtype_key = "carbon_material"
    if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
        carbon_subtype_ui = st.selectbox(
            "Carbon subtype",
            list(CARBON_SUBTYPE_MAP.keys()),
            help="Used for C_dl validation range",
        )
        carbon_subtype_key = CARBON_SUBTYPE_MAP[carbon_subtype_ui]
    elif catalyst_type == "carbon_material":
        carbon_subtype_ui = st.selectbox(
            "Carbon subtype",
            ["Graphene / rGO","N-doped Graphene","Carbon Nanotubes (CNT / MWCNT)",
             "Activated / Porous Carbon","Carbon Black (Vulcan XC-72)","Other / Unknown"],
        )

    # ── Electrolyte ────────────────────────────────────────────────────────
    electrolyte = st.selectbox("Electrolyte media",["Acidic","Alkaline","NaCl","PBS","Other"])
    ekey = "acidic" if electrolyte == "Acidic" else "alkaline" if electrolyte == "Alkaline" else "acidic"

    if electrolyte == "Acidic":
        elec_compound = st.selectbox("Acid type", ["H₂SO₄", "HClO₄", "HCl", "HNO₃"])
        _COMPOUND_MAP = {"H₂SO₄":"H2SO4","HClO₄":"HClO4","HCl":"HCl","HNO₃":"HNO3"}
        elec_compound_key = _COMPOUND_MAP[elec_compound]
    elif electrolyte == "Alkaline":
        elec_compound = st.selectbox("Base type", ["KOH", "NaOH", "Na₂CO₃", "NH₃"])
        _COMPOUND_MAP = {"KOH":"KOH","NaOH":"NaOH","Na₂CO₃":"Na2CO3","NH₃":"NH3"}
        elec_compound_key = _COMPOUND_MAP[elec_compound]
    else:
        elec_compound = electrolyte
        elec_compound_key = electrolyte

    # ── CHANGE 1 & 2: Alcohol — isopropanol added, carbon_material stays enabled ──
    alcohol = st.selectbox(
        "Alcohol",
        ["ethanol", "methanol", "isopropanol (2-propanol)",
         "ethylene glycol", "glycerol", "N/A"],
        disabled=(system_type != "AOR"),
    )
    # normalise key for analyzers
    _ALCOHOL_KEY_MAP = {
        "ethanol": "ethanol",
        "methanol": "methanol",
        "isopropanol (2-propanol)": "isopropanol",
        "ethylene glycol": "ethylene glycol",
        "glycerol": "glycerol",
        "N/A": "N/A",
    }
    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)

    eis_pot = st.number_input("EIS potential (V)", value=0.5, step=0.01)

    st.divider()
    st.markdown('<p class="section-title">Electrode Parameters</p>', unsafe_allow_html=True)

    # ── CHANGE 3: Geometric area — 4 decimal places ────────────────────────
    area = st.number_input(
        "Geometric area (cm²)",
        value=1.0,
        step=0.0001,
        min_value=0.0001,
        format="%.4f",
    )
    _ecsa_label = "ECSA (cm²_BET)" if catalyst_type == "carbon_material" else "ECSA (cm²_metal)"
    ecsa    = st.number_input(_ecsa_label,             value=0.0, step=0.1,  min_value=0.0)
    loading = st.number_input("Loading (mg/cm²)",      value=0.0, step=0.01, min_value=0.0)

    st.divider()
    st.markdown('<p class="section-title">Experimental Conditions</p>', unsafe_allow_html=True)
    temperature  = st.number_input("Temperature (°C)", value=25, min_value=0, max_value=200)
    current_unit = st.selectbox("Current unit",["mA","A","μA","nA"])
    e_ref_type   = st.selectbox("Reference electrode", list(E_REF_MAP.keys()))
    e_ref_val    = E_REF_MAP[e_ref_type]
    elec_conc    = st.number_input("Electrolyte conc. (M)", value=0.5, step=0.1)

    _PH_MAP = {
        "H2SO4": 0.3, "HClO4": 0.3, "HCl": 0.0, "HNO3": 0.0,
        "KOH": 14.0,  "NaOH": 14.0, "Na2CO3": 11.6, "NH3": 11.6,
    }
    _ph_default = float(_PH_MAP.get(elec_compound_key, 14.0 if ekey == "alkaline" else 0.0))
    ph_value = st.number_input(
        "Solution pH", min_value=0.0, max_value=14.0,
        value=_ph_default, step=0.1,
    )

    sub_conc     = st.number_input("Substrate conc. (M)",   value=1.0, step=0.1)
    unit_factor  = UNIT_MAP.get(current_unit, 1.0)

    if e_ref_type != "RHE":
        _total_offset = e_ref_val + 0.059 * ph_value
        st.info(
            f"RHE conversion: E_RHE = E_meas + {e_ref_val:.3f} V (ref) + "
            f"0.059×pH (pH={ph_value:.1f}) → total offset = {_total_offset:.3f} V"
        )

    st.divider()
    st.markdown('<p class="section-title">⚡ iR Compensation</p>', unsafe_allow_html=True)
    use_ir = st.checkbox(
        "Apply iR compensation", value=False,
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

    # ── CHANGE 4: C_dl validation range (for carbon_material) ─────────────
    if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
        st.divider()
        st.markdown('<p class="section-title">🔬 C_dl Validation Range</p>', unsafe_allow_html=True)
        ref_range = CDL_RANGES.get(carbon_subtype_key, CDL_RANGES["carbon_material"])
        use_custom_cdl = st.checkbox(
            "Use custom C_dl range",
            value=False,
            help=f"Literature default: {ref_range.cdl_min_uF:.0f}–{ref_range.cdl_max_uF:.0f} μF/cm²",
        )
        if use_custom_cdl:
            cdl_user_min = st.number_input("C_dl min (μF/cm²)", value=float(ref_range.cdl_min_uF), step=1.0, min_value=0.1)
            cdl_user_max = st.number_input("C_dl max (μF/cm²)", value=float(ref_range.cdl_max_uF), step=1.0, min_value=1.0)
        else:
            cdl_user_min = None
            cdl_user_max = None
            st.caption(f"Literature range: **{ref_range.cdl_min_uF:.0f}–{ref_range.cdl_max_uF:.0f} μF/cm²** ({ref_range.material})")
    else:
        use_custom_cdl = False
        cdl_user_min = None
        cdl_user_max = None

    st.divider()
    if st.button("📚 Literature Guide"):
        try:
            from eisforge.knowledge.literature_engine import LiteratureEngine
            g = LiteratureEngine().query(
                system_type=system_type, catalyst=catalyst,
                electrolyte=ekey,
                alcohol=alcohol_key if system_type=="AOR" else "",
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


# ── CHANGE 4: E_onset auto-select (computed after sidebar, before tabs) ───────
default_onset_method = _ONSET_METHOD_MAP.get(catalyst_type, "tangent")
_onset_methods = ["tangent", "threshold", "derivative"]
_auto_idx = _onset_methods.index(default_onset_method)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📈 CV Analysis","📉 LSV Analysis","🔬 EIS Analysis",
    "🤖 EIS-GPT","🔗 Correlation","⚗️ K-L Analysis","📏 ECSA Calculator"
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

        # ── CHANGE 4: E_onset method auto + override ───────────────────────
        st.caption(f"🤖 Auto-selected: **{default_onset_method}** (based on catalyst type)")
        om = st.radio(
            "E_onset method (override if needed)",
            _onset_methods,
            index=_auto_idx,
            horizontal=True,
            help=f"Auto: {default_onset_method} for {catalyst_type_ui}. Change if needed.",
        )

    with col2:
        if cv_file:
            try:
                if Path(cv_file.name).suffix.lower()==".idf":
                    pot,cur = load_cv_lsv(cv_file, unit_factor=1.0)
                    st.info("Current auto-converted from A → mA (Autolab)")
                else:
                    pot,cur = load_cv_lsv(cv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot)} points | {cv_file.name}")

                from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo
                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                ana = CVAnalyzer(scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, onset_method=om,
                                 electrolyte=_el, catalyst_type=catalyst_type)
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

        _is_mf = catalyst_type == "carbon_material"

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("E_onset",  f"{r.e_onset:.4f} V")
        c2.metric("I_forward peak", f"{r.i_forward_peak:.4f} mA")
        if _is_mf:
            c3.metric("Net faradaic I", f"{r.net_faradaic_current_mA:.4f} mA")
            c4.metric("C_dl",           f"{r.cdl_mF_cm2:.4f} mF/cm²")
        else:
            c3.metric("I_b",      f"{r.i_backward_peak:.4f} mA")
            c4.metric("I_f/I_b",  f"{r.if_ib_ratio:.3f}" if not np.isnan(r.if_ib_ratio) else "N/A")

        c5,c6,c7 = st.columns(3)
        c5.metric("j_f (geometric)", f"{r.j_forward_peak:.4f} mA/cm²")
        c6.metric("j_b (geometric)", f"{r.j_backward_peak:.4f} mA/cm²")
        _ecsa_unit = "cm²_BET" if _is_mf else "cm²_Pt"
        if r.ecsa>0: c7.metric(f"j_f (ECSA)", f"{r.j_specific_forward:.4f} mA/{_ecsa_unit}")

        st.info(f"**Interpretation:** {r.interpretation}")

        # ── CHANGE 5: C_dl smart validation (carbon only) ─────────────────
        if _is_mf and _STANDARDS_AVAILABLE and hasattr(r, "cdl_mF_cm2"):
            st.markdown("#### 🔬 C_dl Validation")
            val_result = CarbonValidator.validate_cdl(
                cdl_mF_cm2=r.cdl_mF_cm2,
                material_key=carbon_subtype_key,
                user_min_uF=cdl_user_min,
                user_max_uF=cdl_user_max,
            )
            _show_validation(val_result)

            # E_onset validation for carbon
            onset_val = CarbonValidator.validate_onset(r.e_onset)
            _show_validation(onset_val)

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

    # ── Batch CV ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">📊 Batch Analysis — Mean ± SD (n≥3)</p>', unsafe_allow_html=True)
    st.caption("Upload 3 or more CV files from the same experiment for statistical reproducibility.")

    batch_cv_files = st.file_uploader(
        "Upload multiple CV files (n≥3)", type=CV_FORMATS,
        accept_multiple_files=True, key="batch_cv_up"
    )

    if batch_cv_files and len(batch_cv_files) >= 2:
        if st.button(f"▶ Run Batch CV Analysis ({len(batch_cv_files)} files)", type="primary"):
            with st.spinner(f"Analyzing {len(batch_cv_files)} CV files..."):
                try:
                    from eisforge.analysis.batch_analyzer import BatchCVAnalyzer
                    from eisforge.analysis.cv_analyzer import ElectrolyteInfo

                    _el_b = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                    pots_b, curs_b = [], []
                    for f in batch_cv_files:
                        p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)
                        pots_b.append(p); curs_b.append(c)

                    batch_ana = BatchCVAnalyzer(
                        scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                        catalyst_type=catalyst_type, electrolyte=_el_b,
                        onset_method=om, catalyst_loading=loading,
                        r_s_ohms=actual_rs,
                    )
                    br = batch_ana.analyze_arrays(pots_b, curs_b)
                    st.session_state["batch_cv_r"] = br
                    st.success(f"✅ {br.n_valid}/{br.n_files} files analyzed successfully")
                    if br.outlier_indices:
                        st.warning(f"⚠ Outliers detected in files: {br.outlier_indices}")
                except Exception as e:
                    st.error(f"Batch error: {e}")
    elif batch_cv_files:
        st.info("Upload at least 2 files for batch analysis (3 recommended for publications).")

    if "batch_cv_r" in st.session_state:
        br = st.session_state["batch_cv_r"]
        _is_mf = catalyst_type == "carbon_material"
        st.divider()
        n_str = f"n={br.n_valid}"

        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"E_onset ({n_str})", f"{br.e_onset_mean:.4f} V", delta=f"± {br.e_onset_std:.4f}")
        c2.metric(f"I_forward ({n_str})", f"{br.i_fwd_mean:.4f} mA", delta=f"± {br.i_fwd_std:.4f}")
        c3.metric(f"j_forward ({n_str})", f"{br.j_fwd_mean:.4f} mA/cm²", delta=f"± {br.j_fwd_std:.4f}")
        if _is_mf:
            c4.metric(f"C_dl ({n_str})", f"{br.cdl_mean:.4f} mF/cm²", delta=f"± {br.cdl_std:.4f}")
        else:
            c4.metric(f"I_f/I_b ({n_str})", f"{br.if_ib_mean:.3f}", delta=f"± {br.if_ib_std:.3f}")

        if br.potential_common is not None:
            import plotly.graph_objects as go
            fig_b = go.Figure()
            pot_c = br.potential_common
            cur_m = br.current_mean_curve
            cur_s = br.current_std_curve
            fig_b.add_trace(go.Scatter(
                x=np.concatenate([pot_c, pot_c[::-1]]),
                y=np.concatenate([cur_m + cur_s, (cur_m - cur_s)[::-1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(37,99,235,0)"), name=f"± SD band",
            ))
            fig_b.add_trace(go.Scatter(
                x=pot_c, y=cur_m, mode="lines", name=f"Mean CV (n={br.n_valid})",
                line=dict(color="#2563eb", width=2.5),
            ))
            fig_b.add_vline(x=br.e_onset_mean, line_dash="dash", line_color="#d97706",
                            annotation_text=f"E_onset = {br.e_onset_mean:.3f} ± {br.e_onset_std:.3f} V",
                            annotation_font=dict(color="#d97706"))
            fig_b.update_layout(**PLOTLY_LAYOUT,
                                title=f"Averaged CV — {n_str} | {catalyst or 'Catalyst'}",
                                xaxis_title=f"Potential (V vs {e_ref_type})",
                                yaxis_title="Current (mA)")
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("#### Publication-Ready Table")
        st.dataframe(br.to_dataframe(), use_container_width=True, hide_index=True)
        col_md, col_tex = st.columns(2)
        with col_md:
            st.markdown("**Markdown** (for README / GitHub)")
            st.code(br.to_markdown_table(), language="markdown")
        with col_tex:
            st.markdown("**LaTeX** (paste directly into paper)")
            st.code(br.to_latex_table(), language="latex")


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

        # ── CHANGE 4: E_onset method auto + override (LSV) ────────────────
        st.caption(f"🤖 Auto-selected: **{default_onset_method}** (based on catalyst type)")
        om_lsv = st.radio(
            "E_onset method (override if needed)",
            _onset_methods,
            index=_auto_idx,
            horizontal=True,
            key="om_lsv",
            help=f"Auto: {default_onset_method} for {catalyst_type_ui}.",
        )

    with col2:
        if lsv_file:
            try:
                if Path(lsv_file.name).suffix.lower()==".idf":
                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=1.0)
                else:
                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=unit_factor)

                st.success(f"✅ {len(pot_lsv)} points | {lsv_file.name}")

                from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo
                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                la = LSVAnalyzer(scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, electrolyte=_el,
                                 catalyst_type=catalyst_type,
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
        _sa_unit = "cm²_BET" if catalyst_type == "carbon_material" else "cm²_Pt"
        if ecsa>0:    st.metric("Specific activity", f"{r.specific_activity:.4f} mA/{_sa_unit}")

        # ── CHANGE 5: Tafel validation for carbon ─────────────────────────
        if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
            st.markdown("#### 🔬 Tafel Validation")
            tafel_val = CarbonValidator.validate_tafel(r.tafel_slope, electrolyte=ekey)
            _show_validation(tafel_val)

        st.info(f"**Mechanism:** {r.mechanism_interpretation}")
        st.success(f"**Performance:** {r.performance_rating}")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1,cols=2,subplot_titles=("LSV Curve","Tafel Plot"))

        j_lsv = st.session_state["lsv_cur"]/area
        if actual_rs>0:
            from eisforge.analysis.lsv_analyzer import LSVAnalyzer
            cur_ma_lsv = st.session_state["lsv_cur"]*unit_factor
            p_lsv = LSVAnalyzer.apply_ir_compensation(
                st.session_state["lsv_pot"]+e_ref_val, cur_ma_lsv, actual_rs
            )
        else:
            p_lsv = st.session_state["lsv_pot"]+e_ref_val

        fig.add_trace(go.Scatter(x=p_lsv, y=j_lsv, mode="lines",
                                 name="LSV"+(" (iR-corr.)" if actual_rs>0 else ""),
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

    # ── Batch LSV ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">📊 Batch Analysis — Mean ± SD (n≥3)</p>', unsafe_allow_html=True)
    st.caption("Upload 3 or more LSV files for statistical reproducibility.")

    batch_lsv_files = st.file_uploader(
        "Upload multiple LSV files (n≥3)", type=CV_FORMATS,
        accept_multiple_files=True, key="batch_lsv_up"
    )

    if batch_lsv_files and len(batch_lsv_files) >= 2:
        if st.button(f"▶ Run Batch LSV Analysis ({len(batch_lsv_files)} files)", type="primary"):
            with st.spinner(f"Analyzing {len(batch_lsv_files)} LSV files..."):
                try:
                    from eisforge.analysis.batch_analyzer import BatchLSVAnalyzer
                    from eisforge.analysis.lsv_analyzer import ElectrolyteInfo

                    _el_b = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                    pots_b, curs_b = [], []
                    for f in batch_lsv_files:
                        p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)
                        pots_b.append(p); curs_b.append(c)

                    batch_lsv = BatchLSVAnalyzer(
                        scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                        catalyst_type=catalyst_type, electrolyte=_el_b,
                        catalyst_loading=loading, e_ref_vs_rhe=e_ref_val,
                        tafel_current_range=(tj_min, tj_max),
                        r_s_ohms=actual_rs,
                    )
                    blr = batch_lsv.analyze_arrays(pots_b, curs_b)
                    st.session_state["batch_lsv_r"] = blr
                    st.success(f"✅ {blr.n_valid}/{blr.n_files} files analyzed successfully")
                    if blr.outlier_indices:
                        st.warning(f"⚠ Outliers detected in files: {blr.outlier_indices}")
                except Exception as e:
                    st.error(f"Batch error: {e}")
    elif batch_lsv_files:
        st.info("Upload at least 2 files for batch analysis (3 recommended).")

    if "batch_lsv_r" in st.session_state:
        import math
        blr = st.session_state["batch_lsv_r"]
        _is_mf = catalyst_type == "carbon_material"
        st.divider()
        n_str = f"n={blr.n_valid}"

        c1,c2,c3 = st.columns(3)
        c1.metric(f"E_onset ({n_str})", f"{blr.e_onset_mean:.4f} V", delta=f"± {blr.e_onset_std:.4f}")
        _tafel_note = " (normal)" if _is_mf else ""
        c2.metric(f"Tafel ({n_str})", f"{blr.tafel_mean:.1f} mV/dec{_tafel_note}", delta=f"± {blr.tafel_std:.1f}")
        c3.metric(f"j₀ ({n_str})", f"{blr.j0_mean:.3e} mA/cm²", delta=f"± {blr.j0_std:.3e}")

        c4,c5,c6 = st.columns(3)
        c4.metric(f"η@10 ({n_str})", f"{blr.eta10_mean*1000:.1f} mV" if not math.isnan(blr.eta10_mean) else "N/A",
                  delta=f"± {blr.eta10_std*1000:.1f} mV" if not math.isnan(blr.eta10_std) else None)
        c5.metric(f"η@50 ({n_str})", f"{blr.eta50_mean*1000:.1f} mV" if not math.isnan(blr.eta50_mean) else "N/A",
                  delta=f"± {blr.eta50_std*1000:.1f} mV" if not math.isnan(blr.eta50_std) else None)
        c6.metric(f"η@100 ({n_str})", f"{blr.eta100_mean*1000:.1f} mV" if not math.isnan(blr.eta100_mean) else "N/A",
                  delta=f"± {blr.eta100_std*1000:.1f} mV" if not math.isnan(blr.eta100_std) else None)

        if blr.potential_common is not None:
            import plotly.graph_objects as go
            fig_blsv = go.Figure()
            pot_c = blr.potential_common
            j_m   = blr.j_mean_curve
            j_s   = blr.j_std_curve
            fig_blsv.add_trace(go.Scatter(
                x=np.concatenate([pot_c, pot_c[::-1]]),
                y=np.concatenate([j_m + j_s, (j_m - j_s)[::-1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(37,99,235,0)"), name="± SD band",
            ))
            fig_blsv.add_trace(go.Scatter(
                x=pot_c, y=j_m, mode="lines",
                name=f"Mean LSV ({n_str})",
                line=dict(color="#2563eb", width=2.5),
            ))
            fig_blsv.add_vline(x=blr.e_onset_mean, line_dash="dash", line_color="#d97706",
                               annotation_text=f"E_onset = {blr.e_onset_mean:.3f} ± {blr.e_onset_std:.3f} V",
                               annotation_font=dict(color="#d97706"))
            fig_blsv.update_layout(**PLOTLY_LAYOUT,
                                   title=f"Averaged LSV — {n_str} | {catalyst or 'Catalyst'}",
                                   xaxis_title=f"Potential (V vs {e_ref_type})",
                                   yaxis_title="j (mA/cm²)")
            st.plotly_chart(fig_blsv, use_container_width=True)

        st.markdown("#### Publication-Ready Table")
        st.dataframe(blr.to_dataframe(), use_container_width=True, hide_index=True)
        col_md, col_tex = st.columns(2)
        with col_md:
            st.markdown("**Markdown**")
            st.code(blr.to_markdown_table(), language="markdown")
        with col_tex:
            st.markdown("**LaTeX**")
            st.code(blr.to_latex_table(), language="latex")


# ══════════════════ EIS ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Electrochemical Impedance Spectroscopy</p>', unsafe_allow_html=True)

    col1,col2 = st.columns([1,1])
    with col1:
        eis_file = st.file_uploader("Upload EIS file", type=EIS_FORMATS, key="eis_up")

        # ── CHANGE 5: EEC suggestion from carbon_standards ────────────────
        if _STANDARDS_AVAILABLE:
            eec_suggestion = suggest_eec(
                catalyst_type=catalyst_type,
                electrolyte=ekey,
                user_circuit=None,
            )
            _suggested_circuit = eec_suggestion["circuit"]
            _suggested_p0      = ", ".join(f"{v:.3e}" for v in eec_suggestion["p0"]) if eec_suggestion["p0"] else ""
            st.info(f"🤖 Suggested EEC: `{_suggested_circuit}` — " + eec_suggestion["note"])
        else:
            _suggested_circuit = "R0-p(R1,CPE1)"
            _suggested_p0      = "30, 31000, 2e-7, 0.78"

        # Literature guide override
        if "lit" in st.session_state and st.session_state["lit"].system_found:
            g = st.session_state["lit"]
            _suggested_circuit = g.recommended_circuit
            _suggested_p0 = ", ".join(f"{v:.3e}" for v in g.initial_guess.values())

        circ = st.text_input("Equivalent circuit (edit or accept suggestion)", value=_suggested_circuit)
        p0s  = st.text_input("Initial guess (comma-separated)", value=_suggested_p0 if _suggested_p0 else "30, 31000, 2e-7, 0.78")
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
                        if "R0" in fit.parameters and not np.isnan(fit.parameters["R0"]):
                            r0_val = fit.parameters["R0"]
                            st.info(f"💡 R0 = {r0_val:.3f} Ω — use this as R_s in the sidebar for iR compensation")

                            # R_ct validation for carbon
                            if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
                                r1_key = "R1" if "R1" in fit.parameters else None
                                if r1_key:
                                    rct_val = fit.parameters[r1_key]
                                    rct_result = CarbonValidator.validate_rct(rct_val, electrolyte=ekey)
                                    _show_validation(rct_result)
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
                for w in corr.warnings:       st.warning(w)
                for r in corr.recommendations: st.success(r)

                st.markdown("#### Combined Results Summary")
                summary={"Parameter":[],"CV/LSV":[],"EIS":[]}
                summary["Parameter"].append("E_onset (V)" + (" iR-corr." if actual_rs>0 else ""))
                summary["CV/LSV"].append(f"{cv_res.e_onset:.4f}")
                summary["EIS"].append(f"Measured at {eis_pot:.4f} V")
                if has_cv and catalyst_type != "carbon_material":
                    _ratio = st.session_state["cv_r"].if_ib_ratio
                    summary["Parameter"].append("I_f/I_b")
                    summary["CV/LSV"].append(f"{_ratio:.3f}" if not np.isnan(_ratio) else "N/A")
                    summary["EIS"].append("—")
                elif has_cv and catalyst_type == "carbon_material":
                    summary["Parameter"].append("C_dl (mF/cm²)")
                    summary["CV/LSV"].append(f"{st.session_state['cv_r'].cdl_mF_cm2:.4f}")
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


# ══════════════════ K-L Analysis ═════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">Koutecky-Levich Analysis — RDE</p>', unsafe_allow_html=True)
    st.info(
        "Upload LSV files measured at **different rotation speeds** (400, 900, 1600, 2500 rpm). "
        "The K-L plot separates **kinetic current (j_k)** from mass-transport limitation "
        "and determines the **number of electrons (n)** transferred per alcohol molecule."
    )

    kl_col1, kl_col2 = st.columns([1, 1])

    with kl_col1:
        st.markdown('<p class="section-title">Alcohol & Electrolyte</p>', unsafe_allow_html=True)
        kl_alcohol   = st.selectbox(
            "Alcohol",
            ["ethanol","methanol","isopropanol","2-propanol","ethylene_glycol","glycerol"],
            key="kl_alc",
        )
        kl_conc      = st.number_input("Alcohol concentration (M)", value=1.0, step=0.1, min_value=0.01, key="kl_conc")
        kl_temp      = st.number_input("Temperature (°C)", value=25, min_value=0, max_value=100, key="kl_temp")
        kl_pot       = st.number_input("Analysis potential (V vs RHE)", value=0.5, step=0.01, key="kl_pot")
        kl_area      = st.number_input(
            "Electrode area (cm²)", value=0.1963, step=0.0001,
            min_value=0.0001, format="%.4f", key="kl_area",
        )
        st.caption("Standard RDE electrode: 0.1963 cm² (5 mm diameter)")

    with kl_col2:
        st.markdown('<p class="section-title">Upload RDE Files</p>', unsafe_allow_html=True)
        st.caption("Upload one LSV file per rotation speed — name them clearly (e.g. lsv_400rpm.csv)")

        kl_speeds  = []
        kl_files   = []
        for rpm_val in [400, 900, 1600, 2500]:
            f = st.file_uploader(f"{rpm_val} rpm", type=CV_FORMATS, key=f"kl_{rpm_val}")
            if f:
                kl_files.append(f)
                kl_speeds.append(float(rpm_val))

        custom_rpm  = st.number_input("Custom speed (rpm, optional)", value=0, step=100, key="kl_custom_rpm")
        custom_file = st.file_uploader("Custom speed file", type=CV_FORMATS, key="kl_custom_f")
        if custom_file and custom_rpm > 0:
            kl_files.append(custom_file)
            kl_speeds.append(float(custom_rpm))

    st.divider()

    if len(kl_files) >= 3:
        st.success(f"✅ {len(kl_files)} rotation speeds loaded: {[int(s) for s in kl_speeds]} rpm")

        if st.button("▶ Run Koutecky-Levich Analysis", type="primary"):
            with st.spinner("Running K-L analysis..."):
                try:
                    from eisforge.analysis.koutecky_levich import KLAnalyzer

                    pots_kl, curs_kl = [], []
                    for f in kl_files:
                        p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)
                        pots_kl.append(p)
                        curs_kl.append(c)

                    kl_ana = KLAnalyzer(
                        alcohol=kl_alcohol,
                        electrolyte=elec_compound_key,
                        concentration_M=kl_conc,
                        temperature_C=float(kl_temp),
                        catalyst_type=catalyst_type,
                    )
                    kl_result = kl_ana.analyze(
                        rotation_speeds_rpm=kl_speeds,
                        potentials=pots_kl,
                        currents=curs_kl,
                        electrode_area=kl_area,
                        analysis_potential=kl_pot,
                    )
                    st.session_state["kl_result"] = kl_result
                    st.success("✅ K-L analysis complete!")
                except Exception as e:
                    st.error(f"K-L error: {e}")
    elif kl_files:
        st.warning(f"Upload at least 3 rotation speeds. Currently: {len(kl_files)}")
    else:
        st.info("Upload LSV files at 3+ rotation speeds to begin.")

    if "kl_result" in st.session_state:
        kl_r = st.session_state["kl_result"]
        best = kl_r.best_result
        if best:
            st.divider()
            st.markdown("### Results")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("n electrons",    f"{best.n_electrons:.2f}")
            m2.metric("j_kinetic",      f"{best.j_kinetic:.4f} mA/cm²")
            m3.metric("K-L R²",         f"{best.r_squared:.4f}")
            m4.metric("Mass-transport", f"{best.diffusion_controlled_fraction:.0%}")

            st.info(f"**Interpretation:** {best.interpretation}")

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig_kl = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Koutecky-Levich Plot", "j_kinetic vs Potential"))

            inv_sqw = [1.0 / np.sqrt(rpm * 2 * np.pi / 60.0) for rpm in best.rotation_speeds_rpm]
            inv_j   = [1.0 / j if abs(j) > 1e-10 else None for j in best.j_measured]
            valid_pts = [(x, y) for x, y in zip(inv_sqw, inv_j) if y is not None]

            if valid_pts:
                x_kl = [p[0] for p in valid_pts]
                y_kl = [p[1] for p in valid_pts]
                fig_kl.add_trace(go.Scatter(
                    x=x_kl, y=y_kl, mode="markers", name="1/j vs 1/√ω",
                    marker=dict(color="#2563eb", size=10, symbol="circle"),
                ), row=1, col=1)
                x_fit = np.linspace(min(x_kl)*0.9, max(x_kl)*1.1, 50)
                from scipy.stats import linregress
                if len(x_kl) >= 2:
                    sl, ic, _, _, _ = linregress(x_kl, y_kl)
                    fig_kl.add_trace(go.Scatter(
                        x=x_fit, y=sl*x_fit + ic, mode="lines",
                        name=f"K-L fit (R²={best.r_squared:.4f})",
                        line=dict(color="#dc2626", dash="dash", width=2),
                    ), row=1, col=1)
                    if abs(best.intercept) > 1e-12:
                        fig_kl.add_hline(y=best.intercept, line_dash="dot",
                                         line_color="#d97706",
                                         annotation_text=f"1/j_k = {best.intercept:.4f}",
                                         annotation_font=dict(color="#d97706"),
                                         row=1, col=1)

            if len(kl_r.potentials_V) > 1:
                fig_kl.add_trace(go.Scatter(
                    x=kl_r.potentials_V, y=kl_r.j_kinetic_arr,
                    mode="lines+markers", name="j_kinetic",
                    line=dict(color="#7c3aed", width=2),
                    marker=dict(size=8),
                ), row=1, col=2)

            fig_kl.update_xaxes(title_text="1/√ω (s^0.5/rad^0.5)", row=1, col=1)
            fig_kl.update_yaxes(title_text="1/j (cm²/mA)", row=1, col=1)
            fig_kl.update_xaxes(title_text="Potential (V vs RHE)", row=1, col=2)
            fig_kl.update_yaxes(title_text="j_kinetic (mA/cm²)", row=1, col=2)
            fig_kl.update_layout(**PLOTLY_LAYOUT, height=450,
                                 title=f"K-L Analysis — {kl_alcohol} | {catalyst or 'Catalyst'}")
            st.plotly_chart(fig_kl, use_container_width=True)

            st.markdown("#### Publication-Ready Table")
            st.dataframe(kl_r.to_dataframe(), use_container_width=True, hide_index=True)
            col_md, col_tex = st.columns(2)
            with col_md:
                st.markdown("**Markdown**")
                st.code(kl_r.to_markdown_table(), language="markdown")
            with col_tex:
                st.markdown("**LaTeX**")
                st.code(kl_r.to_latex_table(), language="latex")


# ══════════════════════════════════════════
# TAB 7 — ECSA CALCULATOR
# ══════════════════════════════════════════
with tab7:
    st.header("📏 Automated ECSA Calculator")
    st.markdown("Calculate ECSA using **H-UPD**, **CO Stripping**, or **Cdl**.")

    c1, c2 = st.columns(2)
    with c1:
        ecsa_method = st.selectbox(
            "ECSA Method",
            [
                "Method A: H-UPD (Pt/Pd)",
                "Method B: CO Stripping (PtRu/Pd)",
                "Method C: Cdl (Carbon/Metal-Free)"
            ],
            key="ecsa_method_sel"
        )
        ecsa_loading = st.number_input(
            "Catalyst Loading (mg)",
            value=0.1, min_value=0.001, step=0.01,
            key="ecsa_loading"
        )
    with c2:
        ecsa_scan_rate = st.number_input(
            "Scan Rate (mV/s)",
            value=50.0, step=10.0,
            key="ecsa_sr"
        )
        ecsa_q_ref_sel = st.selectbox(
            "Reference Charge",
            [
                "Pt — H-UPD: 210 | CO: 420 uC/cm2",
                "Pd — H-UPD: 212 | CO: 424 uC/cm2"
            ],
            key="ecsa_qref"
        )

    q_hupd = ECSACalculator.Q_H_PD if "Pd" in ecsa_q_ref_sel else ECSACalculator.Q_H_PT
    q_co   = ECSACalculator.Q_CO_PD if "Pd" in ecsa_q_ref_sel else ECSACalculator.Q_CO_PT

    st.divider()

    if not ecsa_method.startswith("Method C"):
        ecsa_file = st.file_uploader(
            "Upload CV file (.csv, .txt, .idf, .dta)",
            type=CV_FORMATS,
            key="ecsa_single_up"
        )

        if ecsa_file:
            try:
                pot_e, cur_e = load_cv_lsv(ecsa_file)
                v_lo = float(pot_e.min())
                v_hi = float(pot_e.max())
                default_win = (0.05, 0.40) if ecsa_method.startswith("Method A") else (0.60, 0.90)
                win_lo = max(v_lo, default_win[0])
                win_hi = min(v_hi, default_win[1])
                v_min_e, v_max_e = st.slider(
                    "Integration Window (V vs RHE)",
                    float(v_lo), float(v_hi),
                    (float(win_lo), float(win_hi)),
                    step=0.01,
                    key="ecsa_window"
                )
            except Exception as ex:
                st.error(f"File read error: {ex}")
                ecsa_file = None

        if ecsa_file and st.button("Calculate ECSA", key="ecsa_calc_ab"):
            try:
                sr_vs = ecsa_scan_rate / 1000.0
                if ecsa_method.startswith("Method A"):
                    res_e = ECSACalculator.method_a_hupd(
                        pot_e, cur_e, sr_vs, ecsa_loading,
                        v_range=(v_min_e, v_max_e),
                        q_ref=q_hupd
                    )
                else:
                    res_e = ECSACalculator.method_b_co(
                        pot_e, cur_e, sr_vs, ecsa_loading,
                        v_range=(v_min_e, v_max_e),
                        q_ref=q_co
                    )

                import plotly.graph_objects as go_e
                fig_e = go_e.Figure()
                fig_e.add_trace(go_e.Scatter(
                    x=pot_e, y=cur_e,
                    mode="lines", name="CV Data"
                ))
                fig_e.add_vrect(
                    x0=v_min_e, x1=v_max_e,
                    fillcolor="green", opacity=0.15,
                    line_width=0,
                    annotation_text="Integration Region"
                )
                fig_e.update_layout(
                    xaxis_title="Potential (V vs RHE)",
                    yaxis_title="Current (A)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_e, use_container_width=True)

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Charge (uC)",      f"{res_e['charge_uC']:.2f}")
                col_r2.metric("ECSA (cm2/mg)",    f"{res_e['ecsa_cm2_mg']:.3f}")
                col_r3.metric("Q_ref (uC/cm2)",   f"{res_e['q_ref_used']:.0f}")

                import pandas as pd_exp
                st.download_button(
                    "Export Results (CSV)",
                    pd_exp.DataFrame([res_e]).to_csv(index=False),
                    "ecsa_results.csv",
                    "text/csv"
                )

            except Exception as ex:
                st.error(f"Error: {ex}")

    else:
        ecsa_files_c = st.file_uploader(
            "Upload CVs at different scan rates (min 3 files)",
            type=CV_FORMATS,
            accept_multiple_files=True,
            key="ecsa_multi_up"
        )
        ecsa_sr_text = st.text_input(
            "Scan Rates (mV/s) comma-separated — same order as files",
            "10,20,40,60,80,100",
            key="ecsa_sr_list"
        )
        ecsa_vmid = st.number_input(
            "Non-Faradaic Midpoint (V)",
            value=0.25, step=0.01,
            key="ecsa_vmid"
        )
        ecsa_cs = st.number_input(
            "Specific Capacitance Cs (mF/cm2)",
            value=0.035, step=0.005,
            help="0.035 Carbon | 0.060 RuO2",
            key="ecsa_cs"
        )

        if ecsa_files_c and len(ecsa_files_c) >= 3:
            sr_list_c = [float(x.strip()) for x in ecsa_sr_text.split(",")]

            import pandas as pd_match
            st.dataframe(
                pd_match.DataFrame({
                    "File": [f.name for f in ecsa_files_c],
                    "Scan Rate (mV/s)": sr_list_c[:len(ecsa_files_c)]
                }),
                use_container_width=True
            )

            if len(sr_list_c) != len(ecsa_files_c):
                st.error(
                    f"Scan rates ({len(sr_list_c)}) must equal "
                    f"number of files ({len(ecsa_files_c)})"
                )
            elif st.button("Calculate Cdl and ECSA", key="ecsa_calc_c"):
                try:
                    pots_c, curs_c = [], []
                    for f_c in ecsa_files_c:
                        p_c, i_c = load_cv_lsv(f_c)
                        pots_c.append(p_c)
                        curs_c.append(i_c)

                    sr_vs_list = [s / 1000.0 for s in sr_list_c]
                    res_c = ECSACalculator.method_c_cdl(
                        pots_c, curs_c, sr_vs_list,
                        v_range=(ecsa_vmid - 0.05, ecsa_vmid + 0.05)
                    )

                    ecsa_from_cdl = (res_c["cdl_value"] / ecsa_cs) if ecsa_cs > 0 else 0.0
                    spec_ecsa_cdl = ecsa_from_cdl / ecsa_loading if ecsa_loading > 0 else 0.0

                    import plotly.graph_objects as go_c
                    fig_c = go_c.Figure()
                    fig_c.add_trace(go_c.Scatter(
                        x=res_c["scan_rates_V_s"],
                        y=res_c["delta_j_list"],
                        mode="markers",
                        marker=dict(size=10),
                        name="Delta-j Data"
                    ))
                    y_fit = [
                        res_c["fit_slope"] * v + res_c["fit_intercept"]
                        for v in res_c["scan_rates_V_s"]
                    ]
                    fig_c.add_trace(go_c.Scatter(
                        x=res_c["scan_rates_V_s"],
                        y=y_fit,
                        mode="lines",
                        name=f"Linear Fit R2={res_c['r_squared']:.4f}"
                    ))
                    fig_c.update_layout(
                        xaxis_title="Scan Rate (V/s)",
                        yaxis_title="Delta-j (A/cm2)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_c, use_container_width=True)

                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Cdl (mF/cm2)",          f"{res_c['cdl_value']*1000:.4f}")
                    col_c2.metric("ECSA (cm2)",             f"{ecsa_from_cdl:.3f}")
                    col_c3.metric("Specific ECSA (cm2/mg)", f"{spec_ecsa_cdl:.3f}")

                    import pandas as pd_exp2
                    st.download_button(
                        "Export Results (CSV)",
                        pd_exp2.DataFrame([{
                            "Method":                "Cdl",
                            "Cdl_mF_cm2":            res_c["cdl_value"] * 1000,
                            "R2":                    res_c["r_squared"],
                            "ECSA_cm2":              ecsa_from_cdl,
                            "Specific_ECSA_cm2_mg":  spec_ecsa_cdl,
                            "Cs_used_mF_cm2":        ecsa_cs,
                            "Loading_mg":            ecsa_loading,
                        }]).to_csv(index=False),
                        "ecsa_cdl_results.csv",
                        "text/csv"
                    )

                except Exception as ex:
                    st.error(f"Error: {ex}")
        else:
            st.info("Please upload at least 3 CV files.")
