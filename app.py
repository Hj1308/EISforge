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
import math
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
    '<a href="https://github.com/Hj1308/EISforge-">GitHub</a></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center">'
    '<span class="fmt">.idf Autolab</span> <span class="fmt">.dta Gamry</span> '
    '<span class="fmt">.mpt BioLogic</span> <span class="fmt">.csv</span> '
    '<span class="fmt">.txt</span></p>',
    unsafe_allow_html=True,
)
st.divider()

EIS_FORMATS = ["idf", "dta", "mpt", "mpr", "csv", "txt"]
CV_FORMATS = ["idf", "csv", "txt", "dta"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(family="Inter", color="#1e293b"),
    margin=dict(l=60, r=20, t=50, b=50),
)

E_REF_MAP = {
    "RHE": 0.000, "Ag/AgCl (sat.)": 0.197, "Ag/AgCl (3M KCl)": 0.210,
    "SCE": 0.241, "Hg/HgO (1M KOH)": 0.098, "NHE/SHE": 0.000,
}
UNIT_MAP = {"A": 1000.0, "mA": 1.0, "μA": 1e-3, "nA": 1e-6}

# ── E_onset auto-select map ───────────────────────────────────────────────────
_ONSET_METHOD_MAP = {
    "noble_metal": "tangent",
    "alloy": "tangent",
    "metal_oxide": "threshold",
    "carbon_material": "derivative",
}

# ── Try loading carbon_standards (graceful fallback if not yet installed) ─────
try:
    from eisforge.standards.carbon_standards import (
        CarbonValidator, suggest_eec, CDL_RANGES,
        CARBON_SUBTYPE_MAP,
    )
    _STANDARDS_AVAILABLE = True
except ImportError:
    _STANDARDS_AVAILABLE = False


def smart_bounds(circuit_str, p0):
    n = len(p0)
    lower = [0.0] * n
    upper = [np.inf] * n
    tokens = re.findall(r"[A-Za-z]+\d+", circuit_str)
    p_idx = 0
    for tok in tokens:
        if p_idx >= n:
            break
        if tok.upper().startswith("CPE"):
            p_idx += 1
            if p_idx < n:
                upper[p_idx] = 1.0
            p_idx += 1
        elif tok.upper().startswith(("WO", "WS", "G")):
            p_idx += 2
        else:
            p_idx += 1
    return (lower, upper)


def read_csv_safe(path):
    for enc in ["latin-1", "cp1252", "utf-8", "utf-16"]:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python",
                               comment="#", skip_blank_lines=True)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1", errors="replace",
                       sep=None, engine="python", comment="#")


def save_upload(f):
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
            return (df["freq/Hz"].to_numpy(), df["Re(Z)/Ohm"].to_numpy(),
                    -df["-Im(Z)/Ohm"].to_numpy(), {"source": "BioLogic"})
        else:
            df = read_csv_safe(tmp)
            c = df.columns.tolist()
            fr, zr, zi = df[c[0]].to_numpy(float), df[c[1]].to_numpy(float), df[c[2]].to_numpy(float)
            if zi.mean() < 0:
                zi = -zi
            return fr, zr, zi, {}
    finally:
        os.unlink(tmp)


def _parse_ivium_current_unit(text: str) -> float:
    """Parse 'Current Range=' from Ivium metadata -> multiplier to mA."""
    m = re.search(r"Current Range\s*=\s*([\d.]+)\s*([A-Za-zµ]+)", text)
    if not m:
        return 1.0
    _, unit = m.groups()
    unit = unit.lower().strip()
    if unit == "a":
        return 1000.0
    elif unit == "ma":
        return 1.0
    elif unit in ("ua", "µa"):
        return 0.001
    return 1.0


def _load_ivium_cv(path: str, cycle_idx: int = -1):
    """Load Ivium .idf CV. Returns (E, I_mA, meta).
    Auto-detects current unit; supports cycle selection (-1 = last complete)."""
    text = open(path, "rb").read().decode("latin-1")
    meta = {}
    for key in ["Scanrate", "N scans", "E start", "Vertex 1", "Vertex 2"]:
        mm = re.search(re.escape(key) + r"=([^\r\n]+)", text)
        if mm:
            try:
                meta[key] = float(mm.group(1).strip())
            except Exception:
                meta[key] = mm.group(1).strip()

    unit_mult = _parse_ivium_current_unit(text)
    meta["_unit_mult"] = unit_mult
    meta["_unit_label"] = ("A" if unit_mult == 1000.0 else
                           "µA" if unit_mult == 0.001 else "mA")

    rows = re.findall(
        r"(-?\d+\.?\d*E[+-]\d+)\s+(-?\d+\.?\d*E[+-]\d+)\s+(-?\d+\.?\d*E[+-]\d+)", text)
    if not rows:
        raise ValueError("No numeric data found in .idf file")
    arr = np.array(rows, dtype=float)
    E_all = arr[:, 0]
    I_all = arr[:, 1] * unit_mult   # -> mA

    sign_ch = np.diff(np.sign(np.diff(E_all)))
    vertices = np.where(sign_ch != 0)[0] + 1
    if len(vertices) < 2:
        meta["_n_cycles"] = 1
        meta["_cycle_used"] = 1
        return E_all, I_all, meta

    # Detect scan direction to split at correct turning points
    dE = np.diff(E_all[:min(50, len(E_all))])
    first_dir = "up" if np.median(dE) > 0 else "down"
    if first_dir == "up":
        cycle_boundaries = vertices[1::2]
    else:
        cycle_boundaries = vertices[0::2]
    if len(cycle_boundaries) == 0:
        cycle_boundaries = vertices[:1]
    cycle_starts = [0] + list(cycle_boundaries + 1)
    cycle_ends = list(cycle_boundaries + 1) + [len(E_all)]

    n_cycles = len(cycle_starts) - 1
    meta["_n_cycles"] = n_cycles
    if cycle_idx == -1:
        chosen = max(0, n_cycles - 2) if n_cycles >= 2 else 0
    else:
        chosen = max(0, min(cycle_idx, n_cycles - 1))
    meta["_cycle_used"] = chosen + 1
    s, e_ = cycle_starts[chosen], cycle_ends[chosen]
    return E_all[s:e_], I_all[s:e_], meta


def _compute_charge(E, I_mA, scan_rate_mV_s):
    """Q (mC) = integral I dt = integral I dE / nu.
    Uses np.trapezoid (np.trapz removed in numpy 2.0)."""
    nu = max(scan_rate_mV_s / 1000.0, 1e-9)
    dE = np.diff(E)
    if len(dE) < 2:
        return 0.0, 0.0, 0.0
    sign_changes = np.where(np.diff(np.sign(dE)) != 0)[0]
    vertex = int(sign_changes[0] + 1) if len(sign_changes) > 0 else len(E) // 2
    Q_f = float(np.trapezoid(I_mA[:vertex + 1], E[:vertex + 1]) / nu)
    Q_b = float(np.trapezoid(I_mA[vertex:], E[vertex:]) / nu)
    return Q_f + Q_b, Q_f, Q_b


def load_cv_lsv(f, unit_factor=1.0):
    """Load CV/LSV from non-IDF formats (CSV, TXT, DTA)."""
    tmp = save_upload(f)
    suffix = Path(f.name).suffix.lower()
    try:
        if suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            ds = GamryParser().parse(tmp)
            E = ds.potential
            I = ds.current * unit_factor
            return E, I, ds.metadata
        else:
            df = read_csv_safe(tmp)
            c = df.columns.tolist()
            E = df[c[0]].to_numpy(float)
            I = df[c[1]].to_numpy(float) * unit_factor
            return E, I, {}
    finally:
        os.unlink(tmp)


@st.cache_data(show_spinner=False)
def _parse_idf_cached(_file_bytes: bytes, cycle_idx: int = -1):
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".idf") as t:
        t.write(_file_bytes)
        tmp = t.name
    try:
        return _load_ivium_cv(tmp, cycle_idx=cycle_idx)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _show_validation(result):
    """Render a ValidationResult in the correct coloured box."""
    css = {"ok": "val-ok", "warning": "val-warn", "error": "val-err"}.get(result.severity, "val-warn")
    html = f'<div class="{css}">{result.message}'
    if result.suggested_action:
        html += f'<br><small>💡 {result.suggested_action}</small>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _auto_ph(compound: str, conc: float) -> float:
    """Estimate pH from electrolyte compound and concentration."""
    if compound == "H2SO4":
        h1 = conc
        Ka2 = 10 ** (-1.99)
        x = Ka2 * conc / (h1 + Ka2) if h1 > 0 else conc
        return max(-math.log10(h1 + x), -1.0)
    elif compound in ("HCl", "HClO4", "HNO3"):
        return max(-math.log10(conc), -1.0)
    elif compound in ("KOH", "NaOH"):
        return min(14.0 + math.log10(conc), 15.0)
    elif compound in ("Na2CO3", "NH3"):
        return 11.6
    return 14.0 if conc > 0 else 7.0


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-title">System Settings</p>', unsafe_allow_html=True)
    system_type = st.selectbox("System type", ["AOR", "Battery", "Corrosion", "Fuel Cell", "Biosensor"])
    catalyst = st.text_input("Catalyst", placeholder="e.g. Pt/C, carbon_material, PtRu/C")

    # ── Catalyst type ────────────────────────────────────────────────────────
    catalyst_type_ui = st.selectbox(
        "Catalyst type",
        ["Noble Metal (Pt, Pd, Au, Rh)", "Alloy (PtRu, PtSn, PdAu, PtCu)",
         "Metal Oxide (NiO, Co₃O₄, MnO₂)", "Carbon Material (N-doped C, CNT, Graphene)"],
    )
    _CTYPE_MAP = {
        "Noble Metal (Pt, Pd, Au, Rh)": "noble_metal",
        "Alloy (PtRu, PtSn, PdAu, PtCu)": "alloy",
        "Metal Oxide (NiO, Co₃O₄, MnO₂)": "metal_oxide",
        "Carbon Material (N-doped C, CNT, Graphene)": "carbon_material",
    }
    catalyst_type = _CTYPE_MAP[catalyst_type_ui]

    # ── Carbon subtype (only for carbon_material) ─────────────────────────
    carbon_subtype_key = "carbon_material"
    if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
        carbon_subtype_ui = st.selectbox(
            "Carbon subtype", list(CARBON_SUBTYPE_MAP.keys()),
            help="Used for C_dl validation range",
        )
        carbon_subtype_key = CARBON_SUBTYPE_MAP[carbon_subtype_ui]
    elif catalyst_type == "carbon_material":
        carbon_subtype_ui = st.selectbox(
            "Carbon subtype",
            ["Graphene / rGO", "N-doped Graphene", "Carbon Nanotubes (CNT / MWCNT)",
             "Activated / Porous Carbon", "Carbon Black (Vulcan XC-72)", "Other / Unknown"],
        )

    # ── Electrolyte ────────────────────────────────────────────────────────
    electrolyte = st.selectbox("Electrolyte media", ["Acidic", "Alkaline", "NaCl", "PBS", "Other"])
    ekey = "acidic" if electrolyte == "Acidic" else "alkaline" if electrolyte == "Alkaline" else "acidic"
    if electrolyte == "Acidic":
        elec_compound = st.selectbox("Acid type", ["H₂SO₄", "HClO₄", "HCl", "HNO₃"])
        _COMPOUND_MAP = {"H₂SO₄": "H2SO4", "HClO₄": "HClO4", "HCl": "HCl", "HNO₃": "HNO3"}
        elec_compound_key = _COMPOUND_MAP[elec_compound]
    elif electrolyte == "Alkaline":
        elec_compound = st.selectbox("Base type", ["KOH", "NaOH", "Na₂CO₃", "NH₃"])
        _COMPOUND_MAP = {"KOH": "KOH", "NaOH": "NaOH", "Na₂CO₃": "Na2CO3", "NH₃": "NH3"}
        elec_compound_key = _COMPOUND_MAP[elec_compound]
    else:
        elec_compound = electrolyte
        elec_compound_key = electrolyte

    # ── Alcohol — isopropanol added ────────────────────────────────────────
    alcohol = st.selectbox(
        "Alcohol",
        ["ethanol", "methanol", "isopropanol (2-propanol)", "ethylene glycol", "glycerol", "N/A"],
        disabled=(system_type != "AOR"),
    )
    _ALCOHOL_KEY_MAP = {
        "ethanol": "ethanol", "methanol": "methanol",
        "isopropanol (2-propanol)": "isopropanol",
        "ethylene glycol": "ethylene glycol",
        "glycerol": "glycerol", "N/A": "N/A",
    }
    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)
    alcohol_conc = st.number_input(
        "Alcohol conc. (M)", value=0.25, step=0.05, min_value=0.0,
        help="Concentration of alcohol in solution",
        disabled=(system_type != "AOR"))
    eis_pot = st.number_input("EIS potential (V)", value=0.5, step=0.01)

    st.divider()
    st.markdown('<p class="section-title">Electrode Parameters</p>', unsafe_allow_html=True)
    diameter_mm = st.number_input(
        "Disk diameter (mm)", value=0.0, min_value=0.0, step=0.5,
        help="If > 0, overrides area below. Standard GCE: 3 mm or 5 mm")
    if diameter_mm > 0:
        area = math.pi * (diameter_mm / 20.0) ** 2
        st.caption(f"→ Area = {area:.4f} cm²")
    else:
        area = st.number_input("Geometric area (cm²)", value=1.0, min_value=0.0001,
                               step=0.0001, format="%.4f")
    _ecsa_label = "ECSA (cm²_BET)" if catalyst_type == "carbon_material" else "ECSA (cm²_metal)"
    ecsa = st.number_input(_ecsa_label, value=0.0, step=0.1, min_value=0.0)
    mass_ug = st.number_input(
        "Deposited mass (µg)", value=0.0, min_value=0.0, step=1.0,
        help="If > 0, overrides loading below. e.g. 5 µg on 3 mm GCE")
    if mass_ug > 0 and area > 0:
        loading = (mass_ug / 1000.0) / area
        st.caption(f"→ Loading = {loading:.4f} mg/cm²  |  mass = {mass_ug:.1f} µg")
    else:
        loading = st.number_input("Loading (mg/cm²)", value=0.0, step=0.01, min_value=0.0)
    _mass_mg = mass_ug / 1000.0 if mass_ug > 0 else loading * area

    st.divider()
    st.markdown('<p class="section-title">Experimental Conditions</p>', unsafe_allow_html=True)
    temperature = st.number_input("Temperature (°C)", value=25, min_value=0, max_value=200)
    current_unit = st.selectbox("Current unit", ["mA", "A", "μA", "nA"])
    e_ref_type = st.selectbox("Reference electrode", list(E_REF_MAP.keys()))
    e_ref_val = E_REF_MAP[e_ref_type]
    elec_conc = st.number_input(
        "Electrolyte conc. (M)", value=1.0, step=0.001,
        min_value=0.0, format="%.4f", help="Used for pH and RHE conversion only")

    _ph_auto = _auto_ph(elec_compound_key, elec_conc)
    _ph_override = st.checkbox("Override pH manually", value=False)
    if _ph_override:
        ph_value = st.number_input("pH (manual)",
            value=float(round(_ph_auto, 2)), min_value=0.0, max_value=14.0, step=0.1)
    else:
        ph_value = _ph_auto
        st.caption(f"pH = **{ph_value:.2f}** (auto — {elec_compound_key} {elec_conc} M)")

    sub_conc = st.number_input("Substrate conc. (M)", value=1.0, step=0.1)
    unit_factor = UNIT_MAP.get(current_unit, 1.0)

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
        "R_s (Ω) — from EIS fit", value=0.0, step=0.1, min_value=0.0,
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

    # ── C_dl validation range (for carbon_material) ───────────────────────
    if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
        st.divider()
        st.markdown('<p class="section-title">🔬 C_dl Validation Range</p>', unsafe_allow_html=True)
        ref_range = CDL_RANGES.get(carbon_subtype_key, CDL_RANGES["carbon_material"])
        use_custom_cdl = st.checkbox(
            "Use custom C_dl range", value=False,
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
                system_type=system_type, catalyst=catalyst, electrolyte=ekey,
                alcohol=alcohol_key if system_type == "AOR" else "",
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

# ── E_onset auto-select (computed after sidebar, before tabs) ─────────────
default_onset_method = _ONSET_METHOD_MAP.get(catalyst_type, "tangent")
_onset_methods = ["tangent", "threshold", "derivative"]
_auto_idx = _onset_methods.index(default_onset_method)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 CV Analysis", "📉 LSV Analysis", "🔬 EIS Analysis",
    "🤖 EIS-GPT", "🔗 Correlation", "⚗️ K-L Analysis"
])

# ══════════════════ CV ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<h3>Cyclic Voltammetry Analysis</h3>', unsafe_allow_html=True)
    if actual_rs > 0:
        st.info(f"⚡ iR compensation will be applied — R_s = {actual_rs:.3f} Ω "
                f"(from sidebar). E_onset and peaks will be on iR-corrected potential.")
    col1, col2 = st.columns([1, 1])
    with col1:
        cv_file = st.file_uploader("Upload CV file", type=CV_FORMATS, key="cv_up")
        sr_cv = st.number_input("Scan rate (mV/s)", value=50, min_value=1)
        cycle_idx = st.number_input(
            "Cycle to analyse (0=first, -1=last complete)", value=-1, min_value=-1, step=1,
            help="Ivium often saves the last cycle incomplete; -1 picks the last closed one")
        use_smooth = st.checkbox("Smooth noisy curve (Savitzky-Golay)", value=False)
        sg_window = st.slider("SG window (odd)", 5, 31, 11, 2) if use_smooth else 11

        # ── E_onset method auto + override ────────────────────────────────
        st.caption(f"🤖 Auto-selected: **{default_onset_method}** (based on catalyst type)")
        om = st.radio(
            "E_onset method (override if needed)",
            _onset_methods, index=_auto_idx, horizontal=True,
            help=f"Auto: {default_onset_method} for {catalyst_type_ui}. Change if needed.",
        )

    with col2:
        if cv_file:
            try:
                if Path(cv_file.name).suffix.lower() == ".idf":
                    pot, cur, _meta = _parse_idf_cached(cv_file.getvalue(), cycle_idx=int(cycle_idx))
                else:
                    pot, cur, _meta = load_cv_lsv(cv_file, unit_factor=unit_factor)
                if "Scanrate" in _meta:
                    sr_cv = int(_meta["Scanrate"] * 1000)
                if use_smooth:
                    from scipy.signal import savgol_filter
                    w = sg_window if sg_window % 2 == 1 else sg_window + 1
                    min_w = 5  # polyorder=3 needs at least 5 points
                    w = max(min_w, min(w, len(cur) - (1 if len(cur) % 2 == 0 else 0)))
                    if w % 2 == 0:
                        w -= 1
                    if w >= min_w:
                        cur = savgol_filter(cur, window_length=w, polyorder=3)
                    else:
                        st.warning(f"Signal too short ({len(cur)} pts) for smoothing. Skipping.")
                _nc = _meta.get("_n_cycles", "?")
                _cu = _meta.get("_cycle_used", "?")
                _ul = _meta.get("_unit_label", "mA")
                st.success(f"✅ {len(pot)} points | {cv_file.name} | "
                           f"cycle {_cu}/{_nc} | unit: {_ul} | sr: {sr_cv} mV/s"
                           + (" | smoothed" if use_smooth else ""))
                Q_total, Q_f, Q_b = _compute_charge(pot, cur, sr_cv)
                st.session_state.update({"cv_Q_total": Q_total, "cv_Q_f": Q_f, "cv_Q_b": Q_b})
                from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo
                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                ana = CVAnalyzer(scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, onset_method=om,
                                 electrolyte=_el, catalyst_type=catalyst_type)
                r = ana.analyze(pot, cur, r_s_ohms=actual_rs)
                st.session_state.update({
                    "cv_r": r, "cv_pot": pot, "cv_cur": cur,
                    "cv_pot_corr": CVAnalyzer.apply_ir_compensation(pot, cur, actual_rs)
                                  if actual_rs > 0 else pot
                })
            except Exception as e:
                st.error(f"Error: {e}")

    if "cv_r" in st.session_state:
        r = st.session_state["cv_r"]
        st.divider()
        if r.ir_compensated:
            st.success(f"✅ iR-corrected | R_s = {r.r_s_used:.3f} Ω")
        _is_mf = catalyst_type == "carbon_material"
        _e_onset_rhe = r.e_onset + e_ref_val + 0.059 * ph_value
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E_onset (vs ref)", f"{r.e_onset:.4f} V")
        c2.metric("E_onset (vs RHE)", f"{_e_onset_rhe:.4f} V",
                  help=f"= {r.e_onset:.4f} + {e_ref_val:.3f}(ref) + 0.059×{ph_value:.2f}(pH)")
        if _is_mf:
            c3.metric("Net faradaic I", f"{r.net_faradaic_current_mA:.4f} mA")
            c4.metric("C_dl", f"{r.cdl_mF_cm2:.4f} mF/cm²")
        else:
            c3.metric("I_b", f"{r.i_backward_peak:.4f} mA")
            c4.metric("I_f/I_b", f"{r.if_ib_ratio:.3f}" if not np.isnan(r.if_ib_ratio) else "N/A")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("j_f (geometric)", f"{r.j_forward_peak:.4f} mA cm⁻²")
        if not _is_mf:
            c6.metric("j_b (geometric)", f"{r.j_backward_peak:.4f} mA cm⁻²")
        _ecsa_unit = "cm²_BET" if _is_mf else "cm²_Pt"
        if r.ecsa > 0:
            c7.metric("j_f (ECSA)", f"{r.j_specific_forward:.4f} mA/{_ecsa_unit}")
        if loading > 0:
            c8.metric("Mass activity", f"{r.j_forward_peak / loading:.1f} A/g",
                      help="j_f / loading")
        if "cv_Q_total" in st.session_state:
            _Qf = st.session_state["cv_Q_f"]
            _Qb = st.session_state["cv_Q_b"]
            cq1, cq2, cq3 = st.columns(3)
            cq1.metric("Q_forward (mC)", f"{_Qf:.3f}")
            cq2.metric("Q_backward (mC)", f"{abs(_Qb):.3f}")
            cq3.metric("Q_f / |Q_b|", f"{abs(_Qf/_Qb):.3f}" if _Qb != 0 else "N/A",
                       help="≈1 = reversible")
        st.caption(f"Alcohol: **{alcohol}** {alcohol_conc} M | "
                   f"Electrolyte: **{elec_compound}** {elec_conc} M | pH = {ph_value:.2f}")
        st.info(f"**Interpretation:** {r.interpretation}")

        # ── C_dl smart validation (carbon only) ───────────────────────────
        if _is_mf and _STANDARDS_AVAILABLE and hasattr(r, "cdl_mF_cm2"):
            st.markdown("#### 🔬 C_dl Validation")
            val_result = CarbonValidator.validate_cdl(
                cdl_mF_cm2=r.cdl_mF_cm2, material_key=carbon_subtype_key,
                user_min_uF=cdl_user_min, user_max_uF=cdl_user_max,
            )
            _show_validation(val_result)
            onset_val = CarbonValidator.validate_onset(r.e_onset)
            _show_validation(onset_val)

        import plotly.graph_objects as go
        fig = go.Figure()
        x_plot = st.session_state.get("cv_pot_corr", st.session_state["cv_pot"])
        _cur_plot = st.session_state["cv_cur"]
        j_arr = _cur_plot / area if area > 0 else _cur_plot
        fig.add_trace(go.Scatter(x=x_plot, y=j_arr, mode="lines",
                                 name="CV" + (" (iR-corrected)" if actual_rs > 0 else ""),
                                 line=dict(color="#2563eb", width=2)))
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset = {r.e_onset:.3f} V",
                      annotation_font=dict(color="#d97706"))
        if r.e_forward_peak is not None:
            fig.add_trace(go.Scatter(x=[r.e_forward_peak], y=[r.i_forward_peak / area],
                                     mode="markers",
                                     name=f"j_f = {r.i_forward_peak/area:.3f} mA cm⁻²",
                                     marker=dict(color="#16a34a", size=12, symbol="star")))
        if r.e_backward_peak is not None and not _is_mf:
            fig.add_trace(go.Scatter(x=[r.e_backward_peak], y=[r.i_backward_peak / area],
                                     mode="markers",
                                     name=f"j_b = {r.i_backward_peak/area:.3f} mA cm⁻²",
                                     marker=dict(color="#dc2626", size=12, symbol="star")))
        title = f"j vs E — {sr_cv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}"
        if actual_rs > 0:
            title += f" | iR-corrected (R_s={actual_rs:.1f}Ω)"
        fig.update_layout(**PLOTLY_LAYOUT, title=title,
                          xaxis_title=f"E (V vs {e_ref_type})",
                          yaxis_title="j (mA cm⁻²)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Batch CV ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<h4>📊 Batch Analysis — Mean ± SD (n≥3)</h4>', unsafe_allow_html=True)
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
                        p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower() == ".idf" else unit_factor)
                        pots_b.append(p)
                        curs_b.append(c)
                    batch_ana = BatchCVAnalyzer(
                        scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,
                        catalyst_type=catalyst_type, electrolyte=_el_b,
                        onset_method=om, catalyst_loading=loading, r_s_ohms=actual_rs,
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
        c1, c2, c3, c4 = st.columns(4)
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
                line=dict(color="rgba(37,99,235,0)"), name="± SD band",
            ))
            fig_b.add_trace(go.Scatter(
                x=pot_c, y=cur_m, mode="lines",
                name=f"Mean CV (n={br.n_valid})", line=dict(color="#2563eb", width=2.5),
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
    st.markdown('<h3>Linear Sweep Voltammetry Analysis</h3>', unsafe_allow_html=True)
    if actual_rs > 0:
        st.info(f"⚡ iR compensation will be applied — R_s = {actual_rs:.3f} Ω. "
                f"Tafel slope and E_onset computed on iR-corrected potential.")
    col1, col2 = st.columns([1, 1])
    with col1:
        lsv_file = st.file_uploader("Upload LSV file", type=CV_FORMATS, key="lsv_up")
        sr_lsv = st.number_input("Scan rate (mV/s)", value=5, min_value=1, key="sr_lsv")
        tj_min = st.number_input("Tafel j_min (mA/cm²)", value=0.1, step=0.05)
        tj_max = st.number_input("Tafel j_max (mA/cm²)", value=2.0, step=0.5)
        st.caption(f"🤖 Auto-selected: **{default_onset_method}** (based on catalyst type)")
        om_lsv = st.radio(
            "E_onset method (override if needed)",
            _onset_methods, index=_auto_idx, horizontal=True, key="om_lsv",
            help=f"Auto: {default_onset_method} for {catalyst_type_ui}.",
        )
    with col2:
        if lsv_file:
            try:
                if Path(lsv_file.name).suffix.lower() == ".idf":
                    pot_lsv, cur_lsv, _ = load_cv_lsv(lsv_file, unit_factor=1.0)
                else:
                    pot_lsv, cur_lsv, _ = load_cv_lsv(lsv_file, unit_factor=unit_factor)
                st.success(f"✅ {len(pot_lsv)} points | {lsv_file.name}")
                from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo
                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                la = LSVAnalyzer(scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, electrolyte=_el,
                                 catalyst_type=catalyst_type, e_ref_vs_rhe=e_ref_val,
                                 tafel_current_range=(tj_min, tj_max))
                lr = la.analyze(pot_lsv, cur_lsv, r_s_ohms=actual_rs)
                st.session_state.update({"lsv_r": lr, "lsv_pot": pot_lsv, "lsv_cur": cur_lsv})
            except Exception as e:
                st.error(f"Error: {e}")

    if "lsv_r" in st.session_state:
        import math
        r = st.session_state["lsv_r"]
        st.divider()
        if r.ir_compensated:
            st.success(f"✅ iR-corrected | R_s = {r.r_s_used:.3f} Ω")
        c1, c2, c3 = st.columns(3)
        c1.metric("E_onset", f"{r.e_onset:.4f} V")
        c2.metric("Tafel slope", f"{r.tafel_slope:.1f} mV/dec")
        c3.metric("j₀", f"{r.exchange_current_density:.3e} mA/cm²")
        c4, c5, c6 = st.columns(3)
        c4.metric("η @ 10 mA/cm²", f"{r.overpotential_10*1000:.1f} mV" if not math.isnan(r.overpotential_10) else "N/A")
        c5.metric("η @ 50 mA/cm²", f"{r.overpotential_50*1000:.1f} mV" if not math.isnan(r.overpotential_50) else "N/A")
        c6.metric("η @ 100 mA/cm²", f"{r.overpotential_100*1000:.1f} mV" if not math.isnan(r.overpotential_100) else "N/A")
        if loading > 0:
            st.metric("Mass activity", f"{r.mass_activity:.3f} mA/mg_cat")
        _sa_unit = "cm²_BET" if catalyst_type == "carbon_material" else "cm²_Pt"
        if ecsa > 0:
            st.metric("Specific activity", f"{r.specific_activity:.4f} mA/{_sa_unit}")
        if catalyst_type == "carbon_material" and _STANDARDS_AVAILABLE:
            st.markdown("#### 🔬 Tafel Validation")
            tafel_val = CarbonValidator.validate_tafel(r.tafel_slope, electrolyte=ekey)
            _show_validation(tafel_val)
        st.info(f"**Mechanism:** {r.mechanism_interpretation}")
        st.success(f"**Performance:** {r.performance_rating}")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("LSV Curve", "Tafel Plot"))
        if area > 0:
            j_lsv = st.session_state["lsv_cur"] / area
        else:
            j_lsv = st.session_state["lsv_cur"].copy()
            st.warning("Electrode area = 0 — showing raw current (mA) instead of current density.")
        _nernst = (8.314 * (273.15 + temperature) / 96485.0) * math.log(10)
        _rhe_offset = e_ref_val + _nernst * ph_value
        if actual_rs > 0:
            from eisforge.analysis.lsv_analyzer import LSVAnalyzer
            cur_a_lsv = st.session_state["lsv_cur"] / 1000.0
            p_lsv = LSVAnalyzer.apply_ir_compensation(st.session_state["lsv_pot"], cur_a_lsv, actual_rs)
            p_lsv = p_lsv + _rhe_offset
        else:
            p_lsv = st.session_state["lsv_pot"] + _rhe_offset
        fig.add_trace(go.Scatter(x=p_lsv, y=j_lsv, mode="lines",
                                 name="LSV" + ("+iR-corr." if actual_rs > 0 else ""),
                                 line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset={r.e_onset:.3f}V",
                      annotation_font=dict(color="#d97706"), row=1, col=1)
        mask = (j_lsv > 0) & (j_lsv >= tj_min) & (j_lsv <= tj_max)
        if np.sum(mask) > 3:
            fig.add_trace(go.Scatter(x=np.log10(j_lsv[mask]), y=p_lsv[mask],
                                     mode="markers", name="Tafel region",
                                     marker=dict(color="#7c3aed", size=6)), row=1, col=2)
        title = f"LSV — {sr_lsv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}"
        if actual_rs > 0:
            title += " | iR-corrected"
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title=title)
        fig.update_xaxes(title_text=f"Potential (V vs {e_ref_type})", row=1, col=1)
        fig.update_yaxes(title_text="j (mA/cm²)", row=1, col=1)
        fig.update_xaxes(title_text="log(j) [mA/cm²]", row=1, col=2)
        fig.update_yaxes(title_text="E (V)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

    # ── Batch LSV ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<h4>📊 Batch Analysis — Mean ± SD (n≥3)</h4>', unsafe_allow_html=True)
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
                        p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower() == ".idf" else unit_factor)
                        pots_b.append(p)
                        curs_b.append(c)
                    batch_lsv = BatchLSVAnalyzer(
                        scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                        catalyst_type=catalyst_type, electrolyte=_el_b,
                        catalyst_loading=loading, e_ref_vs_rhe=e_ref_val,
                        tafel_current_range=(tj_min, tj_max), r_s_ohms=actual_rs,
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
        c1, c2, c3 = st.columns(3)
        c1.metric(f"E_onset ({n_str})", f"{blr.e_onset_mean:.4f} V", delta=f"± {blr.e_onset_std:.4f}")
        _tafel_note = " (normal)" if _is_mf else ""
        c2.metric(f"Tafel ({n_str})", f"{blr.tafel_mean:.1f} mV/dec{_tafel_note}", delta=f"± {blr.tafel_std:.1f}")
        c3.metric(f"j₀ ({n_str})", f"{blr.j0_mean:.3e} mA/cm²", delta=f"± {blr.j0_std:.3e}")
        c4, c5, c6 = st.columns(3)
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
            j_m = blr.j_mean_curve
            j_s = blr.j_std_curve
            fig_blsv.add_trace(go.Scatter(
                x=np.concatenate([pot_c, pot_c[::-1]]),
                y=np.concatenate([j_m + j_s, (j_m - j_s)[::-1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(37,99,235,0)"), name="± SD band",
            ))
            fig_blsv.add_trace(go.Scatter(
                x=pot_c, y=j_m, mode="lines",
                name=f"Mean LSV ({n_str})", line=dict(color="#2563eb", width=2.5),
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
    st.markdown('<h3>Electrochemical Impedance Spectroscopy</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        eis_file = st.file_uploader("Upload EIS file", type=EIS_FORMATS, key="eis_up")
        if _STANDARDS_AVAILABLE:
            eec_suggestion = suggest_eec(
                catalyst_type=catalyst_type, electrolyte=ekey, user_circuit=None,
            )
            _suggested_circuit = eec_suggestion["circuit"]
            _suggested_p0 = ", ".join(f"{v:.3e}" for v in eec_suggestion["p0"]) if eec_suggestion["p0"] else ""
            st.info(f"🤖 Suggested EEC: `{_suggested_circuit}` — " + eec_suggestion["note"])
        else:
            _suggested_circuit = "R0-p(R1,CPE1)"
            _suggested_p0 = "30, 31000, 2e-7, 0.78"
        if "lit" in st.session_state and st.session_state["lit"].system_found:
            g = st.session_state["lit"]
            _suggested_circuit = g.recommended_circuit
            _suggested_p0 = ", ".join(f"{v:.3e}" for v in g.initial_guess.values())
        circ = st.text_input("Equivalent circuit (edit or accept suggestion)", value=_suggested_circuit)
        p0s = st.text_input("Initial guess (comma-separated)", value=_suggested_p0 if _suggested_p0 else "30, 31000, 2e-7, 0.78")
        use_bounds = st.checkbox("Use smart bounds", value=False)
        st.caption("💡 Tip: After fit, use R0 value as R_s for iR compensation in CV/LSV tabs.")
    with col2:
        if eis_file:
            try:
                fr, zr, zi, meta_eis = load_eis(eis_file)
                st.success(f"✅ {len(fr)} points | {eis_file.name}")
                st.session_state.update({"eis_fr": fr, "eis_zr": zr, "eis_zi": zi})
            except Exception as e:
                st.error(f"Error loading EIS: {e}")

    if "eis_fr" in st.session_state:
        fr = st.session_state["eis_fr"]
        zr = st.session_state["eis_zr"]
        zi = st.session_state["eis_zi"]
        import plotly.graph_objects as go
        fig_eis = go.Figure()
        fig_eis.add_trace(go.Scatter(x=zr, y=-zi, mode="markers+lines",
                                     name="Nyquist", marker=dict(color="#2563eb", size=6)))
        fig_eis.update_layout(**PLOTLY_LAYOUT, title="Nyquist Plot",
                              xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)")
        st.plotly_chart(fig_eis, use_container_width=True)

        if st.button("▶ Run CNLS Fit", type="primary"):
            try:
                from eisforge.fitting.cnls_fitter import CNLSFitter
                p0_list = [float(x.strip()) for x in p0s.split(",")]
                bounds = smart_bounds(circ, p0_list) if use_bounds else (-np.inf, np.inf)
                fitter = CNLSFitter(circuit=circ, p0=p0_list, bounds=bounds)
                fit_r = fitter.fit(fr, zr, zi)
                st.session_state["eis_fit"] = fit_r
                st.success(f"✅ Fit converged | χ² = {fit_r.chi_squared:.4e}")
            except Exception as e:
                st.error(f"Fit error: {e}")

        if "eis_fit" in st.session_state:
            fit_r = st.session_state["eis_fit"]
            st.markdown("#### Fit Parameters")
            param_df = pd.DataFrame({"Parameter": fit_r.param_names,
                                     "Value": fit_r.params,
                                     "Std Error": fit_r.errors})
            st.dataframe(param_df, use_container_width=True, hide_index=True)
            if _STANDARDS_AVAILABLE:
                st.markdown("#### 🔬 EEC Validation")
                val = CarbonValidator.validate_eis_params(
                    fit_r.param_names, fit_r.params, catalyst_type, ekey
                )
                for v in val:
                    _show_validation(v)


# ══════════════════ EIS-GPT ══════════════════════════════════════════════════
with tab4:
    st.markdown('<h3>🤖 EIS-GPT Interpreter</h3>', unsafe_allow_html=True)
    st.info("AI-powered EIS spectrum interpretation using physics-informed language model.")
    if st.button("🤖 Interpret EIS Spectrum"):
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
            st.warning("Run CNLS fit first (EIS Analysis tab).")


# ══════════════════ CORRELATION ══════════════════════════════════════════════
with tab5:
    st.markdown('<h3>🔗 EIS–CV Correlation</h3>', unsafe_allow_html=True)
    if "eis_fit" in st.session_state and "cv_r" in st.session_state:
        fit_r = st.session_state["eis_fit"]
        cv_r = st.session_state["cv_r"]
        st.markdown("#### Parameter Correlation Table")
        corr_data = {
            "R_s (Ω)": [next((v for n, v in zip(fit_r.param_names, fit_r.params) if n == "R0"), "N/A")],
            "R_ct (Ω)": [next((v for n, v in zip(fit_r.param_names, fit_r.params) if n in ("R1", "Rct")), "N/A")],
            "E_onset (V)": [cv_r.e_onset],
            "j_f (mA/cm²)": [cv_r.j_forward_peak],
        }
        st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)
    else:
        st.info("Complete both EIS fit (tab 3) and CV analysis (tab 1) to see correlations.")


# ══════════════════ K-L ANALYSIS ═════════════════════════════════════════════
with tab6:
    st.markdown('<h3>⚗️ Koutecký–Levich Analysis</h3>', unsafe_allow_html=True)
    st.info("Upload LSV files at different rotation speeds (rpm) for K-L analysis.")
    kl_files = st.file_uploader(
        "Upload LSV files at different rotation speeds",
        type=CV_FORMATS, accept_multiple_files=True, key="kl_up"
    )
    rpms_str = st.text_input("Rotation speeds (rpm, comma-separated)", value="400,900,1600,2500")
    if kl_files and st.button("▶ Run K-L Analysis", type="primary"):
        try:
            from eisforge.analysis.kl_analyzer import KLAnalyzer
            rpms = [float(x.strip()) for x in rpms_str.split(",")]
            pots_kl, curs_kl = [], []
            for f in kl_files:
                p, c, _ = load_cv_lsv(f, unit_factor=unit_factor)
                pots_kl.append(p)
                curs_kl.append(c)
            kla = KLAnalyzer(electrode_area=area, temperature=temperature,
                             e_ref_vs_rhe=e_ref_val)
            klr = kla.analyze(pots_kl, curs_kl, rpms)
            st.session_state["kl_r"] = klr
            st.success(f"✅ n_electrons = {klr.n_electrons:.2f} | "
                       f"j_k = {klr.kinetic_current:.4f} mA/cm²")
        except Exception as e:
            st.error(f"K-L error: {e}")
    if "kl_r" in st.session_state:
        klr = st.session_state["kl_r"]
        import plotly.graph_objects as go
        fig_kl = go.Figure()
        for rpm, slope_pts in zip(klr.rpms, klr.kl_lines):
            fig_kl.add_trace(go.Scatter(x=slope_pts[0], y=slope_pts[1],
                                        mode="lines", name=f"{rpm:.0f} rpm"))
        fig_kl.update_layout(**PLOTLY_LAYOUT, title="K-L Plot",
                             xaxis_title="ω⁻¹/² (rad/s)⁻¹/²",
                             yaxis_title="j⁻¹ (cm²/mA)")
        st.plotly_chart(fig_kl, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Electron transfer number (n)", f"{klr.n_electrons:.2f}")
        c2.metric("Kinetic current density (j_k)", f"{klr.kinetic_current:.4f} mA/cm²")
