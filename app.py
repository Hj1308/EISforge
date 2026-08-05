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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg:#ffffff;--surface:#f7f7fb;--sidebar:#fbfbfe;--border:#eceaf4;--border-strong:#ddd9ec;--text:#18162a;--muted:#76738a;--accent:#6d28d9;--accent-hover:#5b21b6;--accent-soft:#f3eefe;--accent-bd:#e4dafb;--green:#059669;--green-soft:#ecfdf5;--green-bd:#a7f3d0;--warn:#b45309;--warn-soft:#fffbeb;--warn-bd:#fde68a;--danger:#dc2626;--danger-soft:#fef2f2;--danger-bd:#fecaca;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Plus Jakarta Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--sidebar)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text);}
.title{font-family:'Syne',sans-serif;font-size:2.7rem;font-weight:800;color:var(--text);text-align:center;margin:0;letter-spacing:-.01em;}
.subtitle{color:var(--muted);font-size:.85rem;font-family:'JetBrains Mono',monospace;text-align:center;margin-top:.4rem;}
.subtitle a{color:var(--accent);text-decoration:none;}
.section-title{font-size:.66rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin:.4rem 0 .6rem;font-family:'JetBrains Mono',monospace;font-weight:600;}
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;padding:.5rem 1.4rem!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
.stButton>button:hover{background:var(--accent-hover)!important;}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.85rem 1rem;}
div[data-testid="stMetric"] label{color:var(--muted)!important;font-family:'JetBrains Mono',monospace!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:.08em;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-variant-numeric:tabular-nums;}
.fmt{display:inline-block;padding:.12rem .55rem;border-radius:6px;font-size:.7rem;font-family:'JetBrains Mono',monospace;background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent-bd);margin:.12rem;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
.ir-box{background:var(--accent-soft);border:1px solid var(--accent-bd);border-radius:12px;padding:.8rem;margin:.5rem 0;}
.val-ok{background:var(--green-soft);border:1px solid var(--green-bd);border-radius:12px;padding:.6rem .85rem;margin:.3rem 0;}
.val-warn{background:var(--warn-soft);border:1px solid var(--warn-bd);border-radius:12px;padding:.6rem .85rem;margin:.3rem 0;}
.val-err{background:var(--danger-soft);border:1px solid var(--danger-bd);border-radius:12px;padding:.6rem .85rem;margin:.3rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">⚡ EISForge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Physics-Informed ML for EIS Analysis · by Hoda Jafari · 2026 · '
    '<a href="https://github.com/Hj1308/EISforge">GitHub</a></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center">'
    '<span class="fmt">.idf Ivium</span> <span class="fmt">.dta Gamry</span> '
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
        CarbonValidator, suggest_eec, CDL_RANGES,  # noqa: F401
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
            return pd.read_csv(
                path, encoding=enc, sep=None, engine="python",
                comment="#", skip_blank_lines=True,
            )
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
            from eisforge.parsers.ivium_parser import IviumIDFParser
            ds = IviumIDFParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            ds = GamryParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag, ds.metadata
        elif suffix in (".mpt", ".mpr"):
            from galvani import BioLogic
            mpr = BioLogic.MPRfile(tmp)
            df = mpr.DF
            return (
                df["freq/Hz"].to_numpy(),
                df["Re(Z)/Ohm"].to_numpy(),
                df["-Im(Z)/Ohm"].to_numpy(),
                {"source": "BioLogic"},
            )
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
    """Load Ivium .idf CV. Returns (E_arr, I_mA, meta).
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

    # Extract ONLY the primary_data block (anchored to the section header)
    _lines = text.splitlines()
    E_list, I_list = [], []
    for _i, _ln in enumerate(_lines):
        if _ln.strip().lower() == "primary_data":
            try:
                _npts = int(_lines[_i + 2].strip())
            except (ValueError, IndexError):
                _npts = len(_lines) - (_i + 3)
            for _ln2 in _lines[_i + 3:_i + 3 + _npts]:
                _parts = _ln2.split()
                if len(_parts) >= 2:
                    try:
                        E_list.append(float(_parts[0])); I_list.append(float(_parts[1]))
                    except ValueError:
                        continue
            break
    if not E_list:
        raise ValueError("No primary_data block found in .idf file")
    E_arr = np.array(E_list, dtype=float)
    I_mA = np.array(I_list, dtype=float) * unit_mult   # apply the parsed unit multiplier

    sign_ch = np.diff(np.sign(np.diff(E_arr)))
    vertices = np.where(sign_ch != 0)[0] + 1
    if len(vertices) < 2:
        meta["_n_cycles"] = 1
        meta["_cycle_used"] = 1
        return E_arr, I_mA, meta

    # Detect scan direction to split at correct turning points
    dE = np.diff(E_arr[:min(50, len(E_arr))])
    first_dir = "up" if np.median(dE) > 0 else "down"
    if first_dir == "up":
        cycle_boundaries = vertices[1::2]
    else:
        cycle_boundaries = vertices[0::2]
    if len(cycle_boundaries) == 0:
        cycle_boundaries = vertices[:1]
    cycle_starts = [0] + list(cycle_boundaries + 1)
    cycle_ends = list(cycle_boundaries + 1) + [len(E_arr)]

    n_cycles = len(cycle_starts) - 1
    meta["_n_cycles"] = n_cycles
    if cycle_idx == -1:
        chosen = max(0, n_cycles - 2) if n_cycles >= 2 else 0
    else:
        chosen = max(0, min(cycle_idx, n_cycles - 1))
    meta["_cycle_used"] = chosen + 1
    s, e_ = cycle_starts[chosen], cycle_ends[chosen]
    return E_arr[s:e_], I_mA[s:e_], meta


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
            pot = ds.potential
            cur = ds.current * unit_factor
            return pot, cur, ds.metadata
        else:
            df = read_csv_safe(tmp)
            c = df.columns.tolist()
            pot = df[c[0]].to_numpy(float)
            cur = df[c[1]].to_numpy(float) * unit_factor
            return pot, cur, {}
    finally:
        os.unlink(tmp)


@st.cache_data(show_spinner=False)
def _parse_idf_cached(content_hash: str, _file_bytes: bytes, cycle_idx: int = -1):
    # content_hash (hashable) drives the Streamlit cache key; _file_bytes is
    # underscore-prefixed so Streamlit does not try to hash the raw bytes.
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

    # ── Concentrations: acid/electrolyte + alcohol grouped together ───────
    elec_conc = st.number_input(
        "Acid / electrolyte conc. (M)", value=1.0000, step=0.001,
        min_value=0.0, format="%.4f",
        help="Used for pH and RHE conversion")
    alcohol_conc = st.number_input(
        "Alcohol conc. (M)", value=0.2500, step=0.0001, min_value=0.0,
        format="%.4f", help="Concentration of alcohol in solution",
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
    # Auto-connect: pull R0 (high-frequency intercept) from the EIS fit if one
    # exists in this session; manual override always possible.
    _rs_from_eis = 0.0
    if "eis_fit" in st.session_state:
        _rs_from_eis = float(st.session_state["eis_fit"].parameters.get("R0", 0.0))
    r_s = st.number_input(
        "R_s (Ω) — from EIS fit", value=float(_rs_from_eis), step=0.1, min_value=0.0,
        disabled=not use_ir,
        help="Auto-filled from R0 of the EIS CNLS fit when available; edit to override.",
    )
    if _rs_from_eis > 0:
        st.caption(f"↳ auto-filled from EIS fit: R₀ = {_rs_from_eis:.3f} Ω")
    if use_ir and r_s > 0:
        st.markdown(
            f'<div class="ir-box">✅ iR compensation active<br>'
            f'R_s = {r_s:.3f} Ω<br>'
            f'iR drop @ 1 mA ≈ {r_s * 1e-3 * 1000:.2f} mV</div>',
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
            cdl_user_min = st.number_input(
                "C_dl min (μF/cm²)", value=float(ref_range.cdl_min_uF), step=1.0, min_value=0.1)
            cdl_user_max = st.number_input(
                "C_dl max (μF/cm²)", value=float(ref_range.cdl_max_uF), step=1.0, min_value=1.0)
        else:
            cdl_user_min = None
            cdl_user_max = None
        st.caption(
            f"Literature range: **{ref_range.cdl_min_uF:.0f}–{ref_range.cdl_max_uF:.0f} μF/cm²**"
            f" ({ref_range.material})")
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 CV Analysis", "📉 LSV Analysis", "🔬 EIS Analysis",
    "🤖 EIS-GPT", "🔗 Correlation", "🧪 ECSA Analysis",
    "⚗️ K-L Analysis", "📊 Scan-Rate Kinetics", "⏱️ Chronoamperometry"
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
                    import hashlib as _hl
                    _cvb = cv_file.getvalue()
                    pot, cur, _meta = _parse_idf_cached(
                        _hl.md5(_cvb).hexdigest(), _cvb, cycle_idx=int(cycle_idx))
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
            c8.metric("Mass activity", f"{r.j_forward_peak / loading:.4f} A/g",
                      help="j_f / loading")
        if "cv_Q_total" in st.session_state:
            _Qf = st.session_state["cv_Q_f"]
            _Qb = st.session_state["cv_Q_b"]
            cq1, cq2, cq3 = st.columns(3)
            cq1.metric("Q_forward (mC)", f"{_Qf:.3f}")
            cq2.metric("Q_backward (mC)", f"{abs(_Qb):.3f}")
            cq3.metric("Q_f / |Q_b|", f"{abs(_Qf / _Qb):.3f}" if _Qb != 0 else "N/A",
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
        # -- close the CV loop if it is a complete cycle (endpoints meet) --
        x_plot = np.asarray(x_plot, dtype=float)
        j_arr = np.asarray(j_arr, dtype=float)
        if x_plot.size > 2:
            _span = float(x_plot.max() - x_plot.min()) or 1.0
            if abs(x_plot[-1] - x_plot[0]) < 0.05 * _span:
                x_plot = np.append(x_plot, x_plot[0])
                j_arr = np.append(j_arr, j_arr[0])
        fig.add_trace(go.Scatter(x=x_plot, y=j_arr, mode="lines",
                                 name="CV" + (" (iR-corrected)" if actual_rs > 0 else ""),
                                 line=dict(color="#2563eb", width=2)))
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset = {r.e_onset:.3f} V",
                      annotation_font=dict(color="#d97706"))
        if r.e_forward_peak is not None:
            fig.add_trace(go.Scatter(
                x=[r.e_forward_peak], y=[r.i_forward_peak / area],
                mode="markers",
                name=f"j_f = {r.i_forward_peak / area:.3f} mA cm⁻²",
                marker=dict(color="#16a34a", size=12, symbol="star"),
            ))
        if r.e_backward_peak is not None and not _is_mf:
            fig.add_trace(go.Scatter(
                x=[r.e_backward_peak], y=[r.i_backward_peak / area],
                mode="markers",
                name=f"j_b = {r.i_backward_peak / area:.3f} mA cm⁻²",
                marker=dict(color="#dc2626", size=12, symbol="star"),
            ))
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
                    _el_b = ElectrolyteInfo(
                        media=ekey, compound=elec_compound_key, concentration=elec_conc)
                    pots_b, curs_b = [], []
                    for f in batch_cv_files:
                        p, c, _ = load_cv_lsv(
                            f,
                            unit_factor=1.0 if Path(f.name).suffix.lower() == ".idf" else unit_factor,
                        )
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
        blank_lsv_file = st.file_uploader(
            "Blank LSV (electrolyte only, optional)", type=CV_FORMATS, key="lsv_blank_up",
            help="If provided, the blank is subtracted from the alcohol scan to "
                 "isolate the faradaic (net) current before analysis.",
        )
        blank_emin = st.number_input(
            "Blank subtraction lower limit (V, measured frame)", value=-0.10,
            step=0.05, format="%.2f", disabled=(blank_lsv_file is None),
            help="Data below this potential is dropped before subtraction "
                 "(removes the scan-start transient present in the blank).",
        )
        sr_lsv = st.number_input("Scan rate (mV/s)", value=5, min_value=1, key="sr_lsv")
        tafel_mode_lsv = st.radio(
            "Tafel region selection",
            ["Auto-detect", "Current window (mA/cm²)", "Potential window (V)"],
            index=0, horizontal=True, key="tafel_mode_lsv",
            help="Auto: best linear region. Current window: specify a j range. "
                 "Potential window: specify an E range -- robust for curves with no "
                 "current peak.",
        )
        if tafel_mode_lsv == "Current window (mA/cm²)":
            tj_min = st.number_input("Tafel j_min (mA/cm²)", value=0.0030, step=0.0005, format="%.4f")
            tj_max = st.number_input("Tafel j_max (mA/cm²)", value=0.0080, step=0.0005, format="%.4f")
            te_low = te_high = None
            te_frame_rhe = False
        elif tafel_mode_lsv == "Potential window (V)":
            tj_min = tj_max = None
            _te_frame = st.radio(
                "E_low / E_high reference frame",
                [f"vs {e_ref_type}", "vs RHE"], horizontal=True, key="tafel_e_frame_lsv",
            )
            te_frame_rhe = (_te_frame == "vs RHE")
            te_low = st.number_input(f"E_low ({_te_frame})", value=0.60, format="%.3f", step=0.01)
            te_high = st.number_input(f"E_high ({_te_frame})", value=0.90, format="%.3f", step=0.01)
        else:
            tj_min = tj_max = None
            te_low = te_high = None
            te_frame_rhe = False
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
                    import hashlib as _hl
                    _lvb = lsv_file.getvalue()
                    pot_lsv, cur_lsv, _ = _parse_idf_cached(
                        _hl.md5(_lvb).hexdigest(), _lvb)
                else:
                    pot_lsv, cur_lsv, _ = load_cv_lsv(lsv_file, unit_factor=unit_factor)
                st.success(f"✅ {len(pot_lsv)} points | {lsv_file.name}")
                from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo
                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)
                # ── optional blank subtraction ─────────────────────────────
                _blank_active = False
                if blank_lsv_file is not None:
                    if Path(blank_lsv_file.name).suffix.lower() == ".idf":
                        import hashlib as _hl
                        _blb = blank_lsv_file.getvalue()
                        pot_blk, cur_blk, _ = _parse_idf_cached(
                            _hl.md5(_blb).hexdigest(), _blb)
                    else:
                        pot_blk, cur_blk, _ = load_cv_lsv(blank_lsv_file, unit_factor=unit_factor)
                    _raw_alc_pot, _raw_alc_cur = pot_lsv.copy(), cur_lsv.copy()
                    _raw_blk_pot, _raw_blk_cur = pot_blk.copy(), cur_blk.copy()
                    pot_lsv, cur_lsv = LSVAnalyzer.subtract_blank(
                        pot_lsv, cur_lsv, pot_blk, cur_blk, e_min=blank_emin)
                    _blank_active = True
                    st.info(f"🧪 Blank subtracted ({blank_lsv_file.name}) | "
                            f"net current, {len(pot_lsv)} points, E ≥ {blank_emin:.2f} V")
                la = LSVAnalyzer(scan_rate=sr_lsv, electrode_area=area, ecsa=ecsa,
                                 catalyst_loading=loading, electrolyte=_el,
                                 catalyst_type=catalyst_type, e_ref_vs_rhe=e_ref_val,
                                 tafel_current_range=(tj_min, tj_max) if tj_min is not None else None)
                # Tafel mode wiring -- mirrors the backend contract in lsv_analyzer.py:
                #   auto_tafel_region=False + tafel_current_range  -> current-window fit
                #   tafel_potential_range set                     -> potential-window fit
                #                                                    (checked first, regardless
                #                                                    of auto_tafel_region)
                #   otherwise                                     -> auto-detection
                la.auto_tafel_region = (tafel_mode_lsv != "Current window (mA/cm²)")
                if tafel_mode_lsv == "Potential window (V)":
                    # Convert the user-typed window into the analyzer's internal frame,
                    # which is (raw_potential + e_ref_val) -- see module docstring above.
                    _nernst_w = (8.314 * (273.15 + temperature) / 96485.0) * math.log(10)
                    if te_frame_rhe:
                        la.tafel_potential_range = (
                            te_low - _nernst_w * ph_value, te_high - _nernst_w * ph_value,
                        )
                    else:
                        la.tafel_potential_range = (te_low + e_ref_val, te_high + e_ref_val)
                else:
                    la.tafel_potential_range = None
                lr = la.analyze(pot_lsv, cur_lsv, r_s_ohms=actual_rs)
                _sess = {"lsv_r": lr, "lsv_pot": pot_lsv, "lsv_cur": cur_lsv,
                         "lsv_blank_active": _blank_active}
                if _blank_active:
                    _sess.update({
                        "lsv_raw_alc_pot": _raw_alc_pot, "lsv_raw_alc_cur": _raw_alc_cur,
                        "lsv_raw_blk_pot": _raw_blk_pot, "lsv_raw_blk_cur": _raw_blk_cur,
                    })
                else:
                    for _k in ("lsv_raw_alc_pot", "lsv_raw_alc_cur",
                               "lsv_raw_blk_pot", "lsv_raw_blk_cur"):
                        st.session_state.pop(_k, None)
                st.session_state.update(_sess)
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
        c2.metric("Apparent Tafel slope", f"{r.tafel_slope:.1f} mV/dec",
                  help="LSV-derived apparent slope — intermediate coverage varies with E; confirm with steady-state data.")
        c3.metric("j₀", f"{r.exchange_current_density:.3e} mA/cm²")
        # dual E_onset (threshold + zero-cross) -- most useful on net current
        _ot = getattr(r, "e_onset_threshold", float("nan"))
        _oz = getattr(r, "e_onset_zerocross", float("nan"))
        if not (math.isnan(_ot) and math.isnan(_oz)):
            _rhe_add = e_ref_val + 0.059 * ph_value
            co1, co2 = st.columns(2)
            co1.metric(
                "E_onset (j-threshold)",
                f"{_ot:.3f} V" if not math.isnan(_ot) else "N/A",
                help=(f"= {_ot + _rhe_add:.3f} V vs RHE" if not math.isnan(_ot)
                      else "first E where j reaches 0.01 mA/cm²"))
            co2.metric(
                "E_onset (zero-cross)",
                f"{_oz:.3f} V" if not math.isnan(_oz) else "N/A",
                help=(f"= {_oz + _rhe_add:.3f} V vs RHE" if not math.isnan(_oz)
                      else "first E where current turns sustained-positive"))
        c4, c5, c6 = st.columns(3)
        _ej10  = getattr(r, "e_at_j10",  float("nan"))
        _ej50  = getattr(r, "e_at_j50",  float("nan"))
        _ej100 = getattr(r, "e_at_j100", float("nan"))
        c4.metric("E @ 10 mA/cm²",
                  f"{_ej10:.3f} V" if not math.isnan(_ej10) else "N/A",
                  help="Potential at 10 mA/cm² on the rising branch (analysis frame).")
        c5.metric("E @ 50 mA/cm²",
                  f"{_ej50:.3f} V" if not math.isnan(_ej50) else "N/A")
        c6.metric("E @ 100 mA/cm²",
                  f"{_ej100:.3f} V" if not math.isnan(_ej100) else "N/A")
        if getattr(r, "eta_is_valid", False):
            c7, c8, c9 = st.columns(3)
            c7.metric("η @ 10 mA/cm²",
                      f"{r.overpotential_10 * 1000:.1f} mV" if not math.isnan(r.overpotential_10) else "N/A",
                      help="True overpotential vs the supplied equilibrium potential.")
            c8.metric("η @ 50 mA/cm²",
                      f"{r.overpotential_50 * 1000:.1f} mV" if not math.isnan(r.overpotential_50) else "N/A")
            c9.metric("η @ 100 mA/cm²",
                      f"{r.overpotential_100 * 1000:.1f} mV" if not math.isnan(r.overpotential_100) else "N/A")
        else:
            st.caption("η not shown: supply an equilibrium potential to report true overpotentials (η = E@j − E_eq).")
        if loading > 0:
            st.metric("Mass activity", f"{r.mass_activity:.3f} mA/mg_cat",
                      help=getattr(r, "activity_reference", "") or None)
        _sa_unit = "cm²_BET" if catalyst_type == "carbon_material" else "cm²_Pt"
        if ecsa > 0:
            st.metric("Specific activity", f"{r.specific_activity:.4f} mA/{_sa_unit}",
                      help=getattr(r, "activity_reference", "") or None)
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
        _scale = st.radio("Potential scale", [f"vs {e_ref_type}", "vs RHE"],
                          horizontal=True, key="lsv_scale")
        _use_rhe = (_scale == "vs RHE")
        _off = _rhe_offset if _use_rhe else 0.0
        if actual_rs > 0:
            from eisforge.analysis.lsv_analyzer import LSVAnalyzer
            cur_a_lsv = st.session_state["lsv_cur"] / 1000.0
            p_lsv = LSVAnalyzer.apply_ir_compensation(
                st.session_state["lsv_pot"], cur_a_lsv, actual_rs)
            p_lsv = p_lsv + _off
        else:
            p_lsv = st.session_state["lsv_pot"] + _off
        if st.session_state.get("lsv_blank_active") and "lsv_raw_alc_cur" in st.session_state:
            # three-curve overlay: raw alcohol, blank, net (green = analysed)
            _rap = st.session_state["lsv_raw_alc_pot"] + _off
            _rac = st.session_state["lsv_raw_alc_cur"] / area if area > 0 else st.session_state["lsv_raw_alc_cur"]
            _rbp = st.session_state["lsv_raw_blk_pot"] + _off
            _rbc = st.session_state["lsv_raw_blk_cur"] / area if area > 0 else st.session_state["lsv_raw_blk_cur"]
            fig.add_trace(go.Scatter(x=_rap, y=_rac, mode="lines",
                                     name="Alcohol (raw)",
                                     line=dict(color="#2563eb", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=_rbp, y=_rbc, mode="lines",
                                     name="Blank (electrolyte)",
                                     line=dict(color="#9ca3af", width=1.5, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_lsv, y=j_lsv, mode="lines",
                                     name="Net (alcohol - blank)",
                                     line=dict(color="#059669", width=2.5)), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=p_lsv, y=j_lsv, mode="lines",
                                     name="LSV" + ("+iR-corr." if actual_rs > 0 else ""),
                                     line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig.add_vline(x=r.e_onset + _off, line_dash="dash", line_color="#d97706",
                      annotation_text=f"E_onset={r.e_onset + _off:.3f}V",
                      annotation_font=dict(color="#d97706"), row=1, col=1)
        if tafel_mode_lsv == "Current window (mA/cm²)":
            mask = (j_lsv > 0) & (j_lsv >= tj_min) & (j_lsv <= tj_max)
        else:
            # r.tafel_region is in the analyzer's internal frame
            # (raw_potential + e_ref_val). Convert to the CURRENT display frame
            # (p_lsv), which is raw_potential + _off, so the markers line up:
            #   display = internal - e_ref_val + _off
            _treg_lo = r.tafel_region[0] - e_ref_val + _off
            _treg_hi = r.tafel_region[1] - e_ref_val + _off
            mask = (j_lsv > 0) & (p_lsv >= _treg_lo) & (p_lsv <= _treg_hi)
        if np.sum(mask) > 3:
            fig.add_trace(go.Scatter(x=np.log10(j_lsv[mask]), y=p_lsv[mask],
                                     mode="markers", name="Tafel region",
                                     marker=dict(color="#7c3aed", size=6)), row=1, col=2)
        title = f"LSV — {sr_lsv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}"
        if actual_rs > 0:
            title += " | iR-corrected"
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title=title)
        fig.update_xaxes(title_text=("Potential (V vs RHE)" if _use_rhe else f"Potential (V vs {e_ref_type})"), row=1, col=1)
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
                    _el_b = ElectrolyteInfo(
                        media=ekey, compound=elec_compound_key, concentration=elec_conc)
                    pots_b, curs_b = [], []
                    for f in batch_lsv_files:
                        p, c, _ = load_cv_lsv(
                            f,
                            unit_factor=1.0 if Path(f.name).suffix.lower() == ".idf" else unit_factor,
                        )
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
        c4.metric(
            f"η@10 ({n_str})",
            f"{blr.eta10_mean * 1000:.1f} mV" if not math.isnan(blr.eta10_mean) else "N/A",
            delta=f"± {blr.eta10_std * 1000:.1f} mV" if not math.isnan(blr.eta10_std) else None,
        )
        c5.metric(
            f"η@50 ({n_str})",
            f"{blr.eta50_mean * 1000:.1f} mV" if not math.isnan(blr.eta50_mean) else "N/A",
            delta=f"± {blr.eta50_std * 1000:.1f} mV" if not math.isnan(blr.eta50_std) else None,
        )
        c6.metric(
            f"η@100 ({n_str})",
            f"{blr.eta100_mean * 1000:.1f} mV" if not math.isnan(blr.eta100_mean) else "N/A",
            delta=f"± {blr.eta100_std * 1000:.1f} mV" if not math.isnan(blr.eta100_std) else None,
        )
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
        st.markdown("##### Nyquist shape hints (optional)")
        st.caption(
            "Set these if the low-frequency arc curls below the axis "
            "(4th quadrant) or crosses into negative Re(Z) — the definitive "
            "AOR kinetic fingerprint of adsorbed-intermediate relaxation."
        )
        hint_c1, hint_c2 = st.columns(2)
        inductive_hint = hint_c1.checkbox("Low-f inductive loop (4th quadrant)", value=False)
        ndr_hint = hint_c2.checkbox("Low-f arc crosses Re(Z) < 0 (NDR)", value=False)
        use_bounds = st.checkbox("Use smart bounds (custom circuit only)", value=False)
        st.caption("💡 Tip: After fit, use R0 value as R_s for iR compensation in CV/LSV tabs.")
    with col2:
        if eis_file:
            try:
                fr, zr, zi, meta_eis = load_eis(eis_file)
                st.success(f"✅ {len(fr)} points | {eis_file.name}")
                st.session_state.update({"eis_fr": fr, "eis_zr": zr, "eis_zi": zi,
                                         "eis_filename": eis_file.name})
            except Exception as e:
                st.error(f"Error loading EIS: {e}")

    if "eis_fr" in st.session_state:
        fr = st.session_state["eis_fr"]
        zr = st.session_state["eis_zr"]
        zi = st.session_state["eis_zi"]
        import plotly.graph_objects as go
        fig_eis = go.Figure()
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
            ))
        fig_eis.update_layout(**PLOTLY_LAYOUT, title="Nyquist Plot",
                              xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)")
        st.plotly_chart(fig_eis, use_container_width=True)

        if st.button("🔍 Suggest equivalent circuits", type="primary"):
            try:
                from eisforge.catalogs.suggestion_engine import suggest_circuits
                with st.spinner("Fitting candidate circuits and ranking by AICc..."):
                    suggestions = suggest_circuits(
                        fr, zr, zi,
                        catalyst_type=catalyst_type,
                        inductive_loop=inductive_hint,
                        negative_resistance=ndr_hint,
                    )
                st.session_state["eis_suggestions"] = suggestions
            except Exception as e:
                st.error(f"Suggestion error: {e}")

        if "eis_suggestions" in st.session_state:
            suggestions = st.session_state["eis_suggestions"]
            st.markdown("#### Suggested equivalent circuits (ranked by AICc)")
            st.caption(
                "ΔAICc < 2: essentially equivalent support · 2–7: less support · "
                ">10: no support (Burnham & Anderson convention). "
                "Pick the SIMPLEST model with strong support, not just the top row."
            )
            sugg_df = pd.DataFrame({
                "Circuit": [s.model.name for s in suggestions],
                "Notation": [s.model.notation for s in suggestions],
                "k (params)": [s.n_params for s in suggestions],
                "AICc": [f"{s.aicc:.2f}" if np.isfinite(s.aicc) else "—" for s in suggestions],
                "ΔAICc": [f"{s.delta_aicc:.2f}" if s.converged else "—" for s in suggestions],
                "Support": [s.support_label() for s in suggestions],
            })
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)

            converged_labels = [str(s) for s in suggestions if s.converged]
            if converged_labels:
                choice_label = st.selectbox("Accept a suggestion", converged_labels)
                choice = next(s for s in suggestions if s.converged and str(s) == choice_label)
                with st.expander("Why this circuit? (rationale)"):
                    st.write(choice.model.rationale)
                _suggested_circuit = choice.model.notation
                _suggested_p0 = ", ".join(
                    f"{v:.4e}" for v in choice.fit_result.parameters.values()
                )
            else:
                st.warning("No candidate converged — falling back to manual entry.")
                _suggested_circuit = "R0-p(R1,CPE1)"
                _suggested_p0 = "30, 31000, 2e-7, 0.78"
        else:
            _suggested_circuit = "R0-p(R1,CPE1)"
            _suggested_p0 = "30, 31000, 2e-7, 0.78"

        if "lit" in st.session_state and st.session_state["lit"].system_found:
            g = st.session_state["lit"]
            _suggested_circuit = g.recommended_circuit
            _suggested_p0 = ", ".join(f"{v:.3e}" for v in g.initial_guess.values())

        circ = st.text_input("Equivalent circuit (edit or accept suggestion)", value=_suggested_circuit)
        p0s = st.text_input(
            "Initial guess (comma-separated)",
            value=_suggested_p0 if _suggested_p0 else "30, 31000, 2e-7, 0.78",
        )

        kk_threshold = st.number_input(
            "K-K residual threshold",
            value=0.005, min_value=0.0001, max_value=0.5, step=0.0005,
            format="%.4f",
        )
        st.caption(
            "K-K residual threshold as a fraction of |Z| (Schönleber 2014 "
            "convention: 0.005 = 0.5%). Failing K-K warns but does not block "
            "the fit."
        )

        if st.button("▶ Run CNLS Fit", type="primary"):
            try:
                from eisforge.core.fitter import CNLSFitter
                from eisforge.parsers.base_parser import EISDataset
                from eisforge.core.validators import KramersKronigValidator
                p0_list = [float(x.strip()) for x in p0s.split(",")]
                bounds = smart_bounds(circ, p0_list) if use_bounds else None
                fitter = CNLSFitter(circuit_string=circ, initial_guess=p0_list,
                                    bounds=bounds, allow_negative_r=ndr_hint,
                                    robust=True)
                ds = EISDataset(frequency=fr, z_real=zr, z_imag=zi, metadata={})
                st.session_state["kk_result"] = KramersKronigValidator(
                    residual_threshold=kk_threshold).validate(ds)
                st.session_state["kk_threshold_used"] = kk_threshold
                fit_r = fitter.fit(ds)
                st.session_state["eis_fit"] = fit_r
                if fit_r.converged:
                    st.success(f"✅ Fit converged | reduced χ² = {fit_r.chi_squared:.4e}")
                else:
                    st.warning(f"⚠ Fit did not converge | reduced χ² = {fit_r.chi_squared:.4e}")
            except Exception as e:
                st.error(f"Fit error: {e}")

        if "eis_fit" in st.session_state:
            fit_r = st.session_state["eis_fit"]
            st.markdown("#### Fit Parameters")
            param_df = pd.DataFrame({
                "Parameter": list(fit_r.parameters.keys()),
                "Value": list(fit_r.parameters.values()),
                "Std Error": [fit_r.parameter_errors.get(k, float("nan"))
                             for k in fit_r.parameters.keys()],
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            # ── Kramers-Kronig verdict (diagnostic; does not block fitting) ──
            _kk = st.session_state.get("kk_result")
            if _kk is not None:
                _thr = st.session_state.get("kk_threshold_used", kk_threshold)
                if abs(_thr - kk_threshold) > 1e-12:
                    st.info(
                        "The verdict shown was computed at a K-K threshold of "
                        f"{_thr:.4f} — re-run the fit to apply the current "
                        f"threshold ({kk_threshold:.4f})."
                    )
                _kk_css = "val-ok" if _kk.passed else "val-warn"
                _kk_icon = "✅" if _kk.passed else "⚠"
                st.markdown(
                    f'<div class="{_kk_css}">{_kk_icon} {_kk.summary()}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"K-K method used: {_kk.method} — linKK = linear K-K "
                    "(Schönleber 2014); voigt = Voigt-circuit fallback; "
                    "unavailable = no solver; not_run = fewer than 10 points."
                )
                if _kk.warning_message:
                    st.markdown(
                        f'<div class="{_kk_css}">{_kk.warning_message}</div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("K-K residuals (real & imaginary vs frequency)"):
                    import plotly.graph_objects as _go
                    _fig_kk = _go.Figure()
                    _fig_kk.add_trace(_go.Scatter(
                        x=fr, y=_kk.residuals_real, mode="lines+markers",
                        name="real", line=dict(width=1)))
                    _fig_kk.add_trace(_go.Scatter(
                        x=fr, y=_kk.residuals_imag, mode="lines+markers",
                        name="imag", line=dict(width=1)))
                    _fig_kk.add_hline(y=_thr, line_dash="dash", line_color="red")
                    _fig_kk.add_hline(y=-_thr, line_dash="dash", line_color="red")
                    _fig_kk.update_layout(
                        **PLOTLY_LAYOUT, title="K-K residuals",
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Residual (rel.)", xaxis_type="log",
                    )
                    st.plotly_chart(_fig_kk, use_container_width=True)

            # ── Physical interpretation (rule-based, deterministic) ─────────
            st.markdown("#### Physical Interpretation")
            try:
                from eisforge.analysis.eis_interpreter import interpret_fit
                interp = interpret_fit(
                    fit_r.parameters, fit_r.circuit_string, fit_r.chi_squared
                )
                st.markdown(interp.as_markdown())
                # ── C_dl estimate from EIS (per geometric area) ─────────────
                if area and area > 0:
                    _cdl_lines = [
                        f"- {p.label}: C_eff/A ≈ "
                        f"{p.c_eff / area * 1e6:.1f} μF/cm²"
                        for p in interp.processes if p.c_eff is not None
                    ]
                    if _cdl_lines:
                        st.markdown("**Specific capacitance (per geometric area):**")
                        st.markdown("\n".join(_cdl_lines))
                        st.caption(
                            "The C_eff of an arc equals C_dl only if the spectrum "
                            "was measured at a NON-faradaic potential; at reaction "
                            "potentials it contains pseudocapacitive/adsorption "
                            "contributions."
                        )
            except Exception as e:
                st.warning(f"Interpretation unavailable: {e}")

            # ── Excel export ────────────────────────────────────────────────
            try:
                import io as _io
                _buf = _io.BytesIO()
                with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
                    _kk_exp = st.session_state.get("kk_result")
                    _kk_thr_exp = st.session_state.get("kk_threshold_used",
                                                       kk_threshold)
                    if _kk_exp is not None:
                        _maxpct = _kk_exp.residuals_max_pct
                        _kk_vals = [
                            "yes" if _kk_exp.passed else "no",
                            _maxpct if np.isfinite(_maxpct) else "n/a",
                            _kk_exp.n_rc_elements, _kk_exp.mu, _kk_exp.method,
                        ]
                    else:
                        _kk_vals = ["n/a", "", "", "", ""]
                    pd.DataFrame({
                        "Item": ["Circuit", "Reduced chi2 (modulus-weighted)",
                                 "Converged", "Points used", "Outliers removed",
                                 "Source file", "K-K passed",
                                 "K-K max residual (%)", "K-K N_RC", "K-K mu",
                                 "K-K method", "K-K threshold"],
                        "Value": [fit_r.circuit_string, fit_r.chi_squared,
                                  fit_r.converged, fit_r.n_points_used,
                                  fit_r.n_outliers_removed,
                                  st.session_state.get("eis_filename", ""),
                                  *_kk_vals, _kk_thr_exp],
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
                    "📥 Download EIS results (Excel)",
                    data=_buf.getvalue(),
                    file_name="eisforge_eis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")


# ══════════════════ EIS-GPT ══════════════════════════════════════════════════
with tab4:
    st.markdown('<h3>🤖 EIS-GPT Interpreter</h3>', unsafe_allow_html=True)
    st.info(
        "Rule-based physical interpretation of the CNLS fit (deterministic, "
        "reviewable). The physics-informed transformer (EIS-GPT) is implemented "
        "but not yet trained — ML-based interpretation is planned for v0.4."
    )
    if st.button("🔍 Interpret EIS Spectrum"):
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
            st.warning("Run CNLS fit first (EIS Analysis tab).")


# ══════════════════ CORRELATION ══════════════════════════════════════════════
with tab5:
    st.markdown('<h3>🔗 EIS–CV Correlation</h3>', unsafe_allow_html=True)
    if "eis_fit" in st.session_state and "cv_r" in st.session_state:
        fit_r = st.session_state["eis_fit"]
        cv_r = st.session_state["cv_r"]
        st.markdown("#### Parameter Correlation Table")
        import re as _re
        _params = fit_r.parameters
        _r_items = sorted(
            ((int(m.group(1)), v) for n, v in _params.items()
             if (m := _re.fullmatch(r"R(\d+)", n))),
        )
        _r_s_val = _r_items[0][1] if _r_items else "N/A"
        _faradaic = [v for _, v in _r_items[1:]]
        # dominant charge-transfer resistance = largest |R| among faradaic arcs
        _r_ct_val = max(_faradaic, key=abs) if _faradaic else "N/A"
        corr_data = {
            "R_s (Ω)": [_r_s_val],
            "R_ct (Ω)": [_r_ct_val],
            "E_onset (V)": [cv_r.e_onset],
            "j_f (mA/cm²)": [cv_r.j_forward_peak],
        }
        st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)
    else:
        st.info("Complete both EIS fit (tab 3) and CV analysis (tab 1) to see correlations.")


# ══════════════════ ECSA ANALYSIS ════════════════════════════════════════════
with tab6:
    st.markdown('<h3>🧪 ECSA Calculation</h3>', unsafe_allow_html=True)
    st.info(
        "Electrochemically Active Surface Area from CV: H-UPD (Pt/Pd), "
        "CO stripping (PtRu/PtSn/Pd), or the double-layer capacitance (Cdl) "
        "method for carbon materials."
    )
    st.caption(
        "Potentials are converted to V vs RHE using the sidebar reference "
        "electrode and pH before analysis. Currents are converted to A from the "
        "sidebar current unit. Loading (mg) comes from the sidebar (µg mass or "
        "mg/cm² × area)."
    )
    _ecsa_method = st.radio(
        "ECSA method",
        ["Auto (catalyst-aware)", "H-UPD (Pt/Pd)", "CO stripping", "Cdl (multi-scan-rate CV)"],
        horizontal=True,
        key="ecsa_method",
        help="Auto routes by sidebar catalyst type: noble metal → H-UPD, "
             "alloy (PtRu/PtSn) → CO stripping, carbon → Cdl.",
    )

    _ecsa_vr = st.text_input(
        "Integration window v_range (V vs RHE, comma-separated)",
        value="0.05,0.40",
        key="ecsa_v_range",
        help="H-UPD window (0.05–0.40 V) or CO oxidation window. For Cdl: "
             "the non-Faradaic window; v_mid = midpoint.",
    )
    try:
        _vlo, _vhi = (float(x.strip()) for x in _ecsa_vr.split(","))
        _v_range = (_vlo, _vhi)
    except Exception:
        _v_range = (0.05, 0.40)

    _qref_in = st.number_input(
        "q_ref override (µC/cm², 0 = method default)",
        value=0.0, min_value=0.0, step=10.0, key="ecsa_q_ref",
        help="Defaults: 210 (Pt H-UPD), 420 (Pt CO stripping). Pd: 212/424. "
             "PtRu 1:1 CO ≈ 300. See ECSACalculator docstring.",
    )
    _q_ref = _qref_in if _qref_in > 0 else None

    _is_cdl = _ecsa_method.startswith("Cdl")
    _is_auto = _ecsa_method.startswith("Auto")

    if _is_auto:
        # Catalyst-aware routing: noble → H-UPD, alloy → CO stripping,
        # carbon → Cdl (multi-file). Matches ECSACalculator.auto_ecsa routing.
        _auto_target = (
            "Cdl" if catalyst_type == "carbon_material"
            else "CO stripping" if catalyst_type == "alloy"
            else "H-UPD"
        )
        if catalyst_type == "metal_oxide":
            st.error(
                "Auto cannot choose a method for metal oxides (RuO₂, MnO₂, "
                "Co₃O₄): they are pseudo-capacitive and no automatic dispatch "
                "is safe. **Select the Cdl method manually** and treat the "
                "result as a relative comparison only."
            )
            _auto_target = "Cdl"
        st.caption(f"Auto routing: **{_auto_target}** (catalyst_type={catalyst_type})")
        _is_cdl = _auto_target == "Cdl"
        _eff_method = _auto_target
        _auto_blocked = catalyst_type == "metal_oxide"
    else:
        _eff_method = _ecsa_method
        _auto_blocked = False

    if _is_cdl:
        st.caption(
            "**Caveat:** the Cdl method requires a genuinely non-faradaic "
            "potential window and a flat, well-defined double-layer region. On "
            "rough or porous electrodes (carbon paste, thick catalyst films) "
            "the Cdl method is unreliable and the resulting ECSA can be "
            "meaningless."
        )
        ecsa_files = st.file_uploader(
            "Upload CV files at different scan rates",
            type=CV_FORMATS, accept_multiple_files=True, key="ecsa_cdl_up",
        )
        sr_list = st.text_input(
            "Scan rates (mV/s, comma-separated, same order as files)",
            value="20,50,100,200", key="ecsa_scan_rates",
        )
        _cs_in = st.number_input(
            "cs specific capacitance (mF/cm², 0 = carbon default 0.035)",
            value=0.0, min_value=0.0, step=0.001, format="%.4f",
            key="ecsa_cs",
        )
        st.caption(
            "Default 0 = **CS_CARBON = 0.035 mF/cm²** (porous carbon / CNT / "
            "graphene). ECSA scales linearly with cs — halving cs doubles the "
            "answer. The module cites **no literature source** for this "
            "constant and states no electrolyte; treat it as an assumption to "
            "verify for your material and electrolyte."
        )
        _cs = _cs_in if _cs_in > 0 else None
    else:
        ecsa_file = st.file_uploader(
            "Upload CV file", type=CV_FORMATS, key="ecsa_cv_up",
        )
        sr_ecsa = st.number_input("Scan rate (mV/s)", value=50, min_value=1,
                                  key="ecsa_scan_rate")

    _run_ecsa = st.button("▶ Run ECSA Analysis", type="primary", key="ecsa_run")

    if _run_ecsa:
        try:
            from eisforge.analysis.ecsa_calculator import ECSACalculator

            _load_ecsa = []
            if _is_cdl:
                if not ecsa_files:
                    raise ValueError("Upload at least 3 CV files for Cdl.")
                rates = [float(x.strip()) for x in sr_list.split(",")]
                if len(rates) != len(ecsa_files):
                    raise ValueError(
                        f"{len(ecsa_files)} files but {len(rates)} scan rates."
                    )
                for _f in ecsa_files:
                    if Path(_f.name).suffix.lower() == ".idf":
                        import hashlib as _hl2
                        _b = _f.getvalue()
                        _p, _c, _m = _parse_idf_cached(
                            _hl2.md5(_b).hexdigest(), _b, cycle_idx=-1)
                    else:
                        _p, _c, _m = load_cv_lsv(_f, unit_factor=unit_factor)
                    _load_ecsa.append((_p + e_ref_val + 0.059 * ph_value,
                                       _c / 1000.0))
                _sr_Vs = [r / 1000.0 for r in rates]
                if _is_auto and not _auto_blocked:
                    res_ecsa = ECSACalculator.auto_ecsa(
                        catalyst_type=catalyst_type,
                        potential=np.zeros(1), current=np.zeros(1),
                        scan_rate=1.0, loading_mg=_mass_mg, area_cm2=area,
                        potentials_list=[p for p, _ in _load_ecsa],
                        currents_list=[c for _, c in _load_ecsa],
                        scan_rates_list=_sr_Vs, v_range=_v_range,
                        cs_mF_cm2=_cs,
                    )
                else:
                    res_ecsa = ECSACalculator.method_c_cdl(
                        potentials_list=[p for p, _ in _load_ecsa],
                        currents_list=[c for _, c in _load_ecsa],
                        scan_rates=_sr_Vs, v_range=_v_range, cs_mF_cm2=_cs,
                        loading_mg=_mass_mg, area_cm2=area,
                    )
            else:
                if not ecsa_file:
                    raise ValueError("Upload a CV file.")
                if Path(ecsa_file.name).suffix.lower() == ".idf":
                    import hashlib as _hl2
                    _b = ecsa_file.getvalue()
                    _p, _c, _m = _parse_idf_cached(
                        _hl2.md5(_b).hexdigest(), _b, cycle_idx=-1)
                else:
                    _p, _c, _m = load_cv_lsv(ecsa_file, unit_factor=unit_factor)
                _p_rhe = _p + e_ref_val + 0.059 * ph_value
                _c_a = _c / 1000.0
                _sr_Vs = float(sr_ecsa) / 1000.0
                if _is_auto:
                    res_ecsa = ECSACalculator.auto_ecsa(
                        catalyst_type=catalyst_type,
                        potential=_p_rhe, current=_c_a, scan_rate=_sr_Vs,
                        loading_mg=_mass_mg, area_cm2=area,
                        v_range=_v_range, q_ref=_q_ref,
                    )
                elif _ecsa_method.startswith("CO"):
                    res_ecsa = ECSACalculator.method_b_co(
                        potential=_p_rhe, current=_c_a, scan_rate=_sr_Vs,
                        loading_mg=_mass_mg, v_range=_v_range, q_ref=_q_ref,
                        area_cm2=area, catalyst=catalyst,
                    )
                else:
                    res_ecsa = ECSACalculator.method_a_hupd(
                        potential=_p_rhe, current=_c_a, scan_rate=_sr_Vs,
                        loading_mg=_mass_mg, v_range=_v_range, q_ref=_q_ref,
                        area_cm2=area, catalyst=catalyst,
                    )
            st.session_state["ecsa_r"] = res_ecsa
            st.session_state["ecsa_method_ui"] = _ecsa_method
            st.success(
                f"✅ {res_ecsa['method']}: ECSA = {res_ecsa['ecsa_cm2']:.4f} cm²"
            )
        except Exception as e:
            st.error(f"ECSA error: {e}")

    if "ecsa_r" in st.session_state:
        res_ecsa = st.session_state["ecsa_r"]
        st.divider()
        st.markdown("#### ECSA Result — full output")
        st.caption(
            f"Method selected: **{st.session_state['ecsa_method_ui']}** | "
            f"resolved: **{res_ecsa['method']}** | "
            f"loading = {_mass_mg:.4f} mg | area = {area:.4f} cm²"
        )
        _dict_df = pd.DataFrame({
            "Key": list(res_ecsa.keys()),
            "Value": [str(v) for v in res_ecsa.values()],
        })
        st.dataframe(_dict_df, use_container_width=True, hide_index=True)

        _kka_vals = {
            "H-UPD": ("charge_uC", "µC"),
            "CO stripping": ("charge_uC", "µC"),
            "Cdl": ("cdl_mF_cm2", "mF/cm²"),
        }
        if res_ecsa["method"] in _kka_vals:
            _ck, _cu = _kka_vals[res_ecsa["method"]]
            cA, cB, cC = st.columns(3)
            cA.metric(f"{_ck} ({_cu})", f"{res_ecsa[_ck]:.4g}")
            cB.metric("ECSA (cm²)", f"{res_ecsa['ecsa_cm2']:.4f}")
            cC.metric("Specific ECSA (cm²/mg)",
                      f"{res_ecsa['specific_ecsa_cm2_mg']:.4f}")
            if res_ecsa["method"] == "Cdl":
                dA, dB = st.columns(2)
                dA.metric("R² (Cdl fit)", f"{res_ecsa['r_squared']:.4f}")
                dB.metric("cs used (mF/cm²)", f"{res_ecsa['cs_used_mF_cm2']:.4f}")

        # ── Excel export ──────────────────────────────────────────────────
        try:
            import io as _io3
            _buf3 = _io3.BytesIO()
            with pd.ExcelWriter(_buf3, engine="openpyxl") as _xw:
                _dict_df.to_excel(_xw, sheet_name="Result", index=False)
                _inputs = {
                    "Method selected": st.session_state["ecsa_method_ui"],
                    "Method used": res_ecsa["method"],
                    "Electrode area (cm²)": area,
                    "Scan rate(s) (mV/s)":
                        (sr_list if _is_cdl else sr_ecsa) if _ecsa_method
                        else "",
                    "Loading (mg)": _mass_mg,
                    "Current unit": current_unit,
                    "v_range (V vs RHE)": _v_range,
                }
                if res_ecsa["method"] == "Cdl":
                    _inputs["Caveat"] = (
                        "Cdl-derived ECSA assumes a genuinely non-faradaic "
                        "window and a well-defined double-layer region; "
                        "unreliable on rough/porous electrodes."
                    )
                pd.DataFrame({
                    "Item": list(_inputs.keys()),
                    "Value": [str(v) for v in _inputs.values()],
                }).to_excel(_xw, sheet_name="Inputs", index=False)
            st.download_button(
                "📥 Download ECSA results (Excel)",
                data=_buf3.getvalue(),
                file_name="eisforge_ecsa_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ecsa_download",
            )
        except Exception as e:
            st.warning(f"Excel export unavailable: {e}")


# ══════════════════ K-L ANALYSIS ═════════════════════════════════════════════
with tab7:
    st.markdown('<h3>⚗️ Koutecký–Levich Analysis</h3>', unsafe_allow_html=True)
    st.info("Upload LSV files at different rotation speeds (rpm) for K-L analysis.")
    st.caption("Potentials are converted to V vs RHE using the sidebar reference "
               "electrode and pH before analysis.")
    diff_species = st.radio(
        "Diffusing species",
        ["O2 (ORR)", "Alcohol (AOR)"],
        index=0,
        help="What actually diffuses to the electrode surface. n_electrons is "
             "only meaningful for the species selected here.",
    )
    if diff_species == "O2 (ORR)":
        oc1, oc2, oc3 = st.columns(3)
        D_O2 = oc1.number_input("D (O₂, cm²/s)", value=1.9e-5, min_value=0.0,
                                format="%.2e")
        nu_kl = oc2.number_input("ν (electrolyte, cm²/s)", value=1.0e-2,
                                 min_value=0.0, format="%.2e")
        C_O2 = oc3.number_input("C (O₂, mol/cm³)", value=1.2e-6, min_value=0.0,
                                format="%.2e")
        st.caption("O₂ defaults are literature values for O2-saturated 0.1 M KOH "
                   "(~25 °C, 1 atm). D, ν, C are electrolyte- and "
                   "temperature-dependent — adjust for acid media or other "
                   "temperatures.")
    else:
        kc1, kc2, kc3 = st.columns(3)
        from eisforge.analysis.koutecky_levich import _DEFAULT_DIFFUSION
        kl_alcohol = kc1.selectbox(
            "Alcohol", list(_DEFAULT_DIFFUSION.keys()), index=1,
            help="D lookup keys are these exact names — free text would silently "
                 "hit the 1.0e-5 cm²/s fallback.",
        )
        kl_electrolyte = kc2.text_input("Electrolyte", value="KOH")
        kl_conc = kc3.number_input("Concentration (M)", value=1.0, min_value=0.0)
        if kl_electrolyte.strip().lower() != "koh":
            st.warning(
                "Built-in diffusion coefficients are for 0.1 M KOH at 25 °C "
                "(our own data is 1 M H₂SO₄). Reported n_electrons assumes those "
                "conditions — verify D and ν for this electrolyte."
            )
    kl_files = st.file_uploader(
        "Upload LSV files at different rotation speeds",
        type=CV_FORMATS, accept_multiple_files=True, key="kl_up"
    )
    rpms_str = st.text_input("Rotation speeds (rpm, comma-separated)", value="400,900,1600,2500")
    if kl_files and st.button("▶ Run K-L Analysis", type="primary"):
        try:
            from eisforge.analysis.koutecky_levich import KLAnalyzer
            rpms = [float(x.strip()) for x in rpms_str.split(",")]
            _nernst = (8.314 * (273.15 + temperature) / 96485.0) * math.log(10)
            pots_kl, curs_kl = [], []
            for f in kl_files:
                p, c, _ = load_cv_lsv(f, unit_factor=unit_factor)
                pots_kl.append(p + e_ref_val + _nernst * ph_value)
                curs_kl.append(c)
            if diff_species == "O2 (ORR)":
                kla = KLAnalyzer(D_cm2_s=D_O2, nu_cm2_s=nu_kl, C_mol_cm3=C_O2,
                                 temperature_C=temperature)
            else:
                kla = KLAnalyzer(alcohol=kl_alcohol, electrolyte=kl_electrolyte,
                                 concentration_M=kl_conc, temperature_C=temperature)
            klr = kla.analyze(rotation_speeds_rpm=rpms, potentials=pots_kl,
                              currents=curs_kl, electrode_area=area)
            st.session_state["kl_r"] = klr
            best = klr.best_result
            st.success(f"✅ n_electrons = {klr.mean_n_electrons:.2f} | "
                       f"j_k = {best.j_kinetic:.4f} mA/cm²")
        except Exception as e:
            st.error(f"K-L error: {e}")
    if "kl_r" in st.session_state:
        klr = st.session_state["kl_r"]
        best = klr.best_result
        import plotly.graph_objects as go
        inv_sqw = [1.0 / np.sqrt(rpm * 2 * np.pi / 60.0)
                   for rpm in best.rotation_speeds_rpm]
        inv_j = [1.0 / abs(j) for j in best.j_measured]
        _slope = 1.0 / best.levich_slope
        fig_kl = go.Figure()
        fig_kl.add_trace(go.Scatter(x=inv_sqw, y=inv_j, mode="markers", name="Data"))
        fig_kl.add_trace(go.Scatter(x=inv_sqw,
                                    y=[best.intercept + _slope * x for x in inv_sqw],
                                    mode="lines", name="K-L fit"))
        fig_kl.update_layout(**PLOTLY_LAYOUT, title="K-L Plot (best E)",
                             xaxis_title="ω⁻¹/² (rad/s)⁻¹/²",
                             yaxis_title="j⁻¹ (cm²/mA)")
        st.plotly_chart(fig_kl, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Electron transfer number (n)", f"{klr.mean_n_electrons:.2f}")
        c2.metric("Kinetic current density (j_k)", f"{best.j_kinetic:.4f} mA/cm²")


# ══════════════════ SCAN-RATE KINETICS ════════════════════════════════════════
with tab8:
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
                m = _re.search(r"([\d.]+)\s*mV", lab)
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
                st.success(f"✅ Loaded {len(data)} scan rates: "
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
                                    title="log(Ipa) vs log(ν)",
                                    xaxis_title="log ν (V/s)",
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
                    mode="lines", name=f"R²={res.rs_r2:.4f}",
                    line=dict(color="#dc2626", width=2)))
                fig_rs.update_layout(**PLOTLY_LAYOUT,
                                     title="Randles–Ševčík: Ipa vs √ν",
                                     xaxis_title="√ν (V/s)^½",
                                     yaxis_title="Ipa")
                st.plotly_chart(fig_rs, use_container_width=True)

                # findings + warnings
                st.markdown("#### Interpretation")
                for f_ in res.findings:
                    st.markdown(f"- {f_}")
                for w in res.warnings:
                    st.markdown(f"- ⚠ {w}")
                if res.diffusion_coeff is not None:
                    st.metric("Apparent D (cm²/s)",
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
                        "📥 Download scan-rate results (Excel)",
                        data=buf.getvalue(),
                        file_name="eisforge_scan_rate_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.warning(f"Excel export unavailable: {e}")

        except Exception as e:
            st.error(f"Scan-rate analysis error: {e}")


# ══════════════════ CHRONOAMPEROMETRY ═════════════════════════════════════════
with tab9:
    st.markdown('<h3>Chronoamperometry (i–t Stability)</h3>', unsafe_allow_html=True)
    st.caption(
        "Upload a single chronoamperometry file (Ivium .idf: time, current). "
        "Reports descriptive operational-stability metrics — current retention, "
        "steady-state current, and initial drop. No Cottrell/diffusion fit is "
        "performed (early decay is largely capacitive, not catalyst loss)."
    )

    ca_file = st.file_uploader("Upload CA file", type=["idf", "csv", "txt"],
                               key="ca_up")
    c1, c2 = st.columns(2)
    ca_per_area = c1.checkbox("Show current density (per area)", value=True)
    ca_area = c2.number_input("Geometric area (cm²)", value=0.07068583,
                              format="%.5f", min_value=1e-6,
                              disabled=not ca_per_area)

    if ca_file is not None:
        try:
            import numpy as _np
            import pandas as _pd
            from eisforge.analysis.ca_analyzer import analyze_ca

            # ── minimal CA reader: pull (time, current) columns ───────────────
            name = ca_file.name.lower()
            raw = ca_file.read()
            t = i = None
            if name.endswith(".idf"):
                text = raw.decode("latin-1", errors="ignore").splitlines()
                start = next((k for k, l in enumerate(text)
                              if l.strip() == "primary_data"), None)
                if start is None:
                    raise ValueError("No primary_data block found in IDF.")
                npts = int(text[start + 2].strip())
                rows = []
                for l in text[start + 3: start + 3 + npts]:
                    parts = l.split()
                    if len(parts) >= 2:
                        try:
                            rows.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                arr = _np.array(rows)
                t, i = arr[:, 0], arr[:, 1]
            else:
                import io as _io
                df = _pd.read_csv(_io.BytesIO(raw), sep=None, engine="python")
                t = _pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
                i = _pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
                keep = ~(_np.isnan(t) | _np.isnan(i))
                t, i = t[keep], i[keep]

            st.success(f"✅ Loaded {len(t)} points | {ca_file.name}")

            res = analyze_ca(t, i, area_cm2=float(ca_area),
                             per_area=bool(ca_per_area))

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.time, y=res.current, mode="lines",
                                     name="i–t", line=dict(color="#2563eb")))
            fig.update_layout(**PLOTLY_LAYOUT, title="Chronoamperometry",
                              xaxis_title="Time (s)",
                              yaxis_title=f"|Current| ({res.unit_label})")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Stability Metrics")
            st.markdown(res.as_markdown())

            # metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Current retention", f"{res.retention_pct:.1f}%")
            m2.metric("Steady-state current",
                      f"{res.i_steady:.3e} {res.unit_label}")
            m3.metric("Initial drop (60 s)", f"{res.initial_drop_pct:.1f}%")

            # Excel export
            try:
                import io as _io2
                buf = _io2.BytesIO()
                with _pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    _pd.DataFrame({
                        "Item": ["Duration (s)", "I_initial", "I_steady",
                                 "Retention (%)", "Initial drop 60s (%)",
                                 "Unit", "Source file"],
                        "Value": [res.duration_s, res.i_initial, res.i_steady,
                                  res.retention_pct, res.initial_drop_pct,
                                  res.unit_label, ca_file.name],
                    }).to_excel(xw, sheet_name="Summary", index=False)
                    _pd.DataFrame({
                        "Time_s": res.time,
                        f"Current_{res.unit_label.replace('/', '_per_')}": res.current,
                        "Current_raw_A": res.current_raw,
                    }).to_excel(xw, sheet_name="Data", index=False)
                    if res.retention_at:
                        _pd.DataFrame({
                            "Time_s": list(res.retention_at.keys()),
                            "Retention_pct": list(res.retention_at.values()),
                        }).to_excel(xw, sheet_name="Retention_vs_time", index=False)
                st.download_button(
                    "📥 Download CA results (Excel)",
                    data=buf.getvalue(),
                    file_name="eisforge_ca_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")

        except Exception as e:
            st.error(f"Chronoamperometry analysis error: {e}")
