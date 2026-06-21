"""
cv_app.py — Streamlit interface for alcohol-oxidation (AOR) CV analysis.

Run from the repository root (the folder that contains app.py and the eisforge/
package, with analyze_cv.py beside it):

    streamlit run cv_app.py

Upload an Ivium .idf (or .csv/.txt/Gamry .DTA) cyclic-voltammetry file, set the
electrode and electrolyte parameters in the sidebar, and read off E_onset,
current density, and mass activity — optionally with blank subtraction.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from eisforge.analysis.cv_analyzer import (
    CVAnalyzer, ElectrolyteInfo,
    CATALYST_METAL_FREE, CATALYST_NOBLE_METAL, CATALYST_ALLOY, CATALYST_METAL_OXIDE,
)
from analyze_cv import load_cv, _subtract_blank, _UNIT_MA

st.set_page_config(page_title="EISForge — CV / AOR", page_icon="⚡", layout="wide")

# ── Reference-electrode presets (potential vs SHE, V) ────────────────────────
REF_PRESETS = {
    "Ag/AgCl (sat. KCl)": 0.197,
    "Ag/AgCl (3 M KCl)": 0.210,
    "SCE (sat. calomel)": 0.241,
    "Hg/HgO (1 M KOH)": 0.098,
    "Already measured vs RHE": 0.0,
    "Custom…": None,
}
ELECTROLYTES = ["KOH", "NaOH", "Na2CO3", "NH3", "H2SO4", "HClO4", "HCl", "HNO3"]
CAT_TYPES = {
    "Metal-free (carbon / B4C / graphene)": CATALYST_METAL_FREE,
    "Noble metal (Pt/Pd/Au/Rh)": CATALYST_NOBLE_METAL,
    "Alloy (PtRu/PtSn…)": CATALYST_ALLOY,
    "Metal oxide (NiO/Co3O4…)": CATALYST_METAL_OXIDE,
}
ALKALINE = {"KOH", "NaOH", "Na2CO3", "NH3"}


def _save_tmp(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".dat"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return Path(tmp.name)


def _e_to_rhe(e_value, eref, media, conc):
    if media == "alkaline":
        ph = min(14 + math.log10(max(conc, 1e-6)), 14.0)
    else:
        ph = 0.0
    return e_value + eref + 0.059 * ph, ph


# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚡ EISForge — CV / Alcohol-Oxidation Analysis")
st.caption("Upload a cyclic-voltammetry file, set the electrode & electrolyte, "
           "and get E_onset, current density and mass activity.")

# ── Sidebar: parameters ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    cat_label = st.selectbox("Catalyst type", list(CAT_TYPES.keys()), index=0)
    catalyst_type = CAT_TYPES[cat_label]

    st.subheader("Electrolyte")
    compound = st.selectbox("Compound", ELECTROLYTES, index=4)  # default H2SO4
    conc = st.number_input("Concentration (mol/L)", value=1.0, min_value=0.0, step=0.1)
    media = "alkaline" if compound in ALKALINE else "acidic"
    st.caption(f"Medium: **{media}**")

    st.subheader("Reference electrode")
    ref_label = st.selectbox("Type", list(REF_PRESETS.keys()), index=0)
    if REF_PRESETS[ref_label] is None:
        eref = st.number_input("Custom E_ref vs SHE (V)", value=0.197, step=0.001, format="%.3f")
    else:
        eref = REF_PRESETS[ref_label]
        st.caption(f"E_ref vs SHE = **{eref:.3f} V**")

    st.subheader("Electrode & loading")
    diameter_mm = st.number_input("Disk diameter (mm)", value=3.0, min_value=0.0, step=0.5)
    area = math.pi * (diameter_mm / 20.0) ** 2 if diameter_mm > 0 else 1.0
    st.caption(f"Geometric area = **{area:.4f} cm²**")
    mass_ug = st.number_input("Deposited catalyst mass (µg)", value=5.0, min_value=0.0, step=1.0)
    mass_mg = mass_ug / 1000.0
    if mass_mg > 0:
        st.caption(f"Loading = **{mass_mg / area:.4f} mg/cm²**")
    bet = st.number_input("BET / ECSA area (cm², optional)", value=0.0, min_value=0.0, step=1.0)

    st.subheader("Analysis options")
    onset_method = st.selectbox("Onset method", ["tangent", "derivative", "threshold"], index=0)
    no_bg = st.checkbox("Disable capacitive-background subtraction", value=True,
                        help="Recommended ON for irreversible AOR waves")
    use_report_at = st.checkbox("Report current at a fixed potential", value=True,
                                help="Use when the wave has no closed peak in the window")
    report_at = st.number_input("…at potential (V, raw scale)", value=1.0, step=0.05) if use_report_at else None

# ── Main: uploads ─────────────────────────────────────────────────────────────
col_u1, col_u2 = st.columns(2)
with col_u1:
    sample_file = st.file_uploader("CV data file", type=["idf", "csv", "txt", "dta"])
with col_u2:
    use_blank = st.checkbox("Subtract a blank CV (electrolyte without alcohol)")
    blank_file = st.file_uploader("Blank CV file", type=["idf", "csv", "txt", "dta"],
                                  disabled=not use_blank)

if sample_file is None:
    st.info("⬆️ Upload a CV file to begin.")
    st.stop()

# ── Load + analyse ────────────────────────────────────────────────────────────
try:
    spath = _save_tmp(sample_file)
    e, cur, unit, meta = load_cv(spath, None, -1)

    if use_blank and blank_file is not None:
        bpath = _save_tmp(blank_file)
        e_b, cur_b, unit_b, _ = load_cv(bpath, None, -1)
        cur = _subtract_blank(e, cur * _UNIT_MA.get(unit, 1.0),
                              e_b, cur_b * _UNIT_MA.get(unit_b, 1.0))
        unit = "mA"
        blank_active = True
    else:
        blank_active = False

    scan_rate = meta.get("Scanrate", 0.05) * 1000.0  # V/s → mV/s

    analyzer = CVAnalyzer(
        scan_rate=scan_rate, electrode_area=area, ecsa=bet,
        onset_method=onset_method,
        electrolyte=ElectrolyteInfo(media=media, compound=compound, concentration=conc),
        catalyst_type=catalyst_type, current_unit=unit, catalyst_loading=mass_mg / area if area else 0.0,
        e_ref_vs_rhe=eref,
    )
    if no_bg or blank_active:
        analyzer._subtract_capacitive_background = (
            lambda potential, current_ma: (current_ma, np.zeros_like(current_ma))
        )
    res = analyzer.analyze(e, cur)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not analyse this file: {exc}")
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.success(f"Loaded **{sample_file.name}** — {len(e)} points · scan rate "
           f"{scan_rate:.0f} mV/s" + (" · blank subtracted" if blank_active else ""))

e_onset_rhe, ph = _e_to_rhe(res.e_onset, eref, media, conc)

cur_ma = cur * _UNIT_MA.get(unit, 1.0)
vertex = int(np.argmax(e))
i_fwd = cur_ma[: vertex + 1]
e_fwd = e[: vertex + 1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("E_onset (vs RHE)", f"{e_onset_rhe:.3f} V", help=f"{res.e_onset:.3f} V vs ref · pH≈{ph:.1f}")
m2.metric("E_onset (vs ref)", f"{res.e_onset:.3f} V", help=res.e_onset_method)

if report_at is not None:
    idx = int(np.argmin(np.abs(e_fwd - report_at)))
    i_at = float(i_fwd[idx])
    m3.metric(f"j @ {e_fwd[idx]:+.2f} V", f"{i_at / area:.2f} mA/cm²",
              help="geometric current density at the chosen potential")
    if mass_mg > 0:
        m4.metric(f"Mass act. @ {e_fwd[idx]:+.2f} V", f"{i_at / mass_mg:.0f} A/g")
else:
    m3.metric("Forward peak j", f"{res.j_forward_peak:.2f} mA/cm²")
    if mass_mg > 0:
        m4.metric("Mass activity", f"{res.i_forward_peak / mass_mg:.0f} A/g")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(e, cur_ma, color="#2563eb", lw=1.6, label="net" if blank_active else "CV")
ax.axvline(res.e_onset, color="#dc2626", ls="--", lw=1, label=f"E_onset = {res.e_onset:.3f} V")
if report_at is not None:
    ax.axvline(e_fwd[idx], color="#16a34a", ls=":", lw=1, label=f"report @ {e_fwd[idx]:+.2f} V")
ax.axhline(0, color="#94a3b8", lw=0.6)
ax.set_xlabel("E (V vs ref)")
ax.set_ylabel("I (mA)")
ax.set_title("Cyclic voltammogram")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
st.pyplot(fig)

# ── Full text summary ─────────────────────────────────────────────────────────
with st.expander("Full analysis summary", expanded=False):
    st.code(res.summary(), language="text")
