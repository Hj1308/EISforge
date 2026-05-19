"""
EISForge — Standalone Preprocessing Demo
A focused single-page app showing the full Upload → Preprocess → Fit workflow.

Author: Hoda Jafari | May 2026
Run: streamlit run app_simple.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from eisforge.parsers.base_parser import EISDataset
from eisforge.core.preprocessor   import DataPreprocessor
from eisforge.core.fitter         import CNLSFitter
from eisforge.core.validators     import KramersKronigValidator

st.set_page_config(page_title="EISForge — Smart Fit", page_icon="⚡", layout="wide")
st.title("⚡ EISForge — Smart Preprocessing & Fitting")
st.caption("Upload → Clean → Fit | by Hoda Jafari, 2026")

# ── Sidebar: preprocessing controls ──────────────────────────────────────────
st.sidebar.header("🧹 Preprocessing Settings")

remove_inductive = st.sidebar.checkbox(
    "Remove inductive artifacts (Z'' < 0)",
    value=True,
    help="High-frequency cable inductance is not real electrochemistry.",
)
crop_low_freq = st.sidebar.checkbox(
    "Crop very low frequencies (f < f_min)",
    value=True,
)
f_min = st.sidebar.number_input(
    "f_min (Hz)", value=0.01, format="%.4g",
    disabled=not crop_low_freq,
)
remove_jumps = st.sidebar.checkbox(
    "Remove |Z| jumps",
    value=False,
    help="Single-point spikes from instrument glitches or contact issues.",
)
jump_threshold = st.sidebar.slider(
    "Jump threshold (%)", min_value=10, max_value=100, value=30, step=5,
    disabled=not remove_jumps,
)

st.sidebar.divider()
st.sidebar.header("🔧 Fit Settings")
circuit_str = st.sidebar.text_input("Circuit", value="R0-p(R1,CPE1)")
initial_str = st.sidebar.text_input(
    "Initial guess (comma-separated)",
    value="30, 31000, 2e-7, 0.78",
)


# ── File loading utilities ───────────────────────────────────────────────────

def _read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ["latin-1", "cp1252", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, sep=None,
                               engine="python", comment="#")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", errors="replace",
                       sep=None, engine="python", comment="#")


def load_eis_dataset(uploaded_file) -> EISDataset:
    """Load any supported EIS file into an EISDataset."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(uploaded_file.read())
        tmp = t.name
    try:
        if suffix == ".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            return AutolabIDFParser().parse(tmp)
        elif suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            return GamryParser().parse(tmp)
        else:
            df = _read_csv_safe(tmp)
            c  = df.columns.tolist()
            fr = df[c[0]].to_numpy(float)
            zr = df[c[1]].to_numpy(float)
            zi = df[c[2]].to_numpy(float)
            if zi.mean() < 0:
                zi = -zi
            return EISDataset(frequency=fr, z_real=zr, z_imag=zi)
    finally:
        os.unlink(tmp)


# ── Main workflow ────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload EIS file",
    type=["idf", "dta", "csv", "txt", "mpt"],
    help="Autolab (.idf), Gamry (.dta), BioLogic (.mpt), or CSV/TXT",
)

if uploaded is None:
    st.info("👆 Upload an EIS file to begin")
    st.stop()

# Load raw data
try:
    raw_dataset = load_eis_dataset(uploaded)
    n_raw = len(raw_dataset.frequency)
    st.success(f"✅ Loaded {n_raw} raw points from {uploaded.name}")
except Exception as e:
    st.error(f"Loading failed: {e}")
    st.stop()


# Apply preprocessing
clean_dataset = raw_dataset

if remove_inductive:
    clean_dataset = DataPreprocessor.remove_inductive_artifacts(clean_dataset, verbose=False)

if crop_low_freq:
    clean_dataset = DataPreprocessor.crop_frequencies(clean_dataset, f_min=f_min, verbose=False)

if remove_jumps:
    clean_dataset = DataPreprocessor.remove_z_jumps(
        clean_dataset, threshold_pct=float(jump_threshold), verbose=False,
    )

n_clean = len(clean_dataset.frequency)
n_removed = n_raw - n_clean


# ── Preprocessing summary ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Raw points",     n_raw)
col2.metric("Clean points",   n_clean)
col3.metric("Removed",        n_removed, delta=f"{n_removed/n_raw*100:.1f}%" if n_raw else "0%")


# ── Show raw vs clean Nyquist comparison ─────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig_prep = make_subplots(rows=1, cols=2, subplot_titles=("Raw Data", "After Preprocessing"))
fig_prep.add_trace(go.Scatter(
    x=raw_dataset.z_real, y=raw_dataset.z_imag,
    mode="markers", name="Raw", marker=dict(color="#94a3b8", size=8),
), row=1, col=1)
fig_prep.add_trace(go.Scatter(
    x=clean_dataset.z_real, y=clean_dataset.z_imag,
    mode="markers", name="Clean", marker=dict(color="#2563eb", size=8),
), row=1, col=2)
fig_prep.update_xaxes(title_text="Z' (Ω)")
fig_prep.update_yaxes(title_text="-Z'' (Ω)")
fig_prep.update_layout(
    template="plotly_white", height=400,
    showlegend=False, paper_bgcolor="#ffffff",
)
st.plotly_chart(fig_prep, use_container_width=True)


# ── Run fit ──────────────────────────────────────────────────────────────────
st.divider()

if st.button("🚀 Run Smart Fitting", type="primary"):
    try:
        p0 = [float(x.strip()) for x in initial_str.split(",")]
    except ValueError:
        st.error("Invalid initial guess. Use comma-separated numbers.")
        st.stop()

    # Auto-bounds: detect CPE_n params (last in each CPE group; values in 0-1)
    n_params = len(p0)
    lower = [0.0] * n_params
    upper = []
    for v in p0:
        if 0 < v <= 1:
            upper.append(1.0)
        else:
            upper.append(np.inf)
    bounds = (lower, upper)

    with st.spinner("Fitting in progress..."):
        try:
            fitter = CNLSFitter(circuit_str, p0, bounds=bounds, remove_outliers=False)
            # remove_outliers=False because we already preprocessed
            result = fitter.fit(clean_dataset)
        except Exception as e:
            st.error(f"Fit error: {e}")
            st.stop()

    # ── Report ──────────────────────────────────────────────────────────────
    if result.converged:
        st.success(f"✅ Fit converged | χ² = {result.chi_squared:.5f}")
    else:
        st.warning(f"⚠ Fit did not fully converge | χ² = {result.chi_squared:.4f}")

    # Parameter table
    table_data = []
    for name, value in result.parameters.items():
        err = result.parameter_errors.get(name, float("nan"))
        rel_err = abs(err / value) * 100 if value != 0 else float("nan")
        table_data.append({
            "Parameter": name,
            "Value": f"{value:.4e}",
            "±Error": f"{err:.2e}",
            "Rel. Error (%)": f"{rel_err:.2f}" if not np.isnan(rel_err) else "N/A",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Plot fit ────────────────────────────────────────────────────────────
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Nyquist Plot", "Bode Plot"))
    fig.add_trace(go.Scatter(
        x=clean_dataset.z_real, y=clean_dataset.z_imag,
        mode="markers", name="Measured",
        marker=dict(color="#2563eb", size=8),
    ), row=1, col=1)
    if result.z_fit is not None:
        fig.add_trace(go.Scatter(
            x=result.z_fit.real, y=-result.z_fit.imag,
            mode="lines", name=f"Fit (χ²={result.chi_squared:.4f})",
            line=dict(color="#dc2626", width=2, dash="dash"),
        ), row=1, col=1)
    zm = np.sqrt(clean_dataset.z_real**2 + clean_dataset.z_imag**2)
    fig.add_trace(go.Scatter(
        x=clean_dataset.frequency, y=zm,
        mode="markers+lines", name="|Z|",
        marker=dict(color="#16a34a", size=5),
        line=dict(color="#16a34a", width=1),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Z' (Ω)",            row=1, col=1)
    fig.update_yaxes(title_text="-Z'' (Ω)",          row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| (Ω)",       type="log", row=1, col=2)
    fig.update_layout(template="plotly_white", height=450, paper_bgcolor="#ffffff")
    st.plotly_chart(fig, use_container_width=True)
