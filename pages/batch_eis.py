"""
EISForge — Batch EIS Page (patch19)
=====================================
Multi-file upload → warm-start CNLS fitting → trend plots.

Access via the Streamlit multi-page sidebar or directly:
    streamlit run app.py
and navigate to "Batch EIS" in the sidebar.
"""

import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eisforge.visualization.theme import (
    PLOTLY_LAYOUT, series_style,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_eis_bytes(file_bytes: bytes, filename: str):
    """Parse EIS from uploaded file bytes. Returns (freq, z_real, z_imag)."""
    import tempfile, os
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(file_bytes)
        tmp = t.name
    try:
        if suffix == ".idf":
            from eisforge.parsers.ivium_parser import IviumIDFParser
            ds = IviumIDFParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag
        elif suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            ds = GamryParser().parse(tmp)
            return ds.frequency, ds.z_real, ds.z_imag
        elif suffix in (".mpt", ".mpr"):
            from galvani import BioLogic
            mpr = BioLogic.MPRfile(tmp)
            df = mpr.DF
            return df["freq/Hz"].to_numpy(), df["Re(Z)/Ohm"].to_numpy(), df["-Im(Z)/Ohm"].to_numpy()
        else:
            import pandas as _pd
            df = _pd.read_csv(tmp, sep=None, engine="python",
                              encoding="latin-1", comment="#", skip_blank_lines=True)
            c = df.columns.tolist()
            fr = df[c[0]].to_numpy(float)
            zr = df[c[1]].to_numpy(float)
            zi = df[c[2]].to_numpy(float)
            return fr, zr, (zi if zi.mean() > 0 else -zi)
    finally:
        os.unlink(tmp)


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Batch EIS — EISForge", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
body,[data-testid="stAppViewContainer"]{font-family:'Plus Jakarta Sans',sans-serif;background:#fff;}
.title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#18162a;}
.chip{display:inline-block;padding:.15rem .5rem;border-radius:6px;font-size:.7rem;font-family:'JetBrains Mono',monospace;background:#f3eefe;color:#6d28d9;border:1px solid #e4dafb;margin:.1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">📦 Batch EIS Analysis — Warm-Start Fitting</h1>', unsafe_allow_html=True)
st.caption(
    "Upload a series of EIS files (e.g., different alcohol concentrations). "
    "Uses warm-start CNLS: best-fit parameters from spectrum *i* seed spectrum *i+1*, "
    "reducing convergence time by ~60–80 % and ensuring parameter continuity across the series."
)

st.divider()

# ── circuit setup ─────────────────────────────────────────────────────────────
col_cfg1, col_cfg2 = st.columns([2, 1])
with col_cfg1:
    PRESETS = {
        "Randles (R0-p(R1,CPE1))": {
            "circuit": "R0-p(R1,CPE1)",
            "p0": [2.0, 30.0, 1e-4, 0.85],
            "names": ["Rs", "Rct", "Q_CPE", "n_CPE"],
        },
        "Randles + Warburg (R0-p(R1,CPE1)-W1)": {
            "circuit": "R0-p(R1,CPE1)-W1",
            "p0": [2.0, 30.0, 1e-4, 0.85, 5.0],
            "names": ["Rs", "Rct", "Q_CPE", "n_CPE", "sigma_W"],
        },
        "Two time-constants (R0-p(R1,CPE1)-p(R2,CPE2))": {
            "circuit": "R0-p(R1,CPE1)-p(R2,CPE2)",
            "p0": [2.0, 30.0, 1e-4, 0.85, 50.0, 1e-5, 0.9],
            "names": ["Rs", "Rct1", "Q_CPE1", "n_CPE1", "Rct2", "Q_CPE2", "n_CPE2"],
        },
        "Custom": {"circuit": "", "p0": [], "names": []},
    }
    preset_choice = st.selectbox("Circuit preset", list(PRESETS.keys()))
    preset = PRESETS[preset_choice]

    if preset_choice == "Custom":
        circuit_str = st.text_input(
            "Circuit string",
            value="R0-p(R1,CPE1)",
            help="Elements: R#, C#, L#, CPE# (Q,n), W# (sigma), Wo# (R,tau). Series: A-B. Parallel: p(A,B).",
        )
        p0_str = st.text_input(
            "Initial parameters (comma-separated)",
            value="2.0, 30.0, 1e-4, 0.85",
        )
        try:
            p0_list = [float(x.strip()) for x in p0_str.split(",")]
        except Exception:
            p0_list = []
            st.error("Invalid initial parameters — check format.")
    else:
        circuit_str = preset["circuit"]
        p0_str = ", ".join(str(v) for v in preset["p0"])
        p0_list = preset["p0"]
        st.code(f"Circuit: {circuit_str}  |  p0 = [{p0_str}]", language=None)

with col_cfg2:
    warm_start = st.checkbox("Warm-start (recommended)", value=True,
                             help="Propagate best-fit params as p0 for next spectrum.")
    loss_fn = st.selectbox("Loss function", ["linear", "soft_l1"],
                           help="soft_l1 is more robust to outlier frequencies.")
    trend_param = st.selectbox(
        "Trend parameter to plot",
        ["Rct"] + (preset["names"] if preset_choice != "Custom" else []),
        index=0,
    )

st.divider()

# ── file upload ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload EIS Files")
st.caption("Order matters — sort by concentration (low → high) for best warm-start behaviour.")

uploaded = st.file_uploader(
    "Upload EIS files",
    type=["idf", "dta", "mpt", "mpr", "csv", "txt"],
    accept_multiple_files=True,
    key="batch_eis_upload",
)

if not uploaded:
    st.info("Upload at least 2 EIS files to start batch fitting.")
    st.stop()

st.markdown(f"**{len(uploaded)} files uploaded.**  Set labels and conditions below:")

# build per-file label + condition inputs
labels_in, conds_in = [], []
col_hdr1, col_hdr2, col_hdr3 = st.columns([2, 2, 1])
col_hdr1.markdown("**File**")
col_hdr2.markdown("**Label**")
col_hdr3.markdown("**Condition (e.g. M)**")

for i, f in enumerate(uploaded):
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.caption(f.name)
    lbl = c2.text_input("", value=f.name.split(".")[0], key=f"lbl_{i}", label_visibility="collapsed")
    cond = c3.number_input("", value=float(i + 1) * 0.25, step=0.05,
                            format="%.3f", key=f"cond_{i}", label_visibility="collapsed")
    labels_in.append(lbl)
    conds_in.append(cond)

st.divider()

# ── run fitting ───────────────────────────────────────────────────────────────
if st.button(f"▶ Run Batch Fit ({len(uploaded)} spectra)", type="primary"):
    if not circuit_str or not p0_list:
        st.error("Circuit string or initial parameters are empty.")
        st.stop()

    freq_list, zr_list, zi_list = [], [], []
    load_errors = []
    with st.spinner("Loading EIS files..."):
        for f in uploaded:
            try:
                fr, zr, zi = _load_eis_bytes(f.getvalue(), f.name)
                freq_list.append(fr)
                zr_list.append(zr)
                zi_list.append(zi)
                load_errors.append(None)
            except Exception as exc:
                freq_list.append(None)
                zr_list.append(None)
                zi_list.append(None)
                load_errors.append(str(exc))

    failed_loads = [i for i, e in enumerate(load_errors) if e]
    if failed_loads:
        for i in failed_loads:
            st.warning(f"⚠ Could not load {uploaded[i].name}: {load_errors[i]}")

    good_idx = [i for i, e in enumerate(load_errors) if e is None]
    if len(good_idx) < 2:
        st.error("Need at least 2 successfully loaded files.")
        st.stop()

    freq_g = [freq_list[i] for i in good_idx]
    zr_g   = [zr_list[i]   for i in good_idx]
    zi_g   = [zi_list[i]   for i in good_idx]
    lbl_g  = [labels_in[i] for i in good_idx]
    cond_g = [conds_in[i]  for i in good_idx]

    from eisforge.core.batch_fitter import BatchFitter
    fitter = BatchFitter(
        circuit=circuit_str,
        p0=np.array(p0_list, dtype=float),
        warm_start=warm_start,
        loss=loss_fn,
    )

    prog = st.progress(0, text="Fitting...")
    with st.spinner("Running warm-start CNLS batch fit..."):
        batch_result = fitter.fit_series(
            freq_g, zr_g, zi_g,
            labels=lbl_g,
            conditions=cond_g,
        )
    prog.progress(100, text="Done ✅")
    st.session_state["batch_eis_result"] = batch_result
    st.session_state["batch_eis_freq"]   = freq_g
    st.session_state["batch_eis_zr"]     = zr_g
    st.session_state["batch_eis_zi"]     = zi_g
    st.success(
        f"✅ {batch_result.n_success}/{len(batch_result.results)} spectra converged successfully."
    )

# ── results ───────────────────────────────────────────────────────────────────
if "batch_eis_result" not in st.session_state:
    st.stop()

br = st.session_state["batch_eis_result"]
freq_g = st.session_state["batch_eis_freq"]
zr_g   = st.session_state["batch_eis_zr"]
zi_g   = st.session_state["batch_eis_zi"]

# ── parameter table ───────────────────────────────────────────────────────────
st.markdown("### 📊 Fit Results")
df_res = br.to_dataframe()
st.dataframe(df_res.style.format(
    {c: "{:.4e}" for c in df_res.select_dtypes(include=float).columns}
), use_container_width=True)

# CSV download
csv_buf = io.StringIO()
df_res.to_csv(csv_buf, index=False)
st.download_button(
    "⬇ Download results CSV",
    data=csv_buf.getvalue(),
    file_name="batch_eis_results.csv",
    mime="text/csv",
)

st.divider()

# ── Nyquist overlay ───────────────────────────────────────────────────────────
st.markdown("### 🔵 Nyquist Overlay")
fig_ny = go.Figure()

for i, r in enumerate(br.results):
    # Colour pairs each spectrum's data with its own fit (semantic pairing);
    # data vs fit are additionally separated by open markers vs solid lines,
    # and marker symbols cycle, so >3 spectra stay distinguishable.
    _st = series_style(i, marker_size=5)
    fig_ny.add_trace(go.Scatter(
        x=zr_g[i], y=zi_g[i],
        mode="markers", name=f"{r.label} (data)",
        marker=_st["marker"],
    ))
    # fit
    if r.success:
        fig_ny.add_trace(go.Scatter(
            x=r.z_fit.real, y=r.z_fit.imag,
            mode="lines", name=f"{r.label} (fit)",
            line=dict(color=_st["line"]["color"], width=1.5),
        ))

fig_ny.update_layout(
    **PLOTLY_LAYOUT,
    title="Nyquist — all spectra",
    xaxis_title="Z' (Ω)",
    yaxis_title="Z'' (Ω)",
)
st.plotly_chart(fig_ny, use_container_width=True)

st.divider()

# ── parameter trend ───────────────────────────────────────────────────────────
st.markdown("### 📈 Parameter Trend vs Condition")

param_names_avail = br.results[0].param_names if br.results else []
param_plot = st.selectbox(
    "Parameter to plot", param_names_avail,
    index=0 if param_names_avail else 0,
    key="trend_param_select",
)

if param_plot:
    cond_arr = np.array(br.conditions, dtype=float) if br.conditions else np.arange(len(br.results), dtype=float)
    vals = br.get_param_series(param_plot)

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(
        x=cond_arr,
        y=vals,
        mode="lines+markers",
        marker=series_style(0, marker_size=8)["marker"],
        line=series_style(0)["line"],
        name=param_plot,
    ))
    fig_tr.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{param_plot} vs Condition",
        xaxis_title="Condition (M or other)",
        yaxis_title=param_plot,
    )
    st.plotly_chart(fig_tr, use_container_width=True)

st.caption(
    "Warm-start fitting by EISForge · patch19 · "
    "[GitHub](https://github.com/Hj1308/EISforge)"
)
