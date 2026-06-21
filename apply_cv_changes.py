"""
apply_cv_changes.py
===================
Applies 8 incremental changes to app.py, each as a separate git commit.

Usage (run from repo root):
    python apply_cv_changes.py

Requirements:
    pip install gitpython
    (or just use subprocess git — no external deps version below)

Each change patches only the relevant lines using str.replace().
If a patch fails (old text not found), it stops and reports which change failed.
"""

import re
import subprocess
from pathlib import Path

APP = Path("app.py")

# ─────────────────────────────────────────────────────────────────────────────
def read():
    return APP.read_text(encoding="utf-8")

def write(text):
    APP.write_text(text, encoding="utf-8")

def commit(msg):
    subprocess.run(["git", "add", "app.py"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    print(f"  ✅ Committed: {msg}")

def patch(src, old, new, label):
    if old not in src:
        raise ValueError(
            f"\n❌ PATCH FAILED [{label}]\n"
            f"Could not find the target text. Check app.py has not drifted.\n"
            f"--- expected ---\n{old[:200]}\n"
        )
    return src.replace(old, new, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 0 — top-level  import math
# ─────────────────────────────────────────────────────────────────────────────
def change_0(src):
    OLD = "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport re\nimport tempfile\nimport os\nfrom pathlib import Path"
    NEW = "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport re\nimport tempfile\nimport os\nimport math\nfrom pathlib import Path"
    return patch(src, OLD, NEW, "Change 0")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — replace load_cv_lsv with Ivium-aware 3-value loader
#            + fix ALL call sites to unpack 3 values
# ─────────────────────────────────────────────────────────────────────────────
def change_1(src):
    # 1a — replace the old function body
    OLD_FN = '''def load_cv_lsv(f, unit_factor=1.0):
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
        os.unlink(tmp)'''

    NEW_FN = '''def _parse_ivium_current_unit(text: str) -> float:
    """Parse \'Current Range=\' from Ivium metadata -> multiplier to mA."""
    m = re.search(r"Current Range\\s*=\\s*([\\d.]+)\\s*([A-Za-zµ]+)", text)
    if not m:
        return 1.0
    _, unit = m.groups()
    unit = unit.lower().strip()
    if unit == "a":             return 1000.0
    elif unit == "ma":          return 1.0
    elif unit in ("ua", "µa"): return 0.001
    return 1.0


def _load_ivium_cv(path: str, cycle_idx: int = -1):
    """Load Ivium .idf CV. Returns (E, I_mA, meta).
    Auto-detects current unit; supports cycle selection (-1 = last complete)."""
    text = open(path, "rb").read().decode("latin-1")
    meta = {}
    for key in ["Scanrate", "N scans", "E start", "Vertex 1", "Vertex 2"]:
        mm = re.search(re.escape(key) + r"=([^\\r\\n]+)", text)
        if mm:
            try:    meta[key] = float(mm.group(1).strip())
            except: meta[key] = mm.group(1).strip()

    unit_mult = _parse_ivium_current_unit(text)
    meta["_unit_mult"]  = unit_mult
    meta["_unit_label"] = ("A" if unit_mult == 1000.0 else
                           "µA" if unit_mult == 0.001 else "mA")

    rows = re.findall(
        r"(-?\\d\\.\\d+E[+-]\\d+)\\s+(-?\\d\\.\\d+E[+-]\\d+)\\s+(-?\\d\\.\\d+E[+-]\\d+)", text)
    if not rows:
        raise ValueError("No numeric data found in .idf file")
    arr   = np.array(rows, dtype=float)
    E_all = arr[:, 0]
    I_all = arr[:, 1] * unit_mult   # -> mA

    sign_ch  = np.diff(np.sign(np.diff(E_all)))
    vertices = np.where(sign_ch != 0)[0] + 1
    if len(vertices) < 2:
        meta["_n_cycles"] = 1; meta["_cycle_used"] = 1
        return E_all, I_all, meta

    cycle_starts = [0] + list(vertices[::2] + 1)
    cycle_ends   = list(vertices[::2] + 1) + [len(E_all)]
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
    vertex = int(np.argmax(E))
    Q_f = float(np.trapezoid(I_mA[:vertex + 1], E[:vertex + 1]) / nu)
    Q_b = float(np.trapezoid(I_mA[vertex:],     E[vertex:])     / nu)
    return Q_f + Q_b, Q_f, Q_b


def load_cv_lsv(f, unit_factor=1.0, cycle_idx=-1):
    """Unified CV/LSV loader: Ivium .idf (unit+cycle aware) or CSV/TXT.
    Always returns (E, I_mA, meta) — 3 values."""
    suffix = Path(f.name).suffix.lower()
    tmp = save_upload(f)
    try:
        if suffix == ".idf":
            return _load_ivium_cv(tmp, cycle_idx=cycle_idx)
        else:
            df = read_csv_safe(tmp); c = df.columns.tolist()
            E  = df[c[0]].to_numpy(float)
            I  = df[c[1]].to_numpy(float) * unit_factor
            return E, I, {}
    finally:
        os.unlink(tmp)'''

    src = patch(src, OLD_FN, NEW_FN, "Change 1a — new load_cv_lsv")

    # 1b — fix call sites in LSV tab (single file)
    src = src.replace(
        'if Path(lsv_file.name).suffix.lower()==".idf":\n                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=1.0)\n                else:\n                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=unit_factor)',
        'if Path(lsv_file.name).suffix.lower()==".idf":\n                    pot_lsv,cur_lsv,_ = load_cv_lsv(lsv_file, unit_factor=1.0)\n                else:\n                    pot_lsv,cur_lsv,_ = load_cv_lsv(lsv_file, unit_factor=unit_factor)',
    )

    # 1c — fix call sites in batch CV
    src = src.replace(
        'p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_b.append(p); curs_b.append(c)',
        'p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_b.append(p); curs_b.append(c)',
    )

    # 1d — fix call sites in batch LSV
    src = src.replace(
        'p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_b.append(p); curs_b.append(c)',
        'p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_b.append(p); curs_b.append(c)',
    )

    # 1e — fix call sites in K-L tab
    src = src.replace(
        'p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_kl.append(p)\n                        curs_kl.append(c)',
        'p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n                        pots_kl.append(p)\n                        curs_kl.append(c)',
    )

    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — sidebar: diameter -> auto area
# ─────────────────────────────────────────────────────────────────────────────
def change_2(src):
    OLD = '''    # ── CHANGE 3: Geometric area — 4 decimal places ────────────────────────
    area = st.number_input(
        "Geometric area (cm²)",
        value=1.0,
        step=0.0001,
        min_value=0.0001,
        format="%.4f",
    )'''
    NEW = '''    diameter_mm = st.number_input(
        "Disk diameter (mm)", value=0.0, min_value=0.0, step=0.5,
        help="If > 0, overrides area below. Standard GCE: 3 mm or 5 mm")
    if diameter_mm > 0:
        area = math.pi * (diameter_mm / 20.0) ** 2
        st.caption(f"→ Area = {area:.4f} cm²")
    else:
        area = st.number_input("Geometric area (cm²)", value=1.0,
            min_value=0.0001, step=0.0001, format="%.4f")'''
    return patch(src, OLD, NEW, "Change 2")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3 — sidebar: deposited mass (µg) -> auto loading
# ─────────────────────────────────────────────────────────────────────────────
def change_3(src):
    OLD = '    loading = st.number_input("Loading (mg/cm²)",      value=0.0, step=0.01, min_value=0.0)'
    NEW = '''    mass_ug = st.number_input(
        "Deposited mass (µg)", value=0.0, min_value=0.0, step=1.0,
        help="If > 0, overrides loading below. e.g. 5 µg on 3 mm GCE")
    if mass_ug > 0 and area > 0:
        loading = (mass_ug / 1000.0) / area
        st.caption(f"→ Loading = {loading:.4f} mg/cm²  |  mass = {mass_ug:.1f} µg")
    else:
        loading = st.number_input("Loading (mg/cm²)", value=0.0, step=0.01, min_value=0.0)
    _mass_mg = mass_ug / 1000.0 if mass_ug > 0 else loading * area'''
    return patch(src, OLD, NEW, "Change 3")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 4 — sidebar: alcohol concentration field
# ─────────────────────────────────────────────────────────────────────────────
def change_4(src):
    OLD = '''    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)

    eis_pot'''
    NEW = '''    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)

    alcohol_conc = st.number_input(
        "Alcohol conc. (M)", value=0.25, step=0.05, min_value=0.0,
        help="Concentration of alcohol in solution",
        disabled=(system_type != "AOR"))

    eis_pot'''
    return patch(src, OLD, NEW, "Change 4")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 5 — sidebar: elec_conc default + auto pH
# ─────────────────────────────────────────────────────────────────────────────
def change_5(src):
    OLD_CONC = '    elec_conc    = st.number_input("Electrolyte conc. (M)", value=0.5, step=0.1)'
    NEW_CONC = '''    elec_conc = st.number_input("Electrolyte conc. (M)", value=1.0, step=0.1,
        min_value=0.01, help="Used for pH and RHE conversion only")'''
    src = patch(src, OLD_CONC, NEW_CONC, "Change 5a — elec_conc default")

    OLD_PH = '''    _PH_MAP = {
        "H2SO4": 0.3, "HClO4": 0.3, "HCl": 0.0, "HNO3": 0.0,
        "KOH": 14.0,  "NaOH": 14.0, "Na2CO3": 11.6, "NH3": 11.6,
    }
    _ph_default = float(_PH_MAP.get(elec_compound_key, 14.0 if ekey == "alkaline" else 0.0))
    ph_value = st.number_input(
        "Solution pH", min_value=0.0, max_value=14.0,
        value=_ph_default, step=0.1,
    )'''
    NEW_PH = '''    def _auto_ph(compound: str, conc: float) -> float:
        if compound == "H2SO4":
            return max(-math.log10(2 * conc), -1.0)
        elif compound in ("HCl", "HClO4", "HNO3"):
            return max(-math.log10(conc), -1.0)
        elif compound in ("KOH", "NaOH"):
            return min(14.0 + math.log10(conc), 15.0)
        elif compound in ("Na2CO3", "NH3"):
            return 11.6
        return 14.0 if ekey == "alkaline" else 7.0

    _ph_auto = _auto_ph(elec_compound_key, elec_conc)
    _ph_override = st.checkbox("Override pH manually", value=False)
    if _ph_override:
        ph_value = st.number_input("pH (manual)",
            value=float(round(_ph_auto, 2)), min_value=0.0, max_value=14.0, step=0.1)
    else:
        ph_value = _ph_auto
        st.caption(f"pH = **{ph_value:.2f}** (auto — {elec_compound_key} {elec_conc} M)")'''
    src = patch(src, OLD_PH, NEW_PH, "Change 5b — auto pH")
    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 6 — CV tab: cycle selector + smoothing + updated col2 block
# ─────────────────────────────────────────────────────────────────────────────
def change_6(src):
    # 6a — add cycle_idx + use_smooth after sr_cv
    OLD_SR = '''        sr_cv   = st.number_input("Scan rate (mV/s)", value=50, min_value=1)

        # ── CHANGE 4: E_onset method auto + override ───────────────────────'''
    NEW_SR = '''        sr_cv   = st.number_input("Scan rate (mV/s)", value=50, min_value=1)

        cycle_idx = st.number_input(
            "Cycle to analyse (0=first, -1=last complete)",
            value=-1, min_value=-1, step=1,
            help="Ivium often saves the last cycle incomplete; -1 picks the last closed one")
        use_smooth = st.checkbox("Smooth noisy curve (Savitzky-Golay)", value=False)
        sg_window  = st.slider("SG window (odd)", 5, 31, 11, 2) if use_smooth else 11

        # ── CHANGE 4: E_onset method auto + override ───────────────────────'''
    src = patch(src, OLD_SR, NEW_SR, "Change 6a — cycle selector + smoothing")

    # 6b — replace col2 upload block
    OLD_COL2 = '''    with col2:
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
                st.error(f"Error: {e}")'''
    NEW_COL2 = '''    with col2:
        if cv_file:
            try:
                pot, cur, _meta = load_cv_lsv(cv_file, unit_factor=unit_factor,
                                               cycle_idx=int(cycle_idx))
                if "Scanrate" in _meta:
                    sr_cv = int(_meta["Scanrate"] * 1000)
                if use_smooth:
                    from scipy.signal import savgol_filter
                    w = sg_window if sg_window % 2 == 1 else sg_window + 1
                    cur = savgol_filter(cur, window_length=min(w, len(cur)//2*2-1), polyorder=3)
                _nc = _meta.get("_n_cycles", "?"); _cu = _meta.get("_cycle_used", "?")
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
                st.session_state.update({"cv_r": r, "cv_pot": pot, "cv_cur": cur,
                    "cv_pot_corr": CVAnalyzer.apply_ir_compensation(pot, cur, actual_rs)
                                   if actual_rs > 0 else pot})
            except Exception as e:
                st.error(f"Error: {e}")'''
    src = patch(src, OLD_COL2, NEW_COL2, "Change 6b — col2 upload block")
    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 7 — CV tab: metrics (RHE onset, mass activity, Q, hide I_f/I_b)
# ─────────────────────────────────────────────────────────────────────────────
def change_7(src):
    OLD = '''        c1,c2,c3,c4 = st.columns(4)
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

        st.info(f"**Interpretation:** {r.interpretation}")'''
    NEW = '''        _e_onset_rhe = r.e_onset + e_ref_val + 0.059 * ph_value
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("E_onset (vs ref)", f"{r.e_onset:.4f} V")
        c2.metric("E_onset (vs RHE)", f"{_e_onset_rhe:.4f} V",
                  help=f"= {r.e_onset:.4f} + {e_ref_val:.3f}(ref) + 0.059×{ph_value:.2f}(pH)")
        if _is_mf:
            c3.metric("Net faradaic I", f"{r.net_faradaic_current_mA:.4f} mA")
            c4.metric("C_dl",           f"{r.cdl_mF_cm2:.4f} mF/cm²")
        else:
            c3.metric("I_b",     f"{r.i_backward_peak:.4f} mA")
            c4.metric("I_f/I_b", f"{r.if_ib_ratio:.3f}" if not np.isnan(r.if_ib_ratio) else "N/A")
        c5,c6,c7,c8 = st.columns(4)
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
            _Qf = st.session_state["cv_Q_f"]; _Qb = st.session_state["cv_Q_b"]
            cq1,cq2,cq3 = st.columns(3)
            cq1.metric("Q_forward (mC)",  f"{_Qf:.3f}")
            cq2.metric("Q_backward (mC)", f"{abs(_Qb):.3f}")
            cq3.metric("Q_f / |Q_b|", f"{abs(_Qf/_Qb):.3f}" if _Qb != 0 else "N/A",
                       help="≈1 = reversible")
        st.caption(f"Alcohol: **{alcohol}** {alcohol_conc} M  |  "
                   f"Electrolyte: **{elec_compound}** {elec_conc} M  |  pH = {ph_value:.2f}")
        st.info(f"**Interpretation:** {r.interpretation}")'''
    return patch(src, OLD, NEW, "Change 7")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 8 — CV tab: plot j (mA cm⁻²) instead of I (mA)
# ─────────────────────────────────────────────────────────────────────────────
def change_8(src):
    OLD = '''        import plotly.graph_objects as go
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
        title = f"CV — {sr_cv} mV/s | {temperature}°C | {catalyst or \'Catalyst\'}"
        if actual_rs>0: title += f" | iR-corrected (R_s={actual_rs:.1f}Ω)"
        fig.update_layout(**PLOTLY_LAYOUT, title=title,
                          xaxis_title=f"Potential (V vs {e_ref_type})",
                          yaxis_title="Current (mA)")
        st.plotly_chart(fig, use_container_width=True)'''
    NEW = '''        import plotly.graph_objects as go
        fig = go.Figure()
        x_plot = st.session_state.get("cv_pot_corr", st.session_state["cv_pot"])
        _cur_plot = st.session_state["cv_cur"]
        j_arr = _cur_plot / area if area > 0 else _cur_plot
        fig.add_trace(go.Scatter(x=x_plot, y=j_arr, mode="lines",
            name="CV" + (" (iR-corrected)" if actual_rs>0 else ""),
            line=dict(color="#2563eb", width=2)))
        fig.add_vline(x=r.e_onset, line_dash="dash", line_color="#d97706",
            annotation_text=f"E_onset = {r.e_onset:.3f} V",
            annotation_font=dict(color="#d97706"))
        if r.e_forward_peak is not None:
            fig.add_trace(go.Scatter(x=[r.e_forward_peak], y=[r.i_forward_peak/area],
                mode="markers", name=f"j_f = {r.i_forward_peak/area:.3f} mA cm⁻²",
                marker=dict(color="#16a34a", size=12, symbol="star")))
        if r.e_backward_peak is not None and not _is_mf:
            fig.add_trace(go.Scatter(x=[r.e_backward_peak], y=[r.i_backward_peak/area],
                mode="markers", name=f"j_b = {r.i_backward_peak/area:.3f} mA cm⁻²",
                marker=dict(color="#dc2626", size=12, symbol="star")))
        title = f"j vs E — {sr_cv} mV/s | {temperature}°C | {catalyst or \'Catalyst\'}"
        if actual_rs>0: title += f" | iR-corrected (R_s={actual_rs:.1f}Ω)"
        fig.update_layout(**PLOTLY_LAYOUT, title=title,
            xaxis_title=f"E (V vs {e_ref_type})", yaxis_title="j (mA cm⁻²)")
        st.plotly_chart(fig, use_container_width=True)'''
    return patch(src, OLD, NEW, "Change 8")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
CHANGES = [
    (change_0, "Change 0: add top-level `import math`"),
    (change_1, "Change 1: Ivium-aware load_cv_lsv + fix all call sites (3-value unpack)"),
    (change_2, "Change 2: sidebar — disk diameter → auto geometric area"),
    (change_3, "Change 3: sidebar — deposited mass (µg) → auto loading"),
    (change_4, "Change 4: sidebar — alcohol concentration field"),
    (change_5, "Change 5: sidebar — electrolyte conc default + auto pH calculator"),
    (change_6, "Change 6: CV tab — cycle selector + Savitzky-Golay smoothing"),
    (change_7, "Change 7: CV tab — RHE onset, mass activity, charge metrics"),
    (change_8, "Change 8: CV tab — plot j (mA cm⁻²) instead of I (mA)"),
]

if __name__ == "__main__":
    print(f"📄 Reading {APP} ...")
    src = read()

    for fn, msg in CHANGES:
        print(f"\n🔧 Applying: {msg}")
        src = fn(src)
        write(src)
        commit(msg)
        print(f"   Lines now: {src.count(chr(10))}")

    print("\n✅ All 9 changes applied and committed.")
    print("👉 Run:  git push origin main")
