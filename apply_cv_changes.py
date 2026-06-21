"""
apply_cv_changes.py
===================
Applies 8 incremental changes to app.py, each as a separate git commit.

Usage (run from repo root):
    python apply_cv_changes.py

Each change patches only the relevant lines using str.replace().
CRLF/LF differences are handled automatically.
"""

import re
import subprocess
from pathlib import Path

APP = Path("app.py")

# ─────────────────────────────────────────────────────────────────────────────
def read():
    raw = APP.read_bytes()
    # normalize CRLF → LF so patches always work
    return raw.replace(b"\r\n", b"\n").decode("utf-8")

def write(text):
    APP.write_bytes(text.encode("utf-8"))   # write LF only

def commit(msg):
    subprocess.run(["git", "add", "app.py"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    print(f"  ✅ Committed: {msg}")

def patch(src, old, new, label):
    # also normalize the search string
    old_n = old.replace("\r\n", "\n")
    if old_n not in src:
        # try a whitespace-flexible search to give a better error
        hint = src[max(0, src.find(old_n[:40])-100) : src.find(old_n[:40])+200]
        raise ValueError(
            f"\n❌ PATCH FAILED [{label}]\n"
            f"Could not find the target text.\n"
            f"--- first 40 chars of expected ---\n{old_n[:80]!r}\n"
            f"--- surrounding context in file ---\n{hint!r}\n"
        )
    return src.replace(old_n, new, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 0 — top-level import math
# ─────────────────────────────────────────────────────────────────────────────
def change_0(src):
    OLD = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import re\n"
        "import tempfile\n"
        "import os\n"
        "from pathlib import Path"
    )
    NEW = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import re\n"
        "import tempfile\n"
        "import os\n"
        "import math\n"
        "from pathlib import Path"
    )
    return patch(src, OLD, NEW, "Change 0")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — replace load_cv_lsv + fix ALL call sites
# ─────────────────────────────────────────────────────────────────────────────
def change_1(src):
    # 1a — replace old function
    OLD_FN = (
        "def load_cv_lsv(f, unit_factor=1.0):\n"
        "    suffix = Path(f.name).suffix.lower()\n"
        "    tmp = save_upload(f)\n"
        "    try:\n"
        "        if suffix==\".idf\":\n"
        "            from eisforge.parsers.autolab_parser import AutolabIDFParser\n"
        "            ds=AutolabIDFParser().parse(tmp)\n"
        "            return ds.z_real, ds.z_imag*unit_factor\n"
        "        else:\n"
        "            df=read_csv_safe(tmp); c=df.columns.tolist()\n"
        "            return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)*unit_factor\n"
        "    finally:\n"
        "        os.unlink(tmp)"
    )
    NEW_FN = (
        "def _parse_ivium_current_unit(text: str) -> float:\n"
        "    \"\"\"Parse 'Current Range=' from Ivium metadata -> multiplier to mA.\"\"\"\n"
        "    m = re.search(r\"Current Range\\s*=\\s*([\\d.]+)\\s*([A-Za-zµ]+)\", text)\n"
        "    if not m:\n"
        "        return 1.0\n"
        "    _, unit = m.groups()\n"
        "    unit = unit.lower().strip()\n"
        "    if unit == \"a\":             return 1000.0\n"
        "    elif unit == \"ma\":          return 1.0\n"
        "    elif unit in (\"ua\", \"µa\"): return 0.001\n"
        "    return 1.0\n"
        "\n"
        "\n"
        "def _load_ivium_cv(path: str, cycle_idx: int = -1):\n"
        "    \"\"\"Load Ivium .idf CV. Returns (E, I_mA, meta).\n"
        "    Auto-detects current unit; supports cycle selection (-1 = last complete).\"\"\"\n"
        "    text = open(path, \"rb\").read().decode(\"latin-1\")\n"
        "    meta = {}\n"
        "    for key in [\"Scanrate\", \"N scans\", \"E start\", \"Vertex 1\", \"Vertex 2\"]:\n"
        "        mm = re.search(re.escape(key) + r\"=([^\\r\\n]+)\", text)\n"
        "        if mm:\n"
        "            try:    meta[key] = float(mm.group(1).strip())\n"
        "            except: meta[key] = mm.group(1).strip()\n"
        "\n"
        "    unit_mult = _parse_ivium_current_unit(text)\n"
        "    meta[\"_unit_mult\"]  = unit_mult\n"
        "    meta[\"_unit_label\"] = (\"A\" if unit_mult == 1000.0 else\n"
        "                           \"µA\" if unit_mult == 0.001 else \"mA\")\n"
        "\n"
        "    rows = re.findall(\n"
        "        r\"(-?\\d\\.\\d+E[+-]\\d+)\\s+(-?\\d\\.\\d+E[+-]\\d+)\\s+(-?\\d\\.\\d+E[+-]\\d+)\", text)\n"
        "    if not rows:\n"
        "        raise ValueError(\"No numeric data found in .idf file\")\n"
        "    arr   = np.array(rows, dtype=float)\n"
        "    E_all = arr[:, 0]\n"
        "    I_all = arr[:, 1] * unit_mult   # -> mA\n"
        "\n"
        "    sign_ch  = np.diff(np.sign(np.diff(E_all)))\n"
        "    vertices = np.where(sign_ch != 0)[0] + 1\n"
        "    if len(vertices) < 2:\n"
        "        meta[\"_n_cycles\"] = 1; meta[\"_cycle_used\"] = 1\n"
        "        return E_all, I_all, meta\n"
        "\n"
        "    cycle_starts = [0] + list(vertices[::2] + 1)\n"
        "    cycle_ends   = list(vertices[::2] + 1) + [len(E_all)]\n"
        "    n_cycles = len(cycle_starts) - 1\n"
        "    meta[\"_n_cycles\"] = n_cycles\n"
        "    if cycle_idx == -1:\n"
        "        chosen = max(0, n_cycles - 2) if n_cycles >= 2 else 0\n"
        "    else:\n"
        "        chosen = max(0, min(cycle_idx, n_cycles - 1))\n"
        "    meta[\"_cycle_used\"] = chosen + 1\n"
        "    s, e_ = cycle_starts[chosen], cycle_ends[chosen]\n"
        "    return E_all[s:e_], I_all[s:e_], meta\n"
        "\n"
        "\n"
        "def _compute_charge(E, I_mA, scan_rate_mV_s):\n"
        "    \"\"\"Q (mC) = integral I dt = integral I dE / nu.\n"
        "    Uses np.trapezoid (np.trapz removed in numpy 2.0).\"\"\"\n"
        "    nu = max(scan_rate_mV_s / 1000.0, 1e-9)\n"
        "    vertex = int(np.argmax(E))\n"
        "    Q_f = float(np.trapezoid(I_mA[:vertex + 1], E[:vertex + 1]) / nu)\n"
        "    Q_b = float(np.trapezoid(I_mA[vertex:],     E[vertex:])     / nu)\n"
        "    return Q_f + Q_b, Q_f, Q_b\n"
        "\n"
        "\n"
        "def load_cv_lsv(f, unit_factor=1.0, cycle_idx=-1):\n"
        "    \"\"\"Unified CV/LSV loader: Ivium .idf (unit+cycle aware) or CSV/TXT.\n"
        "    Always returns (E, I_mA, meta) — 3 values.\"\"\"\n"
        "    suffix = Path(f.name).suffix.lower()\n"
        "    tmp = save_upload(f)\n"
        "    try:\n"
        "        if suffix == \".idf\":\n"
        "            return _load_ivium_cv(tmp, cycle_idx=cycle_idx)\n"
        "        else:\n"
        "            df = read_csv_safe(tmp); c = df.columns.tolist()\n"
        "            E  = df[c[0]].to_numpy(float)\n"
        "            I  = df[c[1]].to_numpy(float) * unit_factor\n"
        "            return E, I, {}\n"
        "    finally:\n"
        "        os.unlink(tmp)"
    )
    src = patch(src, OLD_FN, NEW_FN, "Change 1a — new load_cv_lsv")

    # 1b — LSV single-file call site
    src = src.replace(
        'if Path(lsv_file.name).suffix.lower()==".idf":\n'
        '                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=1.0)\n'
        '                else:\n'
        '                    pot_lsv,cur_lsv = load_cv_lsv(lsv_file, unit_factor=unit_factor)',
        'if Path(lsv_file.name).suffix.lower()==".idf":\n'
        '                    pot_lsv,cur_lsv,_ = load_cv_lsv(lsv_file, unit_factor=1.0)\n'
        '                else:\n'
        '                    pot_lsv,cur_lsv,_ = load_cv_lsv(lsv_file, unit_factor=unit_factor)',
    )

    # 1c & 1d — batch CV + batch LSV (same pattern, replace both occurrences)
    src = src.replace(
        'p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n'
        '                        pots_b.append(p); curs_b.append(c)',
        'p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n'
        '                        pots_b.append(p); curs_b.append(c)',
    )   # replaces ALL occurrences (batch CV + batch LSV both match)

    # 1e — K-L tab
    src = src.replace(
        '                        p, c = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n'
        '                        pots_kl.append(p)\n'
        '                        curs_kl.append(c)',
        '                        p, c, _ = load_cv_lsv(f, unit_factor=1.0 if Path(f.name).suffix.lower()==".idf" else unit_factor)\n'
        '                        pots_kl.append(p)\n'
        '                        curs_kl.append(c)',
    )
    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — sidebar: diameter -> auto area
# ─────────────────────────────────────────────────────────────────────────────
def change_2(src):
    OLD = (
        "    # ── CHANGE 3: Geometric area — 4 decimal places ────────────────────────\n"
        "    area = st.number_input(\n"
        "        \"Geometric area (cm²)\",\n"
        "        value=1.0,\n"
        "        step=0.0001,\n"
        "        min_value=0.0001,\n"
        "        format=\"%.4f\",\n"
        "    )"
    )
    NEW = (
        "    diameter_mm = st.number_input(\n"
        "        \"Disk diameter (mm)\", value=0.0, min_value=0.0, step=0.5,\n"
        "        help=\"If > 0, overrides area below. Standard GCE: 3 mm or 5 mm\")\n"
        "    if diameter_mm > 0:\n"
        "        area = math.pi * (diameter_mm / 20.0) ** 2\n"
        "        st.caption(f\"→ Area = {area:.4f} cm²\")\n"
        "    else:\n"
        "        area = st.number_input(\"Geometric area (cm²)\", value=1.0,\n"
        "            min_value=0.0001, step=0.0001, format=\"%.4f\")"
    )
    return patch(src, OLD, NEW, "Change 2")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3 — sidebar: deposited mass (µg) -> auto loading
# ─────────────────────────────────────────────────────────────────────────────
def change_3(src):
    OLD = '    loading = st.number_input(\"Loading (mg/cm²)\",      value=0.0, step=0.01, min_value=0.0)'
    NEW = (
        "    mass_ug = st.number_input(\n"
        "        \"Deposited mass (µg)\", value=0.0, min_value=0.0, step=1.0,\n"
        "        help=\"If > 0, overrides loading below. e.g. 5 µg on 3 mm GCE\")\n"
        "    if mass_ug > 0 and area > 0:\n"
        "        loading = (mass_ug / 1000.0) / area\n"
        "        st.caption(f\"→ Loading = {loading:.4f} mg/cm²  |  mass = {mass_ug:.1f} µg\")\n"
        "    else:\n"
        "        loading = st.number_input(\"Loading (mg/cm²)\", value=0.0, step=0.01, min_value=0.0)\n"
        "    _mass_mg = mass_ug / 1000.0 if mass_ug > 0 else loading * area"
    )
    return patch(src, OLD, NEW, "Change 3")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 4 — sidebar: alcohol concentration field
# ─────────────────────────────────────────────────────────────────────────────
def change_4(src):
    OLD = (
        "    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)\n"
        "\n"
        "    eis_pot"
    )
    NEW = (
        "    alcohol_key = _ALCOHOL_KEY_MAP.get(alcohol, alcohol)\n"
        "\n"
        "    alcohol_conc = st.number_input(\n"
        "        \"Alcohol conc. (M)\", value=0.25, step=0.05, min_value=0.0,\n"
        "        help=\"Concentration of alcohol in solution\",\n"
        "        disabled=(system_type != \"AOR\"))\n"
        "\n"
        "    eis_pot"
    )
    return patch(src, OLD, NEW, "Change 4")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 5 — sidebar: elec_conc default + auto pH
# ─────────────────────────────────────────────────────────────────────────────
def change_5(src):
    OLD_CONC = '    elec_conc    = st.number_input(\"Electrolyte conc. (M)\", value=0.5, step=0.1)'
    NEW_CONC = (
        "    elec_conc = st.number_input(\"Electrolyte conc. (M)\", value=1.0, step=0.1,\n"
        "        min_value=0.01, help=\"Used for pH and RHE conversion only\")"
    )
    src = patch(src, OLD_CONC, NEW_CONC, "Change 5a — elec_conc default")

    OLD_PH = (
        "    _PH_MAP = {\n"
        "        \"H2SO4\": 0.3, \"HClO4\": 0.3, \"HCl\": 0.0, \"HNO3\": 0.0,\n"
        "        \"KOH\": 14.0,  \"NaOH\": 14.0, \"Na2CO3\": 11.6, \"NH3\": 11.6,\n"
        "    }\n"
        "    _ph_default = float(_PH_MAP.get(elec_compound_key, 14.0 if ekey == \"alkaline\" else 0.0))\n"
        "    ph_value = st.number_input(\n"
        "        \"Solution pH\", min_value=0.0, max_value=14.0,\n"
        "        value=_ph_default, step=0.1,\n"
        "    )"
    )
    NEW_PH = (
        "    def _auto_ph(compound: str, conc: float) -> float:\n"
        "        if compound == \"H2SO4\":\n"
        "            return max(-math.log10(2 * conc), -1.0)\n"
        "        elif compound in (\"HCl\", \"HClO4\", \"HNO3\"):\n"
        "            return max(-math.log10(conc), -1.0)\n"
        "        elif compound in (\"KOH\", \"NaOH\"):\n"
        "            return min(14.0 + math.log10(conc), 15.0)\n"
        "        elif compound in (\"Na2CO3\", \"NH3\"):\n"
        "            return 11.6\n"
        "        return 14.0 if ekey == \"alkaline\" else 7.0\n"
        "\n"
        "    _ph_auto = _auto_ph(elec_compound_key, elec_conc)\n"
        "    _ph_override = st.checkbox(\"Override pH manually\", value=False)\n"
        "    if _ph_override:\n"
        "        ph_value = st.number_input(\"pH (manual)\",\n"
        "            value=float(round(_ph_auto, 2)), min_value=0.0, max_value=14.0, step=0.1)\n"
        "    else:\n"
        "        ph_value = _ph_auto\n"
        "        st.caption(f\"pH = **{ph_value:.2f}** (auto — {elec_compound_key} {elec_conc} M)\")"
    )
    src = patch(src, OLD_PH, NEW_PH, "Change 5b — auto pH")
    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 6 — CV tab: cycle selector + smoothing + updated col2
# ─────────────────────────────────────────────────────────────────────────────
def change_6(src):
    # 6a — add cycle_idx + use_smooth after sr_cv
    OLD_SR = (
        "        sr_cv   = st.number_input(\"Scan rate (mV/s)\", value=50, min_value=1)\n"
        "\n"
        "        # ── CHANGE 4: E_onset method auto + override ───────────────────────"
    )
    NEW_SR = (
        "        sr_cv   = st.number_input(\"Scan rate (mV/s)\", value=50, min_value=1)\n"
        "\n"
        "        cycle_idx = st.number_input(\n"
        "            \"Cycle to analyse (0=first, -1=last complete)\",\n"
        "            value=-1, min_value=-1, step=1,\n"
        "            help=\"Ivium often saves the last cycle incomplete; -1 picks the last closed one\")\n"
        "        use_smooth = st.checkbox(\"Smooth noisy curve (Savitzky-Golay)\", value=False)\n"
        "        sg_window  = st.slider(\"SG window (odd)\", 5, 31, 11, 2) if use_smooth else 11\n"
        "\n"
        "        # ── CHANGE 4: E_onset method auto + override ───────────────────────"
    )
    src = patch(src, OLD_SR, NEW_SR, "Change 6a — cycle selector + smoothing")

    # 6b — replace col2 upload block
    OLD_COL2 = (
        "    with col2:\n"
        "        if cv_file:\n"
        "            try:\n"
        "                if Path(cv_file.name).suffix.lower()==\".idf\":\n"
        "                    pot,cur = load_cv_lsv(cv_file, unit_factor=1.0)\n"
        "                    st.info(\"Current auto-converted from A → mA (Autolab)\")\n"
        "                else:\n"
        "                    pot,cur = load_cv_lsv(cv_file, unit_factor=unit_factor)\n"
        "\n"
        "                st.success(f\"✅ {len(pot)} points | {cv_file.name}\")\n"
        "\n"
        "                from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo\n"
        "                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)\n"
        "                ana = CVAnalyzer(scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,\n"
        "                                 catalyst_loading=loading, onset_method=om,\n"
        "                                 electrolyte=_el, catalyst_type=catalyst_type)\n"
        "                r = ana.analyze(pot, cur, r_s_ohms=actual_rs)\n"
        "                st.session_state.update({\"cv_r\":r,\"cv_pot\":pot,\"cv_cur\":cur,\"cv_pot_corr\":\n"
        "                    CVAnalyzer.apply_ir_compensation(pot, cur, actual_rs) if actual_rs>0 else pot})\n"
        "            except Exception as e:\n"
        "                st.error(f\"Error: {e}\")"
    )
    NEW_COL2 = (
        "    with col2:\n"
        "        if cv_file:\n"
        "            try:\n"
        "                pot, cur, _meta = load_cv_lsv(cv_file, unit_factor=unit_factor,\n"
        "                                               cycle_idx=int(cycle_idx))\n"
        "                if \"Scanrate\" in _meta:\n"
        "                    sr_cv = int(_meta[\"Scanrate\"] * 1000)\n"
        "                if use_smooth:\n"
        "                    from scipy.signal import savgol_filter\n"
        "                    w = sg_window if sg_window % 2 == 1 else sg_window + 1\n"
        "                    cur = savgol_filter(cur, window_length=min(w, len(cur)//2*2-1), polyorder=3)\n"
        "                _nc = _meta.get(\"_n_cycles\", \"?\"); _cu = _meta.get(\"_cycle_used\", \"?\")\n"
        "                _ul = _meta.get(\"_unit_label\", \"mA\")\n"
        "                st.success(f\"✅ {len(pot)} points | {cv_file.name} | \"\n"
        "                           f\"cycle {_cu}/{_nc} | unit: {_ul} | sr: {sr_cv} mV/s\"\n"
        "                           + (\" | smoothed\" if use_smooth else \"\"))\n"
        "                Q_total, Q_f, Q_b = _compute_charge(pot, cur, sr_cv)\n"
        "                st.session_state.update({\"cv_Q_total\": Q_total, \"cv_Q_f\": Q_f, \"cv_Q_b\": Q_b})\n"
        "                from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo\n"
        "                _el = ElectrolyteInfo(media=ekey, compound=elec_compound_key, concentration=elec_conc)\n"
        "                ana = CVAnalyzer(scan_rate=sr_cv, electrode_area=area, ecsa=ecsa,\n"
        "                                 catalyst_loading=loading, onset_method=om,\n"
        "                                 electrolyte=_el, catalyst_type=catalyst_type)\n"
        "                r = ana.analyze(pot, cur, r_s_ohms=actual_rs)\n"
        "                st.session_state.update({\"cv_r\": r, \"cv_pot\": pot, \"cv_cur\": cur,\n"
        "                    \"cv_pot_corr\": CVAnalyzer.apply_ir_compensation(pot, cur, actual_rs)\n"
        "                                   if actual_rs > 0 else pot})\n"
        "            except Exception as e:\n"
        "                st.error(f\"Error: {e}\")"
    )
    src = patch(src, OLD_COL2, NEW_COL2, "Change 6b — col2 upload block")
    return src

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 7 — CV tab: metrics (RHE onset, mass activity, Q)
# ─────────────────────────────────────────────────────────────────────────────
def change_7(src):
    OLD = (
        "        c1,c2,c3,c4 = st.columns(4)\n"
        "        c1.metric(\"E_onset\",  f\"{r.e_onset:.4f} V\")\n"
        "        c2.metric(\"I_forward peak\", f\"{r.i_forward_peak:.4f} mA\")\n"
        "        if _is_mf:\n"
        "            c3.metric(\"Net faradaic I\", f\"{r.net_faradaic_current_mA:.4f} mA\")\n"
        "            c4.metric(\"C_dl\",           f\"{r.cdl_mF_cm2:.4f} mF/cm²\")\n"
        "        else:\n"
        "            c3.metric(\"I_b\",      f\"{r.i_backward_peak:.4f} mA\")\n"
        "            c4.metric(\"I_f/I_b\",  f\"{r.if_ib_ratio:.3f}\" if not np.isnan(r.if_ib_ratio) else \"N/A\")\n"
        "\n"
        "        c5,c6,c7 = st.columns(3)\n"
        "        c5.metric(\"j_f (geometric)\", f\"{r.j_forward_peak:.4f} mA/cm²\")\n"
        "        c6.metric(\"j_b (geometric)\", f\"{r.j_backward_peak:.4f} mA/cm²\")\n"
        "        _ecsa_unit = \"cm²_BET\" if _is_mf else \"cm²_Pt\"\n"
        "        if r.ecsa>0: c7.metric(f\"j_f (ECSA)\", f\"{r.j_specific_forward:.4f} mA/{_ecsa_unit}\")\n"
        "\n"
        "        st.info(f\"**Interpretation:** {r.interpretation}\")"
    )
    NEW = (
        "        _e_onset_rhe = r.e_onset + e_ref_val + 0.059 * ph_value\n"
        "        c1,c2,c3,c4 = st.columns(4)\n"
        "        c1.metric(\"E_onset (vs ref)\", f\"{r.e_onset:.4f} V\")\n"
        "        c2.metric(\"E_onset (vs RHE)\", f\"{_e_onset_rhe:.4f} V\",\n"
        "                  help=f\"= {r.e_onset:.4f} + {e_ref_val:.3f}(ref) + 0.059×{ph_value:.2f}(pH)\")\n"
        "        if _is_mf:\n"
        "            c3.metric(\"Net faradaic I\", f\"{r.net_faradaic_current_mA:.4f} mA\")\n"
        "            c4.metric(\"C_dl\",           f\"{r.cdl_mF_cm2:.4f} mF/cm²\")\n"
        "        else:\n"
        "            c3.metric(\"I_b\",     f\"{r.i_backward_peak:.4f} mA\")\n"
        "            c4.metric(\"I_f/I_b\", f\"{r.if_ib_ratio:.3f}\" if not np.isnan(r.if_ib_ratio) else \"N/A\")\n"
        "        c5,c6,c7,c8 = st.columns(4)\n"
        "        c5.metric(\"j_f (geometric)\", f\"{r.j_forward_peak:.4f} mA cm⁻²\")\n"
        "        if not _is_mf:\n"
        "            c6.metric(\"j_b (geometric)\", f\"{r.j_backward_peak:.4f} mA cm⁻²\")\n"
        "        _ecsa_unit = \"cm²_BET\" if _is_mf else \"cm²_Pt\"\n"
        "        if r.ecsa > 0:\n"
        "            c7.metric(\"j_f (ECSA)\", f\"{r.j_specific_forward:.4f} mA/{_ecsa_unit}\")\n"
        "        if loading > 0:\n"
        "            c8.metric(\"Mass activity\", f\"{r.j_forward_peak / loading:.1f} A/g\",\n"
        "                      help=\"j_f / loading\")\n"
        "        if \"cv_Q_total\" in st.session_state:\n"
        "            _Qf = st.session_state[\"cv_Q_f\"]; _Qb = st.session_state[\"cv_Q_b\"]\n"
        "            cq1,cq2,cq3 = st.columns(3)\n"
        "            cq1.metric(\"Q_forward (mC)\",  f\"{_Qf:.3f}\")\n"
        "            cq2.metric(\"Q_backward (mC)\", f\"{abs(_Qb):.3f}\")\n"
        "            cq3.metric(\"Q_f / |Q_b|\", f\"{abs(_Qf/_Qb):.3f}\" if _Qb != 0 else \"N/A\",\n"
        "                       help=\"≈1 = reversible\")\n"
        "        st.caption(f\"Alcohol: **{alcohol}** {alcohol_conc} M  |  \"\n"
        "                   f\"Electrolyte: **{elec_compound}** {elec_conc} M  |  pH = {ph_value:.2f}\")\n"
        "        st.info(f\"**Interpretation:** {r.interpretation}\")"
    )
    return patch(src, OLD, NEW, "Change 7")

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 8 — CV tab: plot j (mA cm⁻²) instead of I (mA)
# ─────────────────────────────────────────────────────────────────────────────
def change_8(src):
    OLD = (
        "        import plotly.graph_objects as go\n"
        "        fig = go.Figure()\n"
        "        x_plot = st.session_state.get(\"cv_pot_corr\", st.session_state[\"cv_pot\"])\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=x_plot, y=st.session_state[\"cv_cur\"],\n"
        "            mode=\"lines\", name=\"CV\" + (\" (iR-corrected)\" if actual_rs>0 else \"\"),\n"
        "            line=dict(color=\"#2563eb\", width=2),\n"
        "        ))\n"
        "        fig.add_vline(x=r.e_onset, line_dash=\"dash\", line_color=\"#d97706\",\n"
        "                      annotation_text=f\"E_onset = {r.e_onset:.3f} V\",\n"
        "                      annotation_font=dict(color=\"#d97706\"))\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=[r.e_forward_peak], y=[r.i_forward_peak], mode=\"markers\",\n"
        "            name=f\"I_f = {r.i_forward_peak:.3f} mA\",\n"
        "            marker=dict(color=\"#16a34a\", size=12, symbol=\"star\"),\n"
        "        ))\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=[r.e_backward_peak], y=[r.i_backward_peak], mode=\"markers\",\n"
        "            name=f\"I_b = {r.i_backward_peak:.3f} mA\",\n"
        "            marker=dict(color=\"#dc2626\", size=12, symbol=\"star\"),\n"
        "        ))\n"
        "        title = f\"CV — {sr_cv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}\"\n"
        "        if actual_rs>0: title += f\" | iR-corrected (R_s={actual_rs:.1f}Ω)\"\n"
        "        fig.update_layout(**PLOTLY_LAYOUT, title=title,\n"
        "                          xaxis_title=f\"Potential (V vs {e_ref_type})\",\n"
        "                          yaxis_title=\"Current (mA)\")\n"
        "        st.plotly_chart(fig, use_container_width=True)"
    )
    NEW = (
        "        import plotly.graph_objects as go\n"
        "        fig = go.Figure()\n"
        "        x_plot = st.session_state.get(\"cv_pot_corr\", st.session_state[\"cv_pot\"])\n"
        "        _cur_plot = st.session_state[\"cv_cur\"]\n"
        "        j_arr = _cur_plot / area if area > 0 else _cur_plot\n"
        "        fig.add_trace(go.Scatter(x=x_plot, y=j_arr, mode=\"lines\",\n"
        "            name=\"CV\" + (\" (iR-corrected)\" if actual_rs>0 else \"\"),\n"
        "            line=dict(color=\"#2563eb\", width=2)))\n"
        "        fig.add_vline(x=r.e_onset, line_dash=\"dash\", line_color=\"#d97706\",\n"
        "            annotation_text=f\"E_onset = {r.e_onset:.3f} V\",\n"
        "            annotation_font=dict(color=\"#d97706\"))\n"
        "        if r.e_forward_peak is not None:\n"
        "            fig.add_trace(go.Scatter(x=[r.e_forward_peak], y=[r.i_forward_peak/area],\n"
        "                mode=\"markers\", name=f\"j_f = {r.i_forward_peak/area:.3f} mA cm⁻²\",\n"
        "                marker=dict(color=\"#16a34a\", size=12, symbol=\"star\")))\n"
        "        if r.e_backward_peak is not None and not _is_mf:\n"
        "            fig.add_trace(go.Scatter(x=[r.e_backward_peak], y=[r.i_backward_peak/area],\n"
        "                mode=\"markers\", name=f\"j_b = {r.i_backward_peak/area:.3f} mA cm⁻²\",\n"
        "                marker=dict(color=\"#dc2626\", size=12, symbol=\"star\")))\n"
        "        title = f\"j vs E — {sr_cv} mV/s | {temperature}°C | {catalyst or 'Catalyst'}\"\n"
        "        if actual_rs>0: title += f\" | iR-corrected (R_s={actual_rs:.1f}Ω)\"\n"
        "        fig.update_layout(**PLOTLY_LAYOUT, title=title,\n"
        "            xaxis_title=f\"E (V vs {e_ref_type})\", yaxis_title=\"j (mA cm⁻²)\")\n"
        "        st.plotly_chart(fig, use_container_width=True)"
    )
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
    print(f"   Total lines: {src.count(chr(10))}")

    for fn, msg in CHANGES:
        print(f"\n🔧 Applying: {msg}")
        src = fn(src)
        write(src)
        commit(msg)
        print(f"   Lines now: {src.count(chr(10))}")

    print("\n✅ All 9 changes applied and committed.")
    print("👉 Run:  git push origin main")
