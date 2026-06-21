"""
analyze_cv.py — Import an alcohol-oxidation CV file and report the metrics
that metal-free (carbon-based) AOR papers publish.

This is the missing import layer for EISForge's CV engine. The analysis math
already lives in eisforge/analysis/cv_analyzer.py; this script only loads your
file, finds the potential/current columns, and runs the engine in metal-free
mode.

USAGE (Windows CMD, from the repo root):

    python analyze_cv.py my_cv.csv --electrolyte KOH --conc 1.0 ^
        --area 0.07 --scan-rate 50 --eref 0.098 --loading 0.2

KEY ARGUMENTS
    file              Path to a CV export (.csv, .txt, or Gamry .DTA).
    --electrolyte     KOH / NaOH / H2SO4 / HClO4 ... (default KOH).
    --conc            Electrolyte concentration, mol/L (default 1.0).
    --area            Geometric electrode area, cm^2 (default 1.0).
    --scan-rate       Scan rate, mV/s (default 50).
    --current-unit    A / mA / uA  (default: auto-detected, falls back to mA).
    --eref            Reference-electrode potential vs RHE, in V. Add this to
                      convert E_onset to the vs-RHE scale papers use.
                      e.g. Ag/AgCl(sat. KCl) ~ 0.197 + 0.059*pH
                           Hg/HgO (1 M KOH)  ~ 0.098 + 0.059*pH
    --loading         Catalyst loading, mg/cm^2 (for mass activity).
    --bet             BET / ECSA surface area, cm^2 (for specific activity).
    --cycle           Which cycle to use for multi-cycle files (default: last).

Author: helper script for the B4C/graphene metal-free AOR project.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Engine that already exists in the repo
from eisforge.analysis.cv_analyzer import (
    CVAnalyzer,
    ElectrolyteInfo,
    CATALYST_METAL_FREE,
)

# ── Column-name aliases (lower-cased, punctuation-tolerant) ──────────────────
_POT_ALIASES = {
    "e", "potential", "ewe", "ewe/v", "voltage", "vf", "e/v", "e(v)",
    "v", "potential/v", "potential(v)", "ewe(v)", "e_we", "u",
}
_CUR_ALIASES = {
    "i", "current", "im", "i/ma", "i(ma)", "i/a", "i(a)", "current/a",
    "current(a)", "current/ma", "j", "j/macm2", "i_we", "we(1).current",
}


def _norm(name: str) -> str:
    """Normalise a column header for alias matching."""
    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def _infer_current_unit(col_name: str, fallback: str) -> str:
    n = _norm(col_name)
    if "ma" in n:
        return "mA"
    if "ua" in n or "µa" in n:
        return "uA"
    if n.endswith("/a") or n.endswith("(a)") or n == "i" or "currenta" in n:
        return "A"
    return fallback


# ── Gamry .DTA (CV CURVE block) loader ───────────────────────────────────────
def _load_gamry_dta(path: Path, cycle: int):
    """Parse the CURVE block(s) of a Gamry CV .DTA file. Vf=V, Im=A."""
    lines = path.read_text(errors="ignore").splitlines()
    blocks = []  # list of (header_cols, data_rows)
    i = 0
    while i < len(lines):
        if lines[i].lstrip().upper().startswith("CURVE"):
            # next line = column names, line after = units, then data
            header = lines[i + 1].split("\t")
            header = [h.strip() for h in header]
            rows = []
            j = i + 3
            while j < len(lines):
                parts = lines[j].split("\t")
                if len(parts) < 3 or not _is_number(parts[1] if len(parts) > 1 else ""):
                    break
                rows.append(parts)
                j += 1
            blocks.append((header, rows))
            i = j
        else:
            i += 1
    if not blocks:
        return None
    header, rows = blocks[cycle if -len(blocks) <= cycle < len(blocks) else -1]
    df = pd.DataFrame(rows, columns=header[: len(rows[0])]) if rows else pd.DataFrame()
    # Gamry: Vf (V), Im (A)
    vf = _pick(df, ["vf", "v"])
    im = _pick(df, ["im", "i"])
    if vf is None or im is None:
        return None
    e = pd.to_numeric(df[vf], errors="coerce").to_numpy()
    cur = pd.to_numeric(df[im], errors="coerce").to_numpy()
    mask = ~(np.isnan(e) | np.isnan(cur))
    return e[mask], cur[mask], "A", len(blocks)


def _pick(df, candidates):
    cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ── Ivium .idf (CyclicVoltammetry) loader ────────────────────────────────────
def _load_ivium_idf(path: Path):
    """
    Parse an Ivium .idf file. The data block after 'primary_data' holds rows of
    three scientific-notation columns:  E_applied(V)  I(A)  E_measured(V).
    Also reads scan rate and area from the metadata when present.
    Returns (E, I, 'A', meta_dict) or None if it is not an Ivium CV file.
    """
    raw = path.read_bytes()
    text = raw.decode("latin-1")
    if "primary_data" not in text:
        return None

    rows = re.findall(
        r"(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)", text
    )
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    e_applied = arr[:, 0]          # clean staircase potential (V)
    current_a = arr[:, 1]          # current (A)

    # Pull useful metadata
    meta = {}
    for key, cast in (("Scanrate", float), ("Data Options.Area", float),
                      ("N scans", int), ("Current Range", str)):
        m = re.search(rf"{re.escape(key)}=([^\r\n]+)", text)
        if m:
            try:
                meta[key] = cast(m.group(1).strip())
            except ValueError:
                meta[key] = m.group(1).strip()
    return e_applied, current_a, "A", meta


# ── Generic CSV / TXT loader ─────────────────────────────────────────────────
def _load_delimited(path: Path, current_unit_arg: str):
    # Try with a header first
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    df.columns = [str(c) for c in df.columns]
    norm_map = {_norm(c): c for c in df.columns}

    pot_col = next((norm_map[a] for a in _POT_ALIASES if a in norm_map), None)
    cur_col = next((norm_map[a] for a in _CUR_ALIASES if a in norm_map), None)

    # No recognisable headers → assume first two numeric columns are E, I
    if pot_col is None or cur_col is None:
        df2 = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
        num = df2.apply(pd.to_numeric, errors="coerce")
        good = [c for c in num.columns if num[c].notna().mean() > 0.8]
        if len(good) < 2:
            raise ValueError(
                f"Could not find potential & current columns. "
                f"Headers seen: {list(df.columns)}"
            )
        pot_col, cur_col = good[0], good[1]
        e = num[pot_col].to_numpy()
        cur = num[cur_col].to_numpy()
        unit = current_unit_arg or "mA"
        print(f"  [no headers] using column {pot_col}=E, column {cur_col}=I")
    else:
        e = pd.to_numeric(df[pot_col], errors="coerce").to_numpy()
        cur = pd.to_numeric(df[cur_col], errors="coerce").to_numpy()
        unit = current_unit_arg or _infer_current_unit(cur_col, "mA")
        print(f"  detected  E column = '{pot_col}'  |  I column = '{cur_col}'  |  unit = {unit}")

    mask = ~(np.isnan(e) | np.isnan(cur))
    return e[mask], cur[mask], unit


def load_cv(path: Path, current_unit_arg: str, cycle: int):
    meta = {}
    if path.suffix.lower() == ".idf":
        out = _load_ivium_idf(path)
        if out is not None:
            e, cur, unit, meta = out
            sr = meta.get("Scanrate")
            print(f"  Ivium .idf: {len(e)} points"
                  + (f" | scan rate {sr*1000:.0f} mV/s (from file)" if sr else "")
                  + (f" | N scans {meta['N scans']}" if "N scans" in meta else ""))
            return e, cur, (current_unit_arg or unit), meta
        # Not Ivium → maybe Autolab; try Gamry/delimited paths below
        print("  .idf is not an Ivium CV file — trying generic parsing")
    if path.suffix.lower() == ".dta":
        out = _load_gamry_dta(path, cycle)
        if out is not None:
            e, cur, unit, n = out
            print(f"  Gamry .DTA: {n} cycle(s) found, using cycle index {cycle}")
            return e, cur, (current_unit_arg or unit), meta
        print("  .DTA had no CURVE block — falling back to delimited parsing")
    e, cur, unit = _load_delimited(path, current_unit_arg)
    return e, cur, unit, meta


# ── Main ──────────────────────────────────────────────────────────────────────
def _subtract_blank(e_s, i_s_ma, e_b, i_b_ma):
    """
    Subtract a blank CV from a sample CV, matching by potential *within each
    scan branch* (forward and reverse separately, since CV current is
    multivalued in E). The blank is interpolated onto the sample's potentials.
    Returns net current (mA) aligned with e_s.
    """
    vs = int(np.argmax(e_s))
    vb = int(np.argmax(e_b))

    def interp_branch(es, eb, ib):
        order = np.argsort(eb)
        return np.interp(es, eb[order], ib[order])

    net = i_s_ma.copy()
    # forward branch
    net[: vs + 1] -= interp_branch(e_s[: vs + 1], e_b[: vb + 1], i_b_ma[: vb + 1])
    # reverse branch
    if vs + 1 < len(e_s) and vb + 1 < len(e_b):
        net[vs + 1:] -= interp_branch(e_s[vs + 1:], e_b[vb + 1:], i_b_ma[vb + 1:])
    return net


_UNIT_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3}


def main():
    ap = argparse.ArgumentParser(description="Metal-free AOR CV analysis (EISForge)")
    ap.add_argument("file", help="CV file: .csv, .txt, or Gamry .DTA")
    ap.add_argument("--electrolyte", default="KOH")
    ap.add_argument("--conc", type=float, default=1.0)
    ap.add_argument("--area", type=float, default=0.0, help="geometric area, cm^2 (overrides --diameter-mm)")
    ap.add_argument("--diameter-mm", type=float, default=0.0,
                    help="electrode disk diameter in mm; area = pi*(d/2)^2 computed for you")
    ap.add_argument("--scan-rate", type=float, default=50.0, help="mV/s")
    ap.add_argument("--current-unit", default=None, choices=[None, "A", "mA", "uA"])
    ap.add_argument("--eref", type=float, default=0.0,
                    help="reference electrode potential vs RHE (V)")
    ap.add_argument("--loading", type=float, default=0.0, help="catalyst loading, mg/cm^2 (overrides --mass-ug)")
    ap.add_argument("--mass-ug", type=float, default=0.0,
                    help="catalyst mass deposited on the electrode, in micrograms; "
                         "loading and mass activity computed for you")
    ap.add_argument("--bet", type=float, default=0.0, help="BET/ECSA area, cm^2")
    ap.add_argument("--rs", type=float, default=0.0, help="solution resistance for iR (ohm)")
    ap.add_argument("--cycle", type=int, default=-1, help="cycle index (default last)")
    ap.add_argument("--onset", default="derivative",
                    choices=["tangent", "threshold", "derivative"])
    ap.add_argument("--no-bg", action="store_true",
                    help="skip the capacitive-background subtraction "
                         "(use for irreversible AOR waves that start mid-transient)")
    ap.add_argument("--report-at", type=float, default=None,
                    help="report current density at this potential (V, raw scale); "
                         "use when the oxidation wave has no closed peak in the window")
    ap.add_argument("--blank", default=None,
                    help="path to a blank CV (same electrolyte WITHOUT alcohol, same "
                         "electrode & window); its current is subtracted per scan branch "
                         "to isolate the net faradaic (alcohol) current")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    print(f"\nLoading: {path.name}")
    e, cur, unit, meta = load_cv(path, args.current_unit, args.cycle)
    print(f"  loaded {len(e)} points | E range {e.min():.3f}..{e.max():.3f} V "
          f"| I range {cur.min():.4g}..{cur.max():.4g} {unit}\n")

    # ── Optional blank subtraction (isolates the alcohol-oxidation current) ───
    if args.blank:
        bpath = Path(args.blank)
        if not bpath.exists():
            sys.exit(f"Blank file not found: {bpath}")
        print(f"Loading blank: {bpath.name}")
        e_b, cur_b, unit_b, _ = load_cv(bpath, None, args.cycle)
        cur_ma = cur * _UNIT_MA.get(unit, 1.0)
        cur_b_ma = cur_b * _UNIT_MA.get(unit_b, 1.0)
        cur = _subtract_blank(e, cur_ma, e_b, cur_b_ma)   # net, in mA
        unit = "mA"
        print(f"  blank subtracted per scan branch -> net current isolated\n")

    # Auto-fill scan rate / area from file metadata unless user overrode them
    scan_rate = args.scan_rate
    if "Scanrate" in meta and args.scan_rate == 50.0:
        scan_rate = meta["Scanrate"] * 1000.0   # V/s → mV/s
    # ── Resolve geometric area (cm^2) ────────────────────────────────────────
    # priority: explicit --area  >  --diameter-mm  >  file metadata  >  1.0
    import math
    area_from_d = math.pi * (args.diameter_mm / 20.0) ** 2 if args.diameter_mm > 0 else 0.0
    if args.area > 0:
        area = args.area
    elif area_from_d > 0:
        area = area_from_d
        print(f"  electrode {args.diameter_mm:g} mm disk -> geometric area = {area:.4f} cm^2")
    elif "Data Options.Area" in meta:
        area = meta["Data Options.Area"]
    else:
        area = 1.0

    # ── Resolve catalyst loading (mg/cm^2) and total deposited mass (mg) ──────
    # priority: explicit --loading  >  --mass-ug
    if args.loading > 0:
        loading = args.loading
        mass_mg = loading * area
    elif args.mass_ug > 0:
        mass_mg = args.mass_ug / 1000.0
        loading = mass_mg / area
        print(f"  deposited mass {args.mass_ug:g} ug -> loading = {loading:.4f} mg/cm^2")
    else:
        loading = 0.0
        mass_mg = 0.0

    media = "alkaline" if args.electrolyte.upper() in {"KOH", "NAOH", "NA2CO3", "NH3"} else "acidic"
    electrolyte = ElectrolyteInfo(media=media, compound=args.electrolyte, concentration=args.conc)

    analyzer = CVAnalyzer(
        scan_rate=scan_rate,
        electrode_area=area,
        ecsa=args.bet,
        onset_method=args.onset,
        electrolyte=electrolyte,
        catalyst_type=CATALYST_METAL_FREE,   # ← metal-free (B4C/graphene, carbon)
        current_unit=unit,
        catalyst_loading=loading,
        e_ref_vs_rhe=args.eref,
    )

    # --no-bg (also implied by --blank): replace the capacitive-background step
    # with a no-op. Blank subtraction already removes the background, and the
    # auto linear-box subtraction misfires when the scan starts mid-transient.
    if args.no_bg or args.blank:
        analyzer._subtract_capacitive_background = (
            lambda potential, current_ma: (current_ma, np.zeros_like(current_ma))
        )
        why = "blank subtraction active" if args.blank else "--no-bg"
        print(f"  [{why}] capacitive-background subtraction disabled\n")

    res = analyzer.analyze(e, cur, r_s_ohms=args.rs)

    print(res.summary())

    # ── Paper-ready extras the summary doesn't compute directly ──────────────
    print("\n  ── Reporting helpers (vs RHE / activity) ──")
    if args.eref != 0.0:
        ph = (14 + np.log10(max(args.conc, 1e-6))) if media == "alkaline" else 0.0
        ph = min(ph, 14.0)
        e_onset_rhe = res.e_onset + args.eref + 0.059 * ph
        e_peak_rhe = res.e_forward_peak + args.eref + 0.059 * ph
        print(f"  E_onset (vs RHE)    = {e_onset_rhe:.3f} V   (pH~{ph:.1f})")
        print(f"  E_peak  (vs RHE)    = {e_peak_rhe:.3f} V")
    else:
        print("  (pass --eref to convert E_onset / E_peak to the vs-RHE scale)")

    if mass_mg > 0:
        mass_act = res.i_forward_peak / mass_mg  # mA/mg = A/g
        print(f"  Mass activity       = {mass_act:.1f} mA/mg = A/g  "
              f"(peak I / {mass_mg*1000:.0f} ug catalyst)")
    if args.bet > 0:
        print(f"  Specific activity   = {res.j_specific_forward:.4f} mA/cm2_BET")

    # Fixed-potential reporting (for waves with no closed peak in the window)
    if args.report_at is not None:
        vertex = int(np.argmax(e))
        e_fwd = e[: vertex + 1]
        cur_ma = cur * {"A": 1000.0, "mA": 1.0, "uA": 1e-3}.get(unit, 1.0)
        i_fwd = cur_ma[: vertex + 1]
        idx = int(np.argmin(np.abs(e_fwd - args.report_at)))
        i_at = float(i_fwd[idx])
        print(f"\n  ── Fixed-potential report @ {args.report_at:+.3f} V ──")
        print(f"  I  @ {e_fwd[idx]:+.3f} V   = {i_at:.4f} mA")
        print(f"  j  @ {e_fwd[idx]:+.3f} V   = {i_at / area:.2f} mA/cm2 (geometric)")
        if mass_mg > 0:
            print(f"  mass activity        = {i_at / mass_mg:.1f} mA/mg  (= A/g)")

    print()


if __name__ == "__main__":
    main()
