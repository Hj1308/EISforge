# patch17_chronoamperometry_tab.py
# Adds an 8th tab "Chronoamperometry" to app.py.
#   - Loads a single Ivium IDF CA file (time, current) via a small local reader
#   - i-t plot (raw current or current density, user toggle)
#   - Descriptive stability metrics via eisforge.analysis.ca_analyzer
#   - Excel export
import shutil, sys

PATH = r"app.py"
s = open(PATH, encoding="utf-8").read()

# ── 1. extend the tabs() call from 7 to 8 tabs ───────────────────────────────
OLD_TABS = '''tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "\U0001F4C8 CV Analysis", "\U0001F4C9 LSV Analysis", "\U0001F52C EIS Analysis",
    "\U0001F916 EIS-GPT", "\U0001F517 Correlation", "\u2697\ufe0f K-L Analysis",
    "\U0001F4CA Scan-Rate Kinetics"
])'''
NEW_TABS = '''tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "\U0001F4C8 CV Analysis", "\U0001F4C9 LSV Analysis", "\U0001F52C EIS Analysis",
    "\U0001F916 EIS-GPT", "\U0001F517 Correlation", "\u2697\ufe0f K-L Analysis",
    "\U0001F4CA Scan-Rate Kinetics", "\u23F1\ufe0f Chronoamperometry"
])'''

TAB_BODY = '''

# ══════════════════ CHRONOAMPEROMETRY ═════════════════════════════════════════
with tab8:
    st.markdown('<h3>Chronoamperometry (i\u2013t Stability)</h3>', unsafe_allow_html=True)
    st.caption(
        "Upload a single chronoamperometry file (Ivium .idf: time, current). "
        "Reports descriptive operational-stability metrics \u2014 current retention, "
        "steady-state current, and initial drop. No Cottrell/diffusion fit is "
        "performed (early decay is largely capacitive, not catalyst loss)."
    )

    ca_file = st.file_uploader("Upload CA file", type=["idf", "csv", "txt"],
                               key="ca_up")
    c1, c2 = st.columns(2)
    ca_per_area = c1.checkbox("Show current density (per area)", value=True)
    ca_area = c2.number_input("Geometric area (cm\u00b2)", value=0.07068583,
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

            st.success(f"\u2705 Loaded {len(t)} points | {ca_file.name}")

            res = analyze_ca(t, i, area_cm2=float(ca_area),
                             per_area=bool(ca_per_area))

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.time, y=res.current, mode="lines",
                                     name="i\u2013t", line=dict(color="#2563eb")))
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
                    "\U0001F4E5 Download CA results (Excel)",
                    data=buf.getvalue(),
                    file_name="eisforge_ca_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")

        except Exception as e:
            st.error(f"Chronoamperometry analysis error: {e}")
'''

n = 0
if NEW_TABS in s:
    print("[1/2] tabs already updated")
else:
    if OLD_TABS not in s:
        print("ERROR step 1: OLD tabs block not found. Aborting.")
        sys.exit(1)
    s = s.replace(OLD_TABS, NEW_TABS, 1)
    n += 1
    print("[1/2] OK — added tab8 to st.tabs()")

if "with tab8:" in s:
    print("[2/2] tab body already present")
else:
    s = s.rstrip() + "\n" + TAB_BODY
    n += 1
    print("[2/2] OK — appended tab8 body")

if n:
    shutil.copy(PATH, PATH + ".bak_patch17")
    open(PATH, "w", encoding="utf-8").write(s)
    print("Patched OK:", PATH, "(backup: app.py.bak_patch17)")
else:
    print("Nothing to do.")
