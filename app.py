"""
EISForge — رابط کاربری Streamlit با پشتیبانی از همه فرمت‌ها.

فرمت‌های پشتیبانی‌شده:
    EIS:  .idf (Autolab)، .dta (Gamry)، .mpt/.mpr (BioLogic)، .csv، .txt
    CV:   .csv، .txt، .idf

نویسنده: Hoda Jafari | May 2026
اجرا: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="EISForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root {
    --bg:#0a0e1a; --surface:#111827; --border:#1e293b;
    --accent:#38bdf8; --accent2:#818cf8; --success:#34d399;
    --warning:#fbbf24; --danger:#f87171; --text:#e2e8f0; --muted:#64748b;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.title{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(135deg,#38bdf8,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-2px;margin:0;text-align:center;}
.subtitle{color:var(--muted);font-size:.9rem;font-family:'JetBrains Mono',monospace;text-align:center;margin-top:.4rem;}
.section-title{font-size:.65rem;text-transform:uppercase;letter-spacing:3px;color:var(--muted);margin-bottom:.8rem;font-family:'JetBrains Mono',monospace;}
.stButton>button{background:linear-gradient(135deg,#38bdf8,#818cf8)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem;}
.format-badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.7rem;font-family:'JetBrains Mono',monospace;background:rgba(56,189,248,.15);color:#38bdf8;border:1px solid #38bdf8;margin:.1rem;}
</style>
""", unsafe_allow_html=True)

# ── هدر ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="title">⚡ EISForge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Physics-Informed ML · by Hoda Jafari · 2026 · '
    '<a href="https://github.com/Hj1308/EISforge-" style="color:#38bdf8">GitHub</a></p>',
    unsafe_allow_html=True,
)

# فرمت‌های پشتیبانی‌شده
st.markdown("""
<div style="text-align:center;margin:.5rem 0 1rem 0;">
<span class="format-badge">.idf Autolab</span>
<span class="format-badge">.dta Gamry</span>
<span class="format-badge">.mpt BioLogic</span>
<span class="format-badge">.csv Generic</span>
<span class="format-badge">.txt Generic</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── فرمت‌های پشتیبانی‌شده ─────────────────────────────────────────────────────
EIS_FORMATS = ["idf", "dta", "mpt", "mpr", "csv", "txt", "isf", "ism"]
CV_FORMATS  = ["idf", "csv", "txt", "dta"]


def load_eis_file(uploaded_file) -> tuple:
    """بارگذاری فایل EIS بر اساس پسوند."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import tempfile, os

    suffix = Path(uploaded_file.name).suffix.lower()

    # ذخیره موقت فایل
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".idf":
            from eisforge.parsers.autolab_parser import AutolabIDFParser
            dataset = AutolabIDFParser().parse(tmp_path)
            return dataset.frequency, dataset.z_real, dataset.z_imag, dataset.metadata

        elif suffix == ".dta":
            from eisforge.parsers.gamry_parser import GamryParser
            dataset = GamryParser().parse(tmp_path)
            return dataset.frequency, dataset.z_real, dataset.z_imag, dataset.metadata

        elif suffix in (".mpt", ".mpr"):
            # BioLogic — از galvani استفاده می‌کنیم
            try:
                from galvani import BioLogic
                mpr = BioLogic.MPRfile(tmp_path)
                df = mpr.DF
                return (
                    df["freq/Hz"].to_numpy(),
                    df["Re(Z)/Ohm"].to_numpy(),
                    -df["-Im(Z)/Ohm"].to_numpy(),
                    {"source": "BioLogic"},
                )
            except Exception:
                raise ValueError(
                    "خطا در خواندن فایل BioLogic. "
                    "مطمئن شوید galvani نصب است: pip install galvani"
                )

        else:
            # CSV/TXT عمومی
            df = pd.read_csv(tmp_path, comment="#", sep=None, engine="python")
            cols = df.columns.tolist()
            freq   = df[cols[0]].to_numpy(dtype=float)
            z_real = df[cols[1]].to_numpy(dtype=float)
            z_imag = df[cols[2]].to_numpy(dtype=float)
            # auto-sign
            if z_imag.mean() < 0:
                z_imag = -z_imag
            return freq, z_real, z_imag, {}

    finally:
        os.unlink(tmp_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">⚙ تنظیمات سیستم</p>', unsafe_allow_html=True)

    system_type = st.selectbox(
        "نوع سیستم",
        ["AOR", "Battery", "Corrosion", "Fuel Cell", "Biosensor"],
    )

    catalyst = st.text_input("کاتالیست", placeholder="Pt/C، PdAu، PtRu...")

    electrolyte = st.selectbox(
        "محیط الکتروشیمیایی",
        ["اسیدی (H₂SO₄)", "بازی (KOH)", "NaCl", "PBS", "سایر"],
    )
    electrolyte_key = (
        "acidic" if "اسیدی" in electrolyte
        else "alkaline" if "بازی" in electrolyte
        else electrolyte.lower()
    )

    alcohol = st.selectbox(
        "الکل (برای AOR)",
        ["ethanol", "methanol", "ethylene glycol", "glycerol", "-"],
        disabled=(system_type != "AOR"),
    )

    eis_potential = st.number_input(
        "پتانسیل EIS (V)", value=0.5, step=0.01,
        help="پتانسیلی که EIS در آن گرفته شده",
    )

    st.markdown("---")

    # ── راهنمای ادبیات ────────────────────────────────────────────────────────
    if st.button("📚 راهنمای ادبیات"):
        try:
            from eisforge.knowledge.literature_engine import LiteratureEngine
            engine = LiteratureEngine()
            guess = engine.query(
                system_type=system_type,
                catalyst=catalyst,
                electrolyte=electrolyte_key,
                alcohol=alcohol if system_type == "AOR" else "",
                potential=eis_potential,
            )
            st.session_state["lit_guess"] = guess
        except Exception as e:
            st.error(f"خطا: {e}")

    if "lit_guess" in st.session_state:
        g = st.session_state["lit_guess"]
        if g.system_found:
            st.success(f"✅ {g.system_name}")
            st.code(f"مدار: {g.recommended_circuit}")
            st.metric("اطمینان", g.confidence)
            if g.warnings:
                for w in g.warnings:
                    st.warning(w)
        else:
            st.warning("سیستم در دیتابیس نیست")


# ── تب‌ها ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 تحلیل CV",
    "🔬 تحلیل EIS",
    "🤖 EIS-GPT",
    "🔗 همبستگی",
])


# ══════════════════════════════════════════════════════════════════
# تب ۱ — CV
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📈 آپلود داده CV</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        cv_file = st.file_uploader(
            "فایل CV",
            type=CV_FORMATS,
            help="فرمت‌های پشتیبانی‌شده: .idf .csv .txt .dta",
        )
        scan_rate = st.number_input("سرعت scan (mV/s)", value=50, min_value=1)
        onset_method = st.radio(
            "روش E_onset",
            ["tangent", "threshold", "derivative"],
            horizontal=True,
        )

    with col2:
        if cv_file:
            import pandas as pd
            import numpy as np
            try:
                suffix = cv_file.name.split(".")[-1].lower()
                if suffix == "idf":
                    freq, z_real, z_imag, meta = load_eis_file(cv_file)
                    # برای CV از idf — فرض می‌کنیم potential و current دارد
                    st.info("فایل IDF بارگذاری شد — ستون اول=پتانسیل، ستون دوم=جریان")
                    potential = z_real  # در IDF مستقیم CV: col1=E, col2=I
                    current   = z_imag
                else:
                    df = pd.read_csv(cv_file, comment="#", sep=None, engine="python")
                    cols = df.columns.tolist()
                    potential = df[cols[0]].to_numpy(dtype=float)
                    current   = df[cols[1]].to_numpy(dtype=float)

                st.success(f"✅ {len(potential)} نقطه بارگذاری شد")

                from eisforge.analysis.cv_analyzer import CVAnalyzer
                analyzer = CVAnalyzer(
                    onset_method=onset_method,
                    electrolyte=electrolyte_key,
                )
                cv_result = analyzer.analyze(potential, current, scan_rate)
                st.session_state["cv_result"]   = cv_result
                st.session_state["cv_potential"] = potential
                st.session_state["cv_current"]   = current

            except Exception as e:
                st.error(f"❌ {e}")

    if "cv_result" in st.session_state:
        r = st.session_state["cv_result"]
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E_onset", f"{r.e_onset:.4f} V")
        c2.metric("I_f", f"{r.i_forward_peak:.3f} mA")
        c3.metric("I_b", f"{r.i_backward_peak:.3f} mA")
        c4.metric("I_f/I_b", f"{r.if_ib_ratio:.3f}")
        st.info(f"💡 {r.interpretation}")

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state["cv_potential"],
            y=st.session_state["cv_current"],
            mode="lines", name="CV",
            line=dict(color="#38bdf8", width=2),
        ))
        fig.add_vline(x=r.e_onset, line_dash="dash",
                      line_color="#fbbf24",
                      annotation_text=f"E_onset={r.e_onset:.3f}V",
                      annotation_font_color="#fbbf24")
        fig.add_trace(go.Scatter(
            x=[r.e_forward_peak], y=[r.i_forward_peak],
            mode="markers", name="I_f",
            marker=dict(color="#34d399", size=12, symbol="star"),
        ))
        fig.add_trace(go.Scatter(
            x=[r.e_backward_peak], y=[r.i_backward_peak],
            mode="markers", name="I_b",
            marker=dict(color="#f87171", size=12, symbol="star"),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827", title="Cyclic Voltammogram",
            xaxis_title="Potential (V)", yaxis_title="Current (mA)",
            font=dict(family="JetBrains Mono"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# تب ۲ — EIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔬 آپلود داده EIS</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        eis_file = st.file_uploader(
            "فایل EIS",
            type=EIS_FORMATS,
            help="فرمت‌های پشتیبانی‌شده: .idf .dta .mpt .mpr .csv .txt",
        )

        # حدس از ادبیات
        lit_circuit = "R0-p(R1,CPE1)-W1"
        lit_guess_str = "10, 200, 1e-5, 0.85, 30"
        if "lit_guess" in st.session_state and st.session_state["lit_guess"].system_found:
            g = st.session_state["lit_guess"]
            lit_circuit   = g.recommended_circuit
            lit_guess_str = ", ".join(f"{v:.3e}" for v in g.initial_guess.values())

        circuit_str = st.text_input("مدار معادل", value=lit_circuit)
        p0_str = st.text_input(
            "Initial Guess",
            value=lit_guess_str,
            help="از دکمه 📚 در sidebar حدس خودکار بگیرید",
        )

        if "lit_guess" in st.session_state and st.session_state["lit_guess"].system_found:
            st.info("💡 حدس اولیه از ادبیات علمی پر شد!")

    with col2:
        if eis_file:
            try:
                freq, z_real, z_imag, meta = load_eis_file(eis_file)
                st.success(f"✅ {len(freq)} نقطه — {eis_file.name}")
                if meta:
                    with st.expander("Metadata"):
                        for k, v in meta.items():
                            st.text(f"{k}: {v}")
                st.session_state["eis_freq"]   = freq
                st.session_state["eis_z_real"] = z_real
                st.session_state["eis_z_imag"] = z_imag

                # K-K validation خودکار
                import numpy as np
                from eisforge.parsers.base_parser import EISDataset
                from eisforge.core.validators import KramersKronigValidator
                dataset = EISDataset(
                    frequency=freq, z_real=z_real, z_imag=z_imag
                )
                kk = KramersKronigValidator().validate(dataset)
                if kk.passed:
                    st.success(f"✅ K-K: {kk.summary()}")
                else:
                    st.warning(f"⚠️ K-K: {kk.summary()}")

            except Exception as e:
                st.error(f"❌ {e}")

    if "eis_freq" in st.session_state:
        if st.button("🚀 اجرای CNLS Fit"):
            with st.spinner("در حال فیت..."):
                try:
                    import numpy as np
                    from eisforge.parsers.base_parser import EISDataset
                    from eisforge.core.fitter import CNLSFitter

                    p0 = [float(x.strip()) for x in p0_str.split(",")]
                    dataset = EISDataset(
                        frequency=st.session_state["eis_freq"],
                        z_real=st.session_state["eis_z_real"],
                        z_imag=st.session_state["eis_z_imag"],
                    )
                    fitter = CNLSFitter(circuit_str, p0)
                    fit = fitter.fit(dataset)
                    st.session_state["fit_result"] = fit
                    if fit.converged:
                        st.success(f"✅ χ²={fit.chi_squared:.6f}")
                    else:
                        st.warning(f"⚠️ فیت همگرا نشد — χ²={fit.chi_squared:.4f}")
                except Exception as e:
                    st.error(f"❌ {e}")

        if "fit_result" in st.session_state:
            import pandas as pd
            fit = st.session_state["fit_result"]
            rows = [
                {"پارامتر": n, "مقدار": f"{v:.4e}",
                 "خطا": f"{fit.parameter_errors.get(n, float('nan')):.2e}"}
                for n, v in fit.parameters.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            import plotly.graph_objects as go
            freq   = st.session_state["eis_freq"]
            z_real = st.session_state["eis_z_real"]
            z_imag = st.session_state["eis_z_imag"]

            fig_ny = go.Figure()
            fig_ny.add_trace(go.Scatter(
                x=z_real, y=z_imag, mode="markers",
                name="داده", marker=dict(color="#38bdf8", size=8),
            ))
            if fit.z_fit is not None:
                fig_ny.add_trace(go.Scatter(
                    x=fit.z_fit.real, y=-fit.z_fit.imag,
                    mode="lines", name="فیت",
                    line=dict(color="#f87171", width=2, dash="dash"),
                ))
            fig_ny.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827", title="Nyquist Plot",
                xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)",
                font=dict(family="JetBrains Mono"),
            )
            st.plotly_chart(fig_ny, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# تب ۳ — EIS-GPT
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">🤖 EIS-GPT</p>', unsafe_allow_html=True)
    st.info("مدل Transformer مدار را از طیف EIS پیش‌بینی می‌کند — بدون نیاز به حدس!")

    if "eis_freq" in st.session_state:
        if st.button("🧠 پیش‌بینی"):
            with st.spinner("EIS-GPT در حال تحلیل..."):
                try:
                    import torch
                    from eisforge.ml.eis_gpt.transformer import EISForgeModel
                    model = EISForgeModel(d_model=128, n_heads=8, n_layers=6)
                    freq   = torch.tensor(st.session_state["eis_freq"]).float().unsqueeze(0)
                    z_real = torch.tensor(st.session_state["eis_z_real"]).float().unsqueeze(0)
                    z_imag = torch.tensor(st.session_state["eis_z_imag"]).float().unsqueeze(0)
                    result = model.predict(freq, z_real, z_imag)
                    st.session_state["gpt_result"] = result
                except Exception as e:
                    st.error(f"❌ {e}")

        if "gpt_result" in st.session_state:
            res = st.session_state["gpt_result"]
            c1, c2 = st.columns(2)
            c1.metric("مدار پیش‌بینی‌شده", res["predicted_circuit"])
            c2.metric("اطمینان", f"{res['confidence']*100:.1f}%")
            st.markdown("**سه کاندید برتر:**")
            for c in res["top3"]:
                st.progress(c["probability"],
                            text=f"{c['circuit']} — {c['probability']*100:.1f}%")
    else:
        st.warning("ابتدا داده EIS را در تب 'تحلیل EIS' آپلود کنید.")


# ══════════════════════════════════════════════════════════════════
# تب ۴ — همبستگی
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🔗 همبستگی EIS-CV</p>', unsafe_allow_html=True)

    has_cv  = "cv_result"  in st.session_state
    has_fit = "fit_result" in st.session_state

    if not has_cv:
        st.warning("ابتدا CV را تحلیل کنید.")
    if not has_fit:
        st.warning("ابتدا EIS را فیت کنید.")

    if has_cv and has_fit:
        if st.button("🔗 تحلیل همبستگی"):
            try:
                from eisforge.analysis.eis_cv_correlator import EISCVCorrelator
                correlator = EISCVCorrelator(electrolyte=electrolyte_key)
                corr = correlator.correlate(
                    cv_result=st.session_state["cv_result"],
                    eis_fit_result=st.session_state["fit_result"],
                    eis_potential=eis_potential,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("E_onset", f"{corr.e_onset:.4f} V")
                c2.metric("ناحیه EIS", corr.eis_region)
                c3.metric("سازگاری", f"{corr.consistency_score:.0%}")

                for w in corr.warnings:
                    st.warning(w)
                for r in corr.recommendations:
                    st.success(r)
            except Exception as e:
                st.error(f"❌ {e}")
