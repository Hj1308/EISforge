"""
EISForge — رابط کاربری اصلی Streamlit.

نویسنده: Hoda Jafari
تاریخ: May 2026

اجرا:
    streamlit run app.py
"""

import streamlit as st

# ── تنظیمات صفحه ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EISForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS سفارشی ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1e293b;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --success:   #34d399;
    --warning:   #fbbf24;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.eisforge-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.eisforge-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -2px;
    margin: 0;
}

.eisforge-subtitle {
    color: var(--muted);
    font-size: 1rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: var(--accent); }

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

.badge-success { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
.badge-warning { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid #fbbf24; }
.badge-danger  { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }
.badge-info    { background: rgba(56,189,248,0.15);  color: #38bdf8; border: 1px solid #38bdf8; }

.section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--muted);
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.stFileUploader {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

.author-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
    padding: 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── هدر اصلی ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eisforge-header">
    <h1 class="eisforge-title">⚡ EISForge</h1>
    <p class="eisforge-subtitle">
        Physics-Informed ML for Electrochemical Impedance Spectroscopy
    </p>
    <p class="eisforge-subtitle" style="color:#38bdf8; margin-top:0.3rem;">
        by Hoda Jafari · May 2026 · MIT License
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">⚙ تنظیمات</p>', unsafe_allow_html=True)

    electrolyte = st.selectbox(
        "محیط الکتروشیمیایی",
        ["اسیدی (H₂SO₄ / HClO₄)", "بازی (KOH / NaOH)"],
        index=0,
    )
    electrolyte_key = "acidic" if "اسیدی" in electrolyte else "alkaline"

    alcohol = st.selectbox(
        "نوع الکل",
        ["متانول", "اتانول", "اتیلن گلیکول", "گلیسرول"],
    )

    catalyst = st.text_input("کاتالیست", placeholder="مثال: Pt/C، PdAu/C")

    st.markdown("---")
    st.markdown('<p class="section-title">🔬 تنظیمات EIS</p>', unsafe_allow_html=True)

    onset_method = st.radio(
        "روش تشخیص E_onset",
        ["tangent", "threshold", "derivative"],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div class="author-badge">
        EISForge v0.1.0<br>
        github.com/Hj1308/EISforge-
    </div>
    """, unsafe_allow_html=True)


# ── تب‌های اصلی ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 تحلیل CV",
    "🔬 تحلیل EIS",
    "🤖 EIS-GPT",
    "🔗 همبستگی EIS-CV",
])


# ══════════════════════════════════════════════════════════════════════════════
# تب ۱ — تحلیل CV
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📈 آپلود داده CV</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        cv_file = st.file_uploader(
            "فایل CV (CSV)",
            type=["csv", "txt"],
            help="ستون اول: پتانسیل (V) | ستون دوم: جریان (mA)",
            key="cv_upload",
        )

        scan_rate = st.number_input("سرعت scan (mV/s)", value=50, min_value=1)
        eis_potential_input = st.number_input(
            "پتانسیل EIS (V) — اگر دارید",
            value=0.5, step=0.01,
            help="پتانسیلی که EIS در آن اندازه‌گیری شده",
        )

    with col2:
        if cv_file is not None:
            import pandas as pd
            import numpy as np

            try:
                df_cv = pd.read_csv(cv_file, comment="#", header=0)
                cols = df_cv.columns.tolist()
                potential = df_cv[cols[0]].to_numpy(dtype=float)
                current   = df_cv[cols[1]].to_numpy(dtype=float)

                st.success(f"✅ {len(potential)} نقطه بارگذاری شد")

                # تحلیل CV
                from eisforge.analysis.cv_analyzer import CVAnalyzer

                analyzer = CVAnalyzer(
                    onset_method=onset_method,
                    electrolyte=electrolyte_key,
                )
                result = analyzer.analyze(potential, current, scan_rate)

                # ذخیره در session state
                st.session_state["cv_result"]   = result
                st.session_state["cv_potential"] = potential
                st.session_state["cv_current"]   = current

            except Exception as e:
                st.error(f"❌ خطا در بارگذاری: {e}")

    # نمایش نتایج
    if "cv_result" in st.session_state:
        result = st.session_state["cv_result"]
        st.markdown("---")
        st.markdown('<p class="section-title">📊 نتایج تحلیل CV</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("E_onset", f"{result.e_onset:.4f} V")
        with c2:
            st.metric("I_f (پیک رفت)", f"{result.i_forward_peak:.3f} mA")
        with c3:
            st.metric("I_b (پیک برگشت)", f"{result.i_backward_peak:.3f} mA")
        with c4:
            ratio_color = "badge-success" if result.if_ib_ratio > 1 else "badge-danger"
            st.metric("I_f / I_b", f"{result.if_ib_ratio:.3f}")

        # تفسیر
        st.info(f"💡 **تفسیر:** {result.interpretation}")

        # نمودار CV
        import plotly.graph_objects as go

        pot = st.session_state["cv_potential"]
        cur = st.session_state["cv_current"]

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=pot, y=cur,
            mode="lines",
            name="CV",
            line=dict(color="#38bdf8", width=2),
        ))
        fig_cv.add_vline(
            x=result.e_onset,
            line_dash="dash",
            line_color="#fbbf24",
            annotation_text=f"E_onset = {result.e_onset:.3f} V",
            annotation_font_color="#fbbf24",
        )
        fig_cv.add_trace(go.Scatter(
            x=[result.e_forward_peak],
            y=[result.i_forward_peak],
            mode="markers",
            name=f"I_f = {result.i_forward_peak:.3f} mA",
            marker=dict(color="#34d399", size=12, symbol="star"),
        ))
        fig_cv.add_trace(go.Scatter(
            x=[result.e_backward_peak],
            y=[result.i_backward_peak],
            mode="markers",
            name=f"I_b = {result.i_backward_peak:.3f} mA",
            marker=dict(color="#f87171", size=12, symbol="star"),
        ))
        fig_cv.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            title="Cyclic Voltammogram",
            xaxis_title="Potential (V)",
            yaxis_title="Current (mA)",
            font=dict(family="JetBrains Mono"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# تب ۲ — تحلیل EIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔬 آپلود داده EIS</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        eis_file = st.file_uploader(
            "فایل EIS (CSV / DTA)",
            type=["csv", "txt", "dta"],
            key="eis_upload",
        )

        circuit_str = st.text_input(
            "مدار معادل",
            value="R0-p(R1,CPE1)-W1",
            help="مثال: R0-p(R1,CPE1)-W1",
        )

        p0_str = st.text_input(
            "Initial Guess (با کاما جدا کنید)",
            value="10, 200, 1e-5, 0.85, 30",
        )

    with col2:
        if eis_file is not None:
            import pandas as pd
            import numpy as np

            try:
                df_eis = pd.read_csv(eis_file, comment="#")
                cols = df_eis.columns.tolist()
                freq   = df_eis[cols[0]].to_numpy(dtype=float)
                z_real = df_eis[cols[1]].to_numpy(dtype=float)
                z_imag = df_eis[cols[2]].to_numpy(dtype=float)

                st.success(f"✅ {len(freq)} نقطه بارگذاری شد")
                st.session_state["eis_freq"]   = freq
                st.session_state["eis_z_real"] = z_real
                st.session_state["eis_z_imag"] = z_imag

            except Exception as e:
                st.error(f"❌ خطا: {e}")

    # فیت و نمودار
    if "eis_freq" in st.session_state:
        if st.button("🚀 اجرای CNLS Fit"):
            with st.spinner("در حال فیت کردن..."):
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
                    fit_result = fitter.fit(dataset)
                    st.session_state["fit_result"] = fit_result
                    st.success("✅ فیت موفق!")
                except Exception as e:
                    st.error(f"❌ خطای فیت: {e}")

        if "fit_result" in st.session_state:
            fit = st.session_state["fit_result"]
            st.markdown("---")
            st.markdown(f"**χ² کاهش‌یافته = {fit.chi_squared:.6f}**")

            # جدول پارامترها
            import pandas as pd
            param_data = []
            for name, val in fit.parameters.items():
                err = fit.parameter_errors.get(name, float("nan"))
                param_data.append({
                    "پارامتر": name,
                    "مقدار": f"{val:.4e}",
                    "خطا (±)": f"{err:.2e}",
                })
            st.dataframe(pd.DataFrame(param_data), use_container_width=True)

            # نمودار Nyquist
            import plotly.graph_objects as go
            import numpy as np

            freq   = st.session_state["eis_freq"]
            z_real = st.session_state["eis_z_real"]
            z_imag = st.session_state["eis_z_imag"]

            fig_ny = go.Figure()
            fig_ny.add_trace(go.Scatter(
                x=z_real, y=z_imag,
                mode="markers",
                name="داده اندازه‌گیری",
                marker=dict(color="#38bdf8", size=8),
            ))
            if fit.z_fit is not None:
                fig_ny.add_trace(go.Scatter(
                    x=fit.z_fit.real,
                    y=-fit.z_fit.imag,
                    mode="lines",
                    name="مدل فیت‌شده",
                    line=dict(color="#f87171", width=2, dash="dash"),
                ))
            fig_ny.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                title="نمودار Nyquist",
                xaxis_title="Z' (Ω)",
                yaxis_title="-Z'' (Ω)",
                font=dict(family="JetBrains Mono"),
            )
            st.plotly_chart(fig_ny, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# تب ۳ — EIS-GPT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">🤖 EIS-GPT — پیش‌بینی هوشمند مدار</p>',
                unsafe_allow_html=True)

    st.info(
        "💡 **EIS-GPT** از طیف EIS شما مستقیماً مدار معادل را پیش‌بینی می‌کند "
        "— بدون نیاز به حدس اولیه!"
    )

    if "eis_freq" in st.session_state:
        if st.button("🧠 پیش‌بینی با EIS-GPT"):
            with st.spinner("EIS-GPT در حال تحلیل..."):
                try:
                    import torch
                    from eisforge.ml.eis_gpt.transformer import EISForgeModel
                    import numpy as np

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
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "مدار پیش‌بینی‌شده",
                    res["predicted_circuit"],
                )
                conf = res["confidence"]
                badge = "badge-success" if conf > 0.7 else "badge-warning"
                st.markdown(
                    f'<span class="{badge} status-badge">'
                    f'اطمینان: {conf*100:.1f}%</span>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown("**سه کاندید برتر:**")
                for c in res["top3"]:
                    prob = c["probability"]
                    st.progress(prob, text=f"{c['circuit']} — {prob*100:.1f}%")

    else:
        st.warning("⚠️ ابتدا داده EIS را در تب 'تحلیل EIS' آپلود کنید.")


# ══════════════════════════════════════════════════════════════════════════════
# تب ۴ — همبستگی EIS-CV
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🔗 همبستگی EIS و CV</p>',
                unsafe_allow_html=True)

    has_cv  = "cv_result"  in st.session_state
    has_eis = "fit_result" in st.session_state

    if not has_cv:
        st.warning("⚠️ ابتدا CV را در تب 'تحلیل CV' آپلود و تحلیل کنید.")
    if not has_eis:
        st.warning("⚠️ ابتدا EIS را در تب 'تحلیل EIS' فیت کنید.")

    if has_cv and has_eis:
        eis_pot = st.number_input(
            "پتانسیل اندازه‌گیری EIS (V)",
            value=0.5, step=0.01,
        )

        if st.button("🔗 تحلیل همبستگی"):
            try:
                from eisforge.analysis.eis_cv_correlator import EISCVCorrelator

                correlator = EISCVCorrelator(electrolyte=electrolyte_key)
                corr = correlator.correlate(
                    cv_result=st.session_state["cv_result"],
                    eis_fit_result=st.session_state["fit_result"],
                    eis_potential=eis_pot,
                )

                # نتایج
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("E_onset (CV)", f"{corr.e_onset:.4f} V")
                with c2:
                    st.metric("پتانسیل EIS", f"{corr.eis_potential:.4f} V")
                with c3:
                    st.metric("امتیاز سازگاری", f"{corr.consistency_score:.0%}")

                # ناحیه
                region_map = {
                    "pre-onset":  ("❌ قبل از E_onset", "badge-danger"),
                    "onset":      ("⚡ روی E_onset", "badge-warning"),
                    "post-onset": ("✅ بعد از E_onset", "badge-success"),
                }
                label, badge = region_map.get(corr.eis_region, ("نامشخص", "badge-info"))
                st.markdown(
                    f'<span class="{badge} status-badge">{label}</span>',
                    unsafe_allow_html=True,
                )

                # هشدارها
                if corr.warnings:
                    st.markdown("---")
                    st.markdown("**⚠️ هشدارها:**")
                    for w in corr.warnings:
                        st.warning(w)

                # پیشنهادها
                if corr.recommendations:
                    st.markdown("**💡 پیشنهادها:**")
                    for r in corr.recommendations:
                        st.success(r)

            except Exception as e:
                st.error(f"❌ خطا: {e}")
