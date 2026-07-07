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
    '<span class="fmt">.idf Autolab</span> <span class="fmt">.dta Gamry</span> '
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

# ── E_onset auto-select map ───────────────────────────────────