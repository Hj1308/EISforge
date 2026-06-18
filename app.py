"""
EISForge — Advanced EIS Analysis with Physics-Informed ML
Author: Hoda Jafari | May 2026
Run: streamlit run app.py
"""

import streamlit as st
from eisforge.analysis.ecsa_calculator import ECSACalculator
import pandas as pd
import numpy as np
import re
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="EISForge", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#fff;--surface:#f8f9fa;--border:#e2e8f0;--accent:#2563eb;--accent2:#7c3aed;--success:#16a34a;--warning:#d97706;--danger:#dc2626;--text:#1e293b;--muted:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.title{font-size:2.4rem;font-weight:700;color:var(--accent);text-align:center;margin:0;letter-spacing:-1px;}
.subtitle{color:var(--muted);font-size:.9rem;font-family:'JetBrains Mono',monospace;text-align:center;margin-top:.3rem;}
.section-title{font-size:.65rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:.6rem;font-family:'JetBrains Mono',monospace;font-weight:600;}
.stButton>button{background:var(--accent)!important;color:white!important;border:none!important;border-radius:6px!important;font-weight:600!important;padding:.4rem 1.2rem!important;}
.stButton>button:hover{background:#1d4ed8!important;}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem;}
div[data-testid="stMetric"] label{color:var(--muted)!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--text)!important;}
.fmt{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.7rem;font-family:'JetBrains Mono',monospace;background:#eff6ff;color:var(--accent);border:1px solid #bfdbfe;margin:.1rem;}
.stTabs [data-baseweb="tab"]{color:va