import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="MediCost AI Pro", layout="wide", page_icon="🛡️")

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #050d1a;
    --bg-card: rgba(10, 25, 47, 0.85);
    --bg-glass: rgba(16, 35, 65, 0.6);
    --accent-cyan: #00e5ff;
    --accent-teal: #00bfa5;
    --accent-blue: #1565c0;
    --accent-glow: rgba(0, 229, 255, 0.15);
    --text-primary: #e8f4fd;
    --text-muted: #7ba3c4;
    --border: rgba(0, 229, 255, 0.15);
    --border-strong: rgba(0, 229, 255, 0.35);
    --success: #00e676;
    --warning: #ffab40;
    --danger: #ff5252;
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --radius: 16px;
    --radius-sm: 10px;
}

/* ===== BASE RESET ===== */
html, body, [class*="css"], .stApp {
    background: var(--bg-primary) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* ===== ANIMATED BACKGROUND ===== */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(0,229,255,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,191,165,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 10%, rgba(21,101,192,0.08) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 12s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { transform: scale(1) rotate(0deg); }
    100% { transform: scale(1.05) rotate(2deg); }
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061020 0%, #0a1a35 100%) !important;
    border-right: 1px solid var(--border-strong) !important;
    box-shadow: 4px 0 40px rgba(0,229,255,0.06) !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent-cyan) !important;
}

/* ===== SIDEBAR HEADER BRANDING ===== */
.sidebar-brand {
    background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(0,191,165,0.08));
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    padding: 20px 16px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.sidebar-brand::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(0,229,255,0.05) 60deg, transparent 120deg);
    animation: spinGlow 8s linear infinite;
}
@keyframes spinGlow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.sidebar-brand h2 {
    font-family: var(--font-display) !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}
.sidebar-brand p {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    margin: 4px 0 0 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}

/* ===== PAGE TITLE ===== */
.page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}
.page-title {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1.1 !important;
}
.page-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ===== METRIC CARDS ===== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--border-strong);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,229,255,0.1);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-cyan);
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
}
.metric-icon {
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 2.5rem;
    opacity: 0.12;
}

/* ===== SECTION CARDS ===== */
.section-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.section-card-title {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ===== DATAFRAME OVERRIDES ===== */
[data-testid="stDataFrame"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,191,165,0.1)) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(0,191,165,0.18)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 24px rgba(0,229,255,0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ===== SUCCESS / ERROR / WARNING ===== */
.stSuccess {
    background: rgba(0, 230, 118, 0.08) !important;
    border: 1px solid rgba(0, 230, 118, 0.3) !important;
    color: var(--success) !important;
    border-radius: var(--radius-sm) !important;
}
.stError {
    background: rgba(255, 82, 82, 0.08) !important;
    border: 1px solid rgba(255, 82, 82, 0.3) !important;
    border-radius: var(--radius-sm) !important;
}
.stWarning {
    background: rgba(255, 171, 64, 0.08) !important;
    border: 1px solid rgba(255, 171, 64, 0.3) !important;
    border-radius: var(--radius-sm) !important;
}
.stInfo {
    background: rgba(0, 229, 255, 0.07) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ===== SELECTBOX / INPUTS ===== */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
}
[data-testid="stTextInput"] input {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(0,229,255,0.03) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(0,229,255,0.06) !important;
    border-color: var(--accent-cyan) !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.08) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 24px !important;
    backdrop-filter: blur(20px) !important;
}

/* ===== RADIO ===== */
.stRadio > div {
    gap: 8px !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}

/* ===== SLIDER ===== */
.stSlider [data-baseweb="slider"] {
    margin-top: 8px !important;
}
.stSlider [data-testid="stThumbValue"] {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
}

/* ===== PLOTLY CHARTS ===== */
.js-plotly-plot {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

/* ===== BEST MODEL BANNER ===== */
.best-model-banner {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,191,165,0.08));
    border: 1px solid rgba(0,230,118,0.35);
    border-radius: var(--radius);
    padding: 20px 28px;
    text-align: center;
    margin-top: 16px;
}
.best-model-banner .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--success);
    margin-bottom: 6px;
}
.best-model-banner .name {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
}

/* ===== SCORE BADGE ===== */
.score-pill {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid var(--border-strong);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: var(--accent-cyan);
    font-weight: 500;
    margin-left: 10px;
}

/* ===== UPLOAD PROMPT ===== */
.upload-prompt {
    text-align: center;
    padding: 80px 40px;
    max-width: 480px;
    margin: 60px auto;
}
.upload-icon {
    font-size: 5rem;
    margin-bottom: 24px;
    display: block;
    filter: drop-shadow(0 0 24px rgba(0,229,255,0.3));
}
.upload-prompt h2 {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin-bottom: 12px !important;
}
.upload-prompt p {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* ===== STEP INDICATOR ===== */
.step-indicator {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.step-badge {
    background: rgba(0,229,255,0.07);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.step-badge.active {
    background: rgba(0,229,255,0.14);
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }

/* ===== MAIN CONTENT PADDING ===== */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* ===== SUBHEADER OVERRIDES ===== */
h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.02em !important;
}
</style>
""", unsafe_allow_html=True)


# ---- Plotly dark theme helper ----
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,25,47,0.5)",
    font=dict(family="DM Sans, sans-serif", color="#7ba3c4"),
    title_font=dict(family="Syne, sans-serif", color="#e8f4fd"),
    colorway=["#00e5ff", "#00bfa5", "#1565c0", "#7c4dff", "#ff6d00"],
    xaxis=dict(gridcolor="rgba(0,229,255,0.07)", zerolinecolor="rgba(0,229,255,0.1)"),
    yaxis=dict(gridcolor="rgba(0,229,255,0.07)", zerolinecolor="rgba(0,229,255,0.1)"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🛡️ MediCost AI Pro</h2>
        <p>Medical Cost Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    menu = st.radio(
        "Pipeline",
        ["📥 Data", "🛠️ Preprocessing", "📈 Visuals", "🤖 Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Upload Dataset**")
    file = st.file_uploader("Upload CSV", label_visibility="collapsed")

    if file:
        st.markdown("""
        <div style="background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.25);
        border-radius:10px;padding:10px 14px;margin-top:8px;">
        <span style="color:#00e676;font-size:0.78rem;font-weight:500;">✓ Dataset loaded</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#4a6a8a;text-align:center;line-height:1.6;">
    Pipeline Steps<br>
    <span style="color:#00e5ff">Data</span> → <span style="color:#00bfa5">Preprocess</span> → 
    <span style="color:#7c4dff">Visualize</span> → <span style="color:#ff6d00">Model</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
if file:
    if 'main_df' not in st.session_state:
        st.session_state['main_df'] = pd.read_csv(file)

    df = st.session_state['main_df']

    # ================= DATA =================
    if menu == "📥 Data":
        st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Dataset Overview</div>
                <div class="page-subtitle">Inspect structure · quality · completeness</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards
        num_missing = int(df.isna().sum().sum())
        num_cols_n = len(df.select_dtypes(include=np.number).columns)
        num_cols_c = len(df.select_dtypes(exclude=np.number).columns)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-sub">Observations in dataset</div>
                <div class="metric-icon">📋</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-sub">{num_cols_n} numeric · {num_cols_c} categorical</div>
                <div class="metric-icon">📊</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value" style="color:{'#ff5252' if num_missing>0 else '#00e676'}">{num_missing}</div>
                <div class="metric-sub">{'Requires attention' if num_missing>0 else 'Dataset is clean'}</div>
                <div class="metric-icon">🔍</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-card-title">🗃️ Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card"><div class="section-card-title">📐 Data Types</div>', unsafe_allow_html=True)
            dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Non-Null": df.count().values})
            st.dataframe(dtype_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card"><div class="section-card-title">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ================= PREPROCESS =================
    elif menu == "🛠️ Preprocessing":
        st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Data Preprocessing</div>
                <div class="page-subtitle">Clean · Transform · Engineer Features</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🧽  Missing Values", "📉  Outliers", "✨  Feature Engineering"])

        # -------- Missing Values --------
        with tab1:
            col1, col2 = st.columns([1.8, 1])
            with col1:
                missing = df.isna().sum()
                missing = missing[missing > 0]
                if not missing.empty:
                    fig = go.Figure(go.Bar(
                        x=missing.index.tolist(),
                        y=missing.values.tolist(),
                        marker_color="#00e5ff",
                        marker_line_color="rgba(0,229,255,0.3)",
                        marker_line_width=1,
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, title="Missing Values per Column")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.25);
                    border-radius:12px;padding:32px;text-align:center;">
                    <div style="font-size:2.5rem;">✅</div>
                    <div style="color:#00e676;font-weight:600;margin-top:10px;">No Missing Values Found</div>
                    <div style="color:#4a6a8a;font-size:0.85rem;margin-top:6px;">Your dataset is complete.</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Strategy**")
                st.markdown("""
                <div style="color:#7ba3c4;font-size:0.83rem;line-height:1.7;margin-bottom:16px;">
                • Numeric → Median fill<br>
                • Categorical → Mode fill<br>
                • All-null columns → 0 fill
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔧 Fix Missing Values", use_container_width=True):
                    new_df = df.copy()
                    for col in new_df.columns:
                        if pd.api.types.is_numeric_dtype(new_df[col]):
                            if new_df[col].isna().all():
                                new_df[col] = new_df[col].fillna(0)
                            else:
                                new_df[col] = new_df[col].fillna(new_df[col].median())
                        else:
                            new_df[col] = new_df[col].fillna(new_df[col].mode()[0] if not new_df[col].mode().empty else "Unknown")
                    st.session_state['main_df'] = new_df
                    st.success("✓ Missing values handled!")
                    st.rerun()

        # -------- Outliers --------
        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if num_cols:
                col1, col2 = st.columns([1.8, 1])
                with col1:
                    col = st.selectbox("Select Column to Inspect", num_cols)
                    fig = px.box(df, y=col)
                    fig.update_traces(marker_color="#00e5ff", line_color="#00bfa5", fillcolor="rgba(0,229,255,0.08)")
                    fig.update_layout(**PLOTLY_LAYOUT, title=f"Outlier Distribution — {col}")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

                    st.markdown(f"""
                    <div class="section-card" style="padding:16px;">
                        <div class="metric-label">IQR Bounds</div>
                        <div style="color:#00e5ff;font-family:'Syne',sans-serif;font-size:0.95rem;margin:8px 0;">
                            [{lower:.2f}, {upper:.2f}]
                        </div>
                        <div class="metric-label">Outliers Detected</div>
                        <div style="color:{'#ff5252' if outlier_count>0 else '#00e676'};font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;">
                            {outlier_count}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("✂️ Apply Capping", use_container_width=True):
                        new_df = df.copy()
                        new_df[col] = np.clip(new_df[col], lower, upper)
                        st.session_state['main_df'] = new_df
                        st.success("✓ Outliers capped!")
                        st.rerun()

        # -------- Feature Engineering --------
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Step 1: Encode**")
                if st.button("🔠 Encode Data", use_container_width=True):
                    df_ml = df.copy()
                    if 'smoker' in df_ml.columns and 'age' in df_ml.columns:
                        le = LabelEncoder()
                        df_ml['smoker_num'] = le.fit_transform(df_ml['smoker'].astype(str))
                        df_ml['risk_score'] = df_ml['age'] * df_ml['smoker_num']
                    df_ml = pd.get_dummies(df_ml, drop_first=True)
                    df_ml = df_ml.astype(int)
                    st.session_state['processed_df'] = df_ml
                    st.success("✓ Encoding complete")

            if 'processed_df' in st.session_state:
                data = st.session_state['processed_df']

                with col2:
                    target = st.selectbox("Step 2: Select Target Column", data.columns)

                if target not in data.columns:
                    st.error("Invalid target selected")
                    st.stop()

                X = data.drop(columns=[target])
                y = data[target]

                st.markdown("**Step 3: Feature Selection Method**")
                method = st.radio("Feature Selection", ["Correlation", "SelectKBest", "Feature Importance"], horizontal=True)

                if method == "Correlation":
                    corr = data.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
                    fig = go.Figure(go.Bar(
                        x=corr.values[1:16].tolist(), y=corr.index[1:16].tolist(),
                        orientation='h', marker_color="#00e5ff",
                        marker_line_color="rgba(0,229,255,0.3)", marker_line_width=1
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, title=f"Feature Correlation with {target}", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    k = st.slider("Top Features", 1, len(X.columns), 5)
                    selected = corr.index[1:k+1]

                elif method == "SelectKBest":
                    k = st.slider("K Features", 1, len(X.columns), 5)
                    selector = SelectKBest(f_regression, k=k)
                    selector.fit(X, y)
                    selected = X.columns[selector.get_support()]

                else:
                    with st.spinner("Training Random Forest..."):
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(X, y)
                    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    fig = go.Figure(go.Bar(
                        x=imp.values[:15].tolist(), y=imp.index[:15].tolist(),
                        orientation='h',
                        marker=dict(
                            color=imp.values[:15].tolist(),
                            colorscale=[[0,"#1565c0"],[0.5,"#00bfa5"],[1,"#00e5ff"]],
                            showscale=False
                        )
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, title="Feature Importance Scores", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    k = st.slider("Top Features", 1, len(X.columns), 5)
                    selected = imp.head(k).index

                st.success(f"✓ Selected {len(list(selected))} features: {', '.join(list(selected))}")
                final_df = data[list(selected) + [target]]
                st.session_state['final_df'] = final_df
                st.dataframe(final_df.head(), use_container_width=True)

    # ================= VISUALS =================
    elif menu == "📈 Visuals":
        st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Data Visualization</div>
                <div class="page-subtitle">Distribution · Outliers · Correlation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) == 0:
            st.warning("No numeric columns found in the dataset.")
        else:
            col_left, col_right = st.columns([1, 3])
            with col_left:
                feature = st.selectbox("Select Feature", num_cols)
                st.markdown(f"""
                <div class="section-card" style="padding:16px;margin-top:12px;">
                    <div class="metric-label">Mean</div>
                    <div style="color:#00e5ff;font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:10px;">
                        {df[feature].mean():.2f}
                    </div>
                    <div class="metric-label">Median</div>
                    <div style="color:#00bfa5;font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:10px;">
                        {df[feature].median():.2f}
                    </div>
                    <div class="metric-label">Std Dev</div>
                    <div style="color:#7c4dff;font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">
                        {df[feature].std():.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                fig1 = px.histogram(df, x=feature, nbins=30, marginal="rug")
                fig1.update_traces(marker_color="#00e5ff", marker_line_color="#00bfa5", marker_line_width=0.5, opacity=0.8)
                fig1.update_layout(**PLOTLY_LAYOUT, title=f"Distribution — {feature}", bargap=0.05)
                st.plotly_chart(fig1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.box(df, y=feature)
                fig2.update_traces(marker_color="#00e5ff", line_color="#00bfa5", fillcolor="rgba(0,229,255,0.08)")
                fig2.update_layout(**PLOTLY_LAYOUT, title=f"Boxplot — {feature}")
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                corr = df[num_cols].corr()
                fig3 = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale=[[0,"#050d1a"],[0.5,"#1565c0"],[1,"#00e5ff"]],
                    zmin=-1, zmax=1
                )
                fig3.update_layout(**PLOTLY_LAYOUT, title="Correlation Heatmap")
                fig3.update_traces(textfont=dict(size=10, color="#e8f4fd"))
                st.plotly_chart(fig3, use_container_width=True)

    # ================= MODEL =================
    elif menu == "🤖 Model":
        st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Model Training</div>
                <div class="page-subtitle">Train · Evaluate · Compare Algorithms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if 'processed_df' not in st.session_state:
            st.markdown("""
            <div style="background:rgba(255,82,82,0.07);border:1px solid rgba(255,82,82,0.25);
            border-radius:12px;padding:20px 24px;text-align:center;">
            <div style="color:#ff5252;font-weight:600;margin-bottom:6px;">⚠️ Preprocessing Required</div>
            <div style="color:#7ba3c4;font-size:0.85rem;">
            Go to <strong>Preprocessing → Feature Engineering</strong> and run Encode Data first.
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            data = st.session_state.get('final_df', st.session_state['processed_df'])

            col1, col2 = st.columns([1, 2])
            with col1:
                target = st.selectbox("Target Column", data.columns)
                st.markdown(f"""
                <div class="section-card" style="padding:16px;margin-top:12px;">
                    <div class="metric-label">Training Split</div>
                    <div style="color:#00e5ff;font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;margin-bottom:8px;">
                        80% Train · 20% Test
                    </div>
                    <div class="metric-label">Features</div>
                    <div style="color:#00bfa5;font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;margin-bottom:8px;">
                        {len(data.columns)-1} columns
                    </div>
                    <div class="metric-label">Samples</div>
                    <div style="color:#7c4dff;font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;">
                        {len(data):,} rows
                    </div>
                </div>
                """, unsafe_allow_html=True)
                train_btn = st.button("🚀 Train All Models", use_container_width=True)

            with col2:
                if train_btn:
                    if target not in data.columns:
                        st.error("Invalid target selected")
                        st.stop()

                    X = data.drop(columns=[target])
                    y = data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    models = {
                        "Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "KNN": KNeighborsRegressor()
                    }

                    results = []
                    progress = st.progress(0, "Training models...")

                    for i, (name, model) in enumerate(models.items()):
                        progress.progress((i+1)/len(models), f"Training {name}...")
                        xt, xv = (X_train_s, X_test_s) if name == "KNN" else (X_train, X_test)
                        model.fit(xt, y_train)
                        pred = model.predict(xv)
                        results.append({
                            "Model": name,
                            "R2 Score": round(r2_score(y_test, pred), 4),
                            "MAE": round(mean_absolute_error(y_test, pred), 2)
                        })

                    progress.empty()
                    res_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

                    # Results chart
                    colors = ["#00e5ff", "#00bfa5", "#1565c0"]
                    fig = go.Figure()
                    for i, row in res_df.iterrows():
                        fig.add_trace(go.Bar(
                            x=[row["Model"]], y=[row["R2 Score"]],
                            name=row["Model"],
                            marker_color=colors[i % len(colors)],
                            text=[f"{row['R2 Score']:.4f}"],
                            textposition='outside',
                            textfont=dict(color="#e8f4fd", size=12, family="Syne")
                        ))
                    fig.update_layout(**PLOTLY_LAYOUT, title="R² Score Comparison", showlegend=False,
                                      yaxis_range=[0, min(1.15, res_df["R2 Score"].max()*1.2)])
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(res_df, use_container_width=True)

                    best = res_df.iloc[0]
                    st.markdown(f"""
                    <div class="best-model-banner">
                        <div class="label">🏆 Best Performing Model</div>
                        <div class="name">{best['Model']}</div>
                        <div style="color:#7ba3c4;font-size:0.85rem;margin-top:8px;">
                            R² Score: <strong style="color:#00e5ff;">{best['R2 Score']}</strong>
                            &nbsp;·&nbsp;
                            MAE: <strong style="color:#00bfa5;">{best['MAE']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-prompt">
        <span class="upload-icon">🛡️</span>
        <h2>MediCost AI Pro</h2>
        <p>Upload a CSV dataset using the sidebar to begin the ML pipeline —
        from raw data to trained predictive models.</p>
    </div>
    """, unsafe_allow_html=True)
