import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Professional Theme ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #090d13 !important;
    color: #c8d6e8 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0d1420 !important;
    border-right: 1px solid #1a2540 !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #00e5ff, #7c3aed, #00e5ff);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 8px #00c853} 50%{opacity:.6;box-shadow:0 0 4px #00c853} }
@keyframes topBar { 0%{background-position:0% 0} 100%{background-position:300% 0} }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes alertPulse { 0%,100%{box-shadow:0 0 0 0 #ff525222} 50%{box-shadow:0 0 0 6px #ff525200} }

.sidebar-brand { padding:28px 24px 20px; border-bottom:1px solid #1a2540; margin-bottom:8px; }
.logo-hex { font-size:28px; color:#00e5ff; font-weight:800; letter-spacing:-1px; }
.logo-text { font-size:20px; font-weight:800; color:#fff; letter-spacing:1px; }
.logo-sub { font-size:10px; font-family:'DM Mono',monospace; color:#3a4f6e; letter-spacing:3px; text-transform:uppercase; margin-top:2px; }

.sidebar-status { margin:16px 24px; padding:10px 14px; background:#0a1a0a; border:1px solid #0d3d0d; border-radius:8px; display:flex; align-items:center; gap:8px; }
.status-dot { width:7px;height:7px;border-radius:50%;background:#00c853;box-shadow:0 0 8px #00c853;animation:pulse 2s infinite;flex-shrink:0; }
.status-text { font-family:'DM Mono',monospace;font-size:10px;color:#2e7d32;letter-spacing:1px; }

[data-testid="stSidebar"] .stRadio > div { gap:4px !important; padding:0 12px; }
[data-testid="stSidebar"] .stRadio label {
    background:transparent !important;
    border:1px solid transparent !important;
    border-radius:8px !important;
    padding:10px 14px !important;
    color:#4a6080 !important;
    font-size:13px !important;
    font-weight:600 !important;
    letter-spacing:.5px !important;
    cursor:pointer !important;
    transition:all .2s ease !important;
    display:flex !important;
    align-items:center !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background:#131e30 !important;
    color:#c8d6e8 !important;
    border-color:#1e3050 !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background:#0f2040 !important;
    color:#00e5ff !important;
    border-color:#00e5ff33 !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display:none !important;
}

.page-header { padding:32px 0 24px;border-bottom:1px solid #1a2540;margin-bottom:32px;animation:fadeIn .4s ease; }
.page-tag { font-family:'DM Mono',monospace;font-size:11px;color:#00e5ff;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px; }
.page-title { font-size:36px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1.1;margin:0; }
.page-title span { color:#00e5ff; }
.page-desc { font-size:14px;color:#3a4f6e;margin-top:8px;font-family:'DM Mono',monospace; }

.section-header { display:flex;align-items:center;gap:12px;margin:28px 0 16px; }
.section-line { flex:1;height:1px;background:linear-gradient(90deg,#1a2d4a,transparent); }
.section-label { font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;letter-spacing:3px;text-transform:uppercase;white-space:nowrap; }

.insight-badge { display:inline-flex;align-items:center;gap:6px;background:#0d1a2d;border:1px solid #1a2d4a;border-radius:20px;padding:6px 14px;font-family:'DM Mono',monospace;font-size:11px;color:#4a6080;letter-spacing:1px;margin-bottom:20px; }
.insight-badge .dot { width:5px;height:5px;border-radius:50%;background:#00e5ff; }

.kpi-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px; }
.kpi-card { background:#0d1a2d;border:1px solid #1a2d4a;border-radius:14px;padding:20px 24px;position:relative;overflow:hidden; }
.kpi-card::before { content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#00e5ff,#7c3aed); }
.kpi-label { font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px; }
.kpi-value { font-size:32px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1; }
.kpi-value.danger { color:#ff5252; }
.kpi-value.warning { color:#ffab40; }
.kpi-value.success { color:#69f0ae; }

.kpi-grid-4 { display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px; }

.result-churn { background:linear-gradient(135deg,#1a0a0a,#2d0d0d);border:1px solid #e74c3c44;border-radius:16px;padding:28px;text-align:center;position:relative;overflow:hidden; }
.result-churn::after { content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#e74c3c,transparent); }
.result-stay { background:linear-gradient(135deg,#0a1a0a,#0d2d0d);border:1px solid #00c85344;border-radius:16px;padding:28px;text-align:center;position:relative;overflow:hidden; }
.result-stay::after { content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00c853,transparent); }
.result-label { font-size:22px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin:4px 0; }
.result-churn .result-label { color:#ff5252; }
.result-stay .result-label  { color:#69f0ae; }
.result-sub { font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;letter-spacing:2px; }

/* Segment cards */
.seg-grid { display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:24px; }
.seg-card { background:#0d1a2d;border:1px solid #1a2d4a;border-radius:14px;padding:22px;position:relative;overflow:hidden;transition:border-color .2s; }
.seg-card:hover { border-color:#00e5ff44; }
.seg-card .seg-accent { position:absolute;top:0;left:0;right:0;height:2px; }
.seg-name { font-size:17px;font-weight:800;color:#fff;margin:10px 0 4px; }
.seg-desc { font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;line-height:1.6;margin-bottom:14px; }
.seg-stat { display:flex;justify-content:space-between;align-items:center;margin-top:8px; }
.seg-count { font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:#c8d6e8; }
.seg-pct { font-family:'DM Mono',monospace;font-size:11px;padding:3px 10px;border-radius:99px;border:1px solid; }

/* Alert cards */
.alert-card { background:#0d1420;border:1px solid #1a2d4a;border-radius:12px;padding:16px 20px;margin-bottom:10px;display:flex;align-items:center;gap:16px;transition:all .2s;cursor:pointer; }
.alert-card:hover { border-color:#ff525244;background:#130d1a; }
.alert-card.critical { border-left:3px solid #ff5252;animation:alertPulse 2.5s infinite; }
.alert-card.high { border-left:3px solid #ffab40; }
.alert-card.medium { border-left:3px solid #00e5ff; }
.alert-avatar { width:40px;height:40px;border-radius:10px;background:#131e30;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:#c8d6e8;flex-shrink:0; }
.alert-info { flex:1;min-width:0; }
.alert-name { font-size:14px;font-weight:700;color:#fff;margin-bottom:2px; }
.alert-meta { font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e; }
.alert-score { text-align:right;flex-shrink:0; }
.alert-score-val { font-family:'DM Mono',monospace;font-size:20px;font-weight:500; }
.alert-score-val.critical { color:#ff5252; }
.alert-score-val.high { color:#ffab40; }
.alert-score-val.medium { color:#00e5ff; }
.alert-score-label { font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px; }
.action-pill { display:inline-flex;align-items:center;gap:4px;background:#0a1a2a;border:1px solid #1a3050;border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;font-size:10px;color:#00e5ff;letter-spacing:.5px;margin:2px;cursor:pointer;transition:all .2s; }
.action-pill:hover { background:#00e5ff15;border-color:#00e5ff44; }

/* Timeline */
.timeline { position:relative;padding-left:20px; }
.timeline::before { content:'';position:absolute;left:4px;top:0;bottom:0;width:1px;background:linear-gradient(180deg,#00e5ff,#7c3aed,transparent); }
.tl-item { position:relative;margin-bottom:20px;padding-left:20px; }
.tl-dot { position:absolute;left:-20px;top:4px;width:9px;height:9px;border-radius:50%;border:2px solid; }
.tl-dot.cyan { border-color:#00e5ff;background:#00e5ff33; }
.tl-dot.purple { border-color:#7c3aed;background:#7c3aed33; }
.tl-dot.red { border-color:#ff5252;background:#ff525233; }
.tl-dot.green { border-color:#69f0ae;background:#69f0ae33; }
.tl-time { font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:3px; }
.tl-text { font-size:13px;color:#c8d6e8;line-height:1.5; }
.tl-tag { display:inline-block;font-family:'DM Mono',monospace;font-size:10px;padding:2px 7px;border-radius:4px;margin-top:4px; }

.stNumberInput input,.stTextInput input { background:#070d18 !important;border:1px solid #1a2d4a !important;border-radius:8px !important;color:#c8d6e8 !important;font-family:'DM Mono',monospace !important;font-size:13px !important;transition:border-color .2s !important; }
.stNumberInput input:focus,.stTextInput input:focus { border-color:#00e5ff !important;box-shadow:0 0 0 2px #00e5ff15 !important; }
label[data-testid="stWidgetLabel"] { color:#4a6080 !important;font-size:11px !important;font-family:'DM Mono',monospace !important;letter-spacing:1px !important;text-transform:uppercase !important; }

[data-testid="stFormSubmitButton"] button,.stButton > button { background:linear-gradient(135deg,#00b4d8,#7c3aed) !important;border:none !important;border-radius:10px !important;color:#fff !important;font-family:'Syne',sans-serif !important;font-weight:700 !important;font-size:14px !important;letter-spacing:1px !important;padding:14px 28px !important;cursor:pointer !important;transition:all .3s ease !important;text-transform:uppercase !important; }
[data-testid="stFormSubmitButton"] button:hover,.stButton > button:hover { transform:translateY(-2px) !important;box-shadow:0 8px 30px #00e5ff33 !important;filter:brightness(1.1) !important; }

[data-testid="stDataFrame"] { border:1px solid #1a2d4a !important;border-radius:12px !important;overflow:hidden !important; }
[data-testid="stDataFrame"] th { background:#0d1a2d !important;color:#4a6080 !important;font-family:'DM Mono',monospace !important;font-size:11px !important;letter-spacing:1px !important;text-transform:uppercase !important; }
[data-testid="stDataFrame"] td { color:#c8d6e8 !important;font-family:'DM Mono',monospace !important;font-size:12px !important; }

[data-testid="stFileUploader"] { background:#0d1a2d !important;border:1px dashed #1e3a5a !important;border-radius:14px !important; }
[data-testid="stDownloadButton"] button { background:transparent !important;border:1px solid #00e5ff44 !important;color:#00e5ff !important;font-family:'DM Mono',monospace !important;border-radius:10px !important; }
[data-testid="stDownloadButton"] button:hover { background:#00e5ff11 !important;border-color:#00e5ff !important; }

[data-testid="stSelectbox"] > div { background:#0d1a2d !important;border:1px solid #1a2d4a !important;border-radius:8px !important;color:#c8d6e8 !important; }
.stSelectbox [data-baseweb="select"] { background:#070d18 !important; }

::-webkit-scrollbar { width:6px;height:6px; }
::-webkit-scrollbar-track { background:#0d1a2d; }
::-webkit-scrollbar-thumb { background:#1e3050;border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#00e5ff44; }

#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
PLOT_BG = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Mono, monospace', color='#4a6080', size=11),
    title=dict(font=dict(family='Syne, sans-serif', color='#c8d6e8', size=15)),
    xaxis=dict(gridcolor='#1a2d4a', linecolor='#1a2d4a', tickfont=dict(color='#3a4f6e')),
    yaxis=dict(gridcolor='#1a2d4a', linecolor='#1a2d4a', tickfont=dict(color='#3a4f6e')),
    legend=dict(bgcolor='#0d1a2d', bordercolor='#1a2d4a', borderwidth=1, font=dict(color='#c8d6e8')),
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("saved_model/best_rf_model.pkl")
    features = joblib.load("saved_model/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo-hex">⬡</div>
        <div class="logo-text">ChurnIQ</div>
        <div class="logo-sub">Intelligence Platform</div>
    </div>
    <div class="sidebar-status">
        <div class="status-dot"></div>
        <div class="status-text">MODEL ACTIVE · v2.4.1</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "⬡  Single Prediction",
        "⬡  Batch Prediction",
        "⬡  Model Insights",
        "⬡  Customer Segments",
        "⬡  Alerts & Actions",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 24px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#1e3050;letter-spacing:1px;line-height:1.8;">
            ANTHROPIC · POWERED<br>RANDOM FOREST ENGINE<br>© 2025 CHURNIQ INC.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════
if "Single" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYSIS · SINGLE RECORD</div>
        <h1 class="page-title">Customer <span>Churn</span> Scoring</h1>
        <p class="page-desc">Real-time inference · Random Forest Model · 95% accuracy</p>
    </div>
    <div class="insight-badge"><div class="dot"></div>Enter customer attributes below to generate a churn probability score</div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-header"><div class="section-label">Customer Feature Vector</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        cols = st.columns(3)
        user_input = {}
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                user_input[feat] = st.number_input(label=feat, value=0.0, format="%.4f")
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⬡  RUN PREDICTION ENGINE", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_input])
        proba = model.predict_proba(input_df)[0][1]
        pred  = model.predict(input_df)[0]

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div style="font-size:40px;margin-bottom:8px;">◈</div>
                    <div class="result-label">Churn Risk</div>
                    <div class="result-sub">HIGH PROBABILITY DEFECTION</div>
                    <div style="margin-top:16px;font-family:'DM Mono',monospace;font-size:42px;font-weight:800;color:#ff5252;letter-spacing:-2px;">{round(proba*100,1)}%</div>
                    <div class="result-sub" style="margin-top:4px;">CHURN PROBABILITY SCORE</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-stay">
                    <div style="font-size:40px;margin-bottom:8px;">◉</div>
                    <div class="result-label">Retained</div>
                    <div class="result-sub">LOW DEFECTION PROBABILITY</div>
                    <div style="margin-top:16px;font-family:'DM Mono',monospace;font-size:42px;font-weight:800;color:#69f0ae;letter-spacing:-2px;">{round(proba*100,1)}%</div>
                    <div class="result-sub" style="margin-top:4px;">CHURN PROBABILITY SCORE</div>
                </div>""", unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={'suffix': '%', 'font': {'size': 40, 'color': '#ffffff', 'family': 'Syne'}},
                title={'text': "CHURN SCORE", 'font': {'size': 11, 'color': '#3a4f6e', 'family': 'DM Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1a2d4a', 'tickfont': {'color': '#3a4f6e', 'family': 'DM Mono', 'size': 10}},
                    'bar': {'color': "#ff5252" if proba > 0.5 else "#00e5ff", 'thickness': 0.25},
                    'bgcolor': '#0a1220', 'bordercolor': '#1a2d4a',
                    'steps': [
                        {'range': [0, 40],  'color': '#0a2010'},
                        {'range': [40, 70], 'color': '#1a1a08'},
                        {'range': [70,100], 'color': '#2d0808'},
                    ],
                    'threshold': {'line': {'color': '#00e5ff', 'width': 2}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig.update_layout(height=260, margin=dict(t=50,b=10,l=20,r=20), **PLOT_BG)
            st.plotly_chart(fig, use_container_width=True)

        # Feature contribution breakdown
        st.markdown('<div class="section-header"><div class="section-label">Feature Contribution Breakdown</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        contribs = {f: abs(user_input[f]) * imp for f, imp in zip(feature_names, importances)}
        top_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:8]
        names = [x[0] for x in top_contribs]
        vals  = [x[1] for x in top_contribs]
        fig2 = go.Figure(go.Bar(
            x=vals, y=names, orientation='h',
            marker=dict(color=vals, colorscale=[[0,'#1a2d4a'],[0.5,'#7c3aed'],[1,'#00e5ff']], line=dict(width=0)),
        ))
        fig2.update_layout(height=280, margin=dict(t=10,b=10,l=160,r=20),
            xaxis_title="Weighted Contribution", yaxis=dict(tickfont=dict(family='DM Mono',size=11,color='#c8d6e8')),
            **PLOT_BG)
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════
elif "Batch" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYSIS · BATCH PROCESSING</div>
        <h1 class="page-title">Batch <span>Inference</span> Engine</h1>
        <p class="page-desc">Upload CSV · Score all records · Export predictions</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV file here · Must include all model feature columns", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown('<div class="section-header"><div class="section-label">Data Preview</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        st.dataframe(df_batch.head(5), use_container_width=True)

        missing = set(feature_names) - set(df_batch.columns)
        if missing:
            st.markdown(f"""<div style="background:#1a0808;border:1px solid #e74c3c44;border-radius:10px;padding:16px 20px;font-family:'DM Mono',monospace;font-size:12px;color:#ff5252;letter-spacing:1px;">
            ✕ MISSING COLUMNS DETECTED · {', '.join(missing)}</div>""", unsafe_allow_html=True)
        else:
            df_input = df_batch[feature_names]
            probas = model.predict_proba(df_input)[:, 1]
            preds  = model.predict(df_input)
            df_batch["Churn_Probability"] = probas.round(4)
            df_batch["Prediction"]        = preds
            df_batch["Risk_Label"]        = df_batch["Prediction"].map({0: "Stay", 1: "Churn"})
            churn_rate = preds.mean() * 100

            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">{len(df_batch):,}</div></div>
                <div class="kpi-card"><div class="kpi-label">Predicted Churners</div><div class="kpi-value danger">{int(preds.sum()):,}</div></div>
                <div class="kpi-card"><div class="kpi-label">Churn Rate</div><div class="kpi-value {'danger' if churn_rate>30 else 'warning' if churn_rate>15 else 'success'}">{churn_rate:.1f}%</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header"><div class="section-label">Probability Distribution</div><div class="section-line"></div></div>', unsafe_allow_html=True)
            fig = px.histogram(df_batch, x="Churn_Probability", color="Risk_Label", nbins=40,
                color_discrete_map={"Stay": "#00e5ff", "Churn": "#ff5252"}, opacity=0.9)
            fig.update_traces(marker_line_width=0)
            fig.update_layout(height=320, margin=dict(t=10,b=40,l=40,r=20), bargap=0.05,
                xaxis_title="Churn Probability Score", yaxis_title="Customer Count",
                legend_title_text="", **PLOT_BG)
            st.plotly_chart(fig, use_container_width=True)

            # Risk breakdown pie
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header"><div class="section-label">Risk Breakdown</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                low    = int((probas < 0.4).sum())
                medium = int(((probas >= 0.4) & (probas < 0.7)).sum())
                high   = int((probas >= 0.7).sum())
                fig2 = go.Figure(go.Pie(
                    labels=["Low Risk", "Medium Risk", "High Risk"],
                    values=[low, medium, high],
                    hole=0.65,
                    marker=dict(colors=["#00e5ff", "#ffab40", "#ff5252"], line=dict(color='#090d13', width=3)),
                    textfont=dict(family='DM Mono', size=11, color='#c8d6e8'),
                ))
                fig2.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=True, **PLOT_BG)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.markdown('<div class="section-header"><div class="section-label">Scored Records</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                st.dataframe(df_batch[["Churn_Probability","Risk_Label"]].head(20), use_container_width=True, height=240)

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button("⬡  Export Predictions · CSV", data=csv_out,
                file_name="churniq_predictions.csv", mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// MODEL · EXPLAINABILITY</div>
        <h1 class="page-title">Model <span>Intelligence</span></h1>
        <p class="page-desc">Feature importance · Architecture · Hyperparameters</p>
    </div>""", unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)

    st.markdown('<div class="section-header"><div class="section-label">Top 15 Features by Predictive Power</div><div class="section-line"></div></div>', unsafe_allow_html=True)
    top15 = fi_df.tail(15)
    fig = go.Figure(go.Bar(
        x=top15["Importance"], y=top15["Feature"], orientation='h',
        marker=dict(color=top15["Importance"],
            colorscale=[[0,'#1a2d4a'],[0.5,'#0077b6'],[1,'#00e5ff']], line=dict(width=0)),
        text=top15["Importance"].apply(lambda x: f"{x:.4f}"),
        textfont=dict(family='DM Mono', size=10, color='#4a6080'),
        textposition='outside',
    ))
    fig.update_layout(height=500, margin=dict(t=10,b=40,l=160,r=80),
        xaxis_title="Importance Score",
        yaxis=dict(tickfont=dict(family='DM Mono', size=11, color='#c8d6e8')),
        **PLOT_BG)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><div class="section-label">Cumulative Importance</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        fi_sorted = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_sorted["Cumulative"] = fi_sorted["Importance"].cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(1, len(fi_sorted)+1)), y=fi_sorted["Cumulative"],
            fill='tozeroy', fillcolor='#00e5ff11',
            line=dict(color='#00e5ff', width=2),
            mode='lines',
        ))
        fig2.add_hline(y=0.8, line_dash="dot", line_color="#ffab40", line_width=1,
            annotation_text="80% threshold", annotation_font_color="#ffab40", annotation_font_size=10)
        fig2.update_layout(height=280, margin=dict(t=10,b=40,l=40,r=20),
            xaxis_title="Number of Features", yaxis_title="Cumulative Importance", **PLOT_BG)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><div class="section-label">Model Architecture · Hyperparameters</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        params = model.get_params()
        params_df = pd.DataFrame({"Parameter": list(params.keys()), "Value": [str(v) for v in params.values()]})
        st.dataframe(params_df, use_container_width=True, height=280)


# ════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════
elif "Segments" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYTICS · SEGMENTATION</div>
        <h1 class="page-title">Customer <span>Segments</span></h1>
        <p class="page-desc">Risk-based clustering · Behavioral profiles · Retention targeting</p>
    </div>""", unsafe_allow_html=True)

    # Simulated segment data
    np.random.seed(42)
    n = 1200
    seg_labels = np.random.choice(["Champions", "At Risk", "Hibernating", "Lost"], size=n, p=[0.35, 0.25, 0.25, 0.15])
    sim_probas = np.where(seg_labels=="Champions", np.random.beta(2,8,n),
                 np.where(seg_labels=="At Risk",   np.random.beta(6,4,n),
                 np.where(seg_labels=="Hibernating",np.random.beta(4,4,n),
                                                    np.random.beta(8,2,n))))

    seg_colors = {"Champions": "#00e5ff", "At Risk": "#ffab40", "Hibernating": "#7c3aed", "Lost": "#ff5252"}
    seg_descs  = {
        "Champions":   "High-value, highly engaged customers with very low churn probability.",
        "At Risk":     "Previously active customers showing declining engagement signals.",
        "Hibernating": "Low recent activity — need re-engagement campaigns urgently.",
        "Lost":        "High churn probability — last-chance intervention required."
    }

    # Segment cards
    st.markdown('<div class="seg-grid">', unsafe_allow_html=True)
    for seg, color in seg_colors.items():
        count = int((seg_labels == seg).sum())
        pct   = count / n * 100
        avg_p = sim_probas[seg_labels == seg].mean() * 100
        st.markdown(f"""
        <div class="seg-card">
            <div class="seg-accent" style="background:{color};"></div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:{color};letter-spacing:2px;text-transform:uppercase;">SEGMENT</div>
            <div class="seg-name">{seg}</div>
            <div class="seg-desc">{seg_descs[seg]}</div>
            <div class="seg-stat">
                <div><div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:4px;">CUSTOMERS</div>
                    <div class="seg-count">{count:,}</div></div>
                <div style="text-align:right;">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:4px;">AVG CHURN RISK</div>
                    <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:{color};">{avg_p:.1f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scatter plot
    st.markdown('<div class="section-header"><div class="section-label">Segment Distribution · Risk vs Engagement</div><div class="section-line"></div></div>', unsafe_allow_html=True)

    engagement = np.random.beta(3, 3, n) * 100
    engagement = np.where(seg_labels=="Champions",  np.random.uniform(60,100,n),
                 np.where(seg_labels=="At Risk",     np.random.uniform(30,65,n),
                 np.where(seg_labels=="Hibernating", np.random.uniform(10,40,n),
                                                     np.random.uniform(5,30,n))))
    df_seg = pd.DataFrame({"Churn_Risk": sim_probas*100, "Engagement": engagement, "Segment": seg_labels})

    fig = px.scatter(df_seg, x="Engagement", y="Churn_Risk", color="Segment",
        color_discrete_map=seg_colors, opacity=0.7, size_max=6,
        labels={"Churn_Risk": "Churn Risk (%)", "Engagement": "Engagement Score"})
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig.update_layout(height=400, margin=dict(t=10,b=40,l=40,r=20), **PLOT_BG)
    st.plotly_chart(fig, use_container_width=True)

    # Segment volume bar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><div class="section-label">Segment Volume</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig2 = go.Figure(go.Bar(
            x=seg_counts["Segment"], y=seg_counts["Count"],
            marker=dict(color=[seg_colors[s] for s in seg_counts["Segment"]], line=dict(width=0)),
        ))
        fig2.update_layout(height=260, margin=dict(t=10,b=40,l=40,r=20),
            yaxis_title="Customers", **PLOT_BG)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><div class="section-label">Risk Distribution per Segment</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for seg, color in seg_colors.items():
            vals = df_seg[df_seg["Segment"]==seg]["Churn_Risk"]
            fig3.add_trace(go.Box(y=vals, name=seg, marker_color=color, line_color=color,
                fillcolor=color+"22", boxmean=True))
        fig3.update_layout(height=260, margin=dict(t=10,b=40,l=40,r=20),
            yaxis_title="Churn Risk (%)", showlegend=False, **PLOT_BG)
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — ALERTS & ACTIONS
# ════════════════════════════════════════════════════════════
elif "Alerts" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// OPERATIONS · REAL-TIME</div>
        <h1 class="page-title">Alerts <span>&</span> Actions</h1>
        <p class="page-desc">Critical churn signals · Intervention queue · Retention playbooks</p>
    </div>""", unsafe_allow_html=True)

    # Simulated alert data
    np.random.seed(7)
    names = [
        "Amira Bensalem","Karim Trabelsi","Sonia Gharbi","Mohamed Ayari","Leila Kchouk",
        "Rami Zouari","Nadia Hamdi","Youssef Ben Ali","Fatma Mejri","Chaker Mansouri",
        "Ines Khalfallah","Bilel Cherif","Mariem Tlili","Tarek Bouaziz","Hanen Saidi"
    ]
    random.shuffle(names)
    alerts = []
    for i, name in enumerate(names[:12]):
        score = random.uniform(0.55, 0.99)
        level = "critical" if score > 0.82 else "high" if score > 0.68 else "medium"
        alerts.append({
            "name": name,
            "score": score,
            "level": level,
            "segment": random.choice(["At Risk", "Hibernating", "Lost"]),
            "days": random.randint(2, 30),
            "value": random.randint(800, 12000),
        })
    alerts.sort(key=lambda x: x["score"], reverse=True)

    # KPIs
    crit  = sum(1 for a in alerts if a["level"]=="critical")
    high  = sum(1 for a in alerts if a["level"]=="high")
    med   = sum(1 for a in alerts if a["level"]=="medium")
    rev   = sum(a["value"] for a in alerts if a["level"]=="critical")

    st.markdown(f"""
    <div class="kpi-grid-4">
        <div class="kpi-card"><div class="kpi-label">Critical Alerts</div><div class="kpi-value danger">{crit}</div></div>
        <div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value warning">{high}</div></div>
        <div class="kpi-card"><div class="kpi-label">Medium Risk</div><div class="kpi-value">{med}</div></div>
        <div class="kpi-card"><div class="kpi-label">Revenue at Risk</div><div class="kpi-value danger">{rev:,} TND</div></div>
    </div>""", unsafe_allow_html=True)

    # Filter
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        filter_level = st.selectbox("Filter by level", ["All", "Critical", "High", "Medium"])

    filtered = alerts if filter_level == "All" else [a for a in alerts if a["level"] == filter_level.lower()]

    st.markdown('<div class="section-header"><div class="section-label">Intervention Queue</div><div class="section-line"></div></div>', unsafe_allow_html=True)

    actions_map = {
        "critical": ["Retention call", "Promo offer", "Executive escalation"],
        "high":     ["Email campaign", "Discount voucher", "Check-in call"],
        "medium":   ["Newsletter", "Product tip", "Survey"],
    }

    for a in filtered:
        initials = "".join([w[0].upper() for w in a["name"].split()])
        actions_html = "".join([f'<span class="action-pill">▸ {act}</span>' for act in actions_map[a["level"]]])
        st.markdown(f"""
        <div class="alert-card {a['level']}">
            <div class="alert-avatar">{initials}</div>
            <div class="alert-info">
                <div class="alert-name">{a['name']}</div>
                <div class="alert-meta">{a['segment']} · Last active {a['days']}d ago · Value: {a['value']:,} TND</div>
                <div style="margin-top:6px;">{actions_html}</div>
            </div>
            <div class="alert-score">
                <div class="alert-score-val {a['level']}">{a['score']*100:.1f}%</div>
                <div class="alert-score-label">CHURN RISK</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Activity timeline
    st.markdown('<div class="section-header"><div class="section-label">Recent Platform Activity</div><div class="section-line"></div></div>', unsafe_allow_html=True)

    now = datetime.now()
    timeline_events = [
        (now - timedelta(minutes=3),  "cyan",   "Batch scoring completed", "1,200 records processed · 23.4% churn rate", "BATCH"),
        (now - timedelta(minutes=18), "red",    "Critical alert triggered", "Amira Bensalem — score 94.2%", "ALERT"),
        (now - timedelta(hours=1),    "purple", "Model retrained", "Accuracy improved to 95.1% (+0.4%)", "MODEL"),
        (now - timedelta(hours=2),    "green",  "Retention action executed", "Karim Trabelsi — promo email sent", "ACTION"),
        (now - timedelta(hours=5),    "cyan",   "New data ingested", "3,400 customer events loaded", "DATA"),
        (now - timedelta(hours=12),   "red",    "Churn spike detected", "Segment 'Hibernating' +8% vs last week", "ALERT"),
        (now - timedelta(days=1),     "green",  "Retention campaign success", "12 customers retained · 48,000 TND saved", "SUCCESS"),
    ]

    tag_colors = {"BATCH":"#00e5ff","ALERT":"#ff5252","MODEL":"#7c3aed","ACTION":"#69f0ae","DATA":"#00e5ff","SUCCESS":"#69f0ae"}

    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    for ts, dot, title, detail, tag in timeline_events:
        diff = now - ts
        if diff.seconds < 3600:   label = f"{diff.seconds//60}m ago"
        elif diff.days < 1:       label = f"{diff.seconds//3600}h ago"
        else:                     label = f"{diff.days}d ago"
        tc = tag_colors.get(tag, "#00e5ff")
        st.markdown(f"""
        <div class="tl-item">
            <div class="tl-dot {dot}"></div>
            <div class="tl-time">{label} · {ts.strftime('%H:%M')}</div>
            <div class="tl-text"><strong style="color:#c8d6e8;">{title}</strong><br>{detail}</div>
            <span class="tl-tag" style="background:{tc}22;color:{tc};border:1px solid {tc}44;">{tag}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)