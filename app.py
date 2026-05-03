import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Single Prediction"

def nav(p):
    st.session_state.page = p

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background-color:#090d13!important;color:#c8d6e8!important;font-family:'Syne',sans-serif!important}
[data-testid="stSidebar"]{background:#0d1420!important;border-right:1px solid #1a2540!important;padding-top:0!important;min-width:260px!important;width:260px!important}
section[data-testid="stSidebar"][aria-expanded="false"]{min-width:260px!important;width:260px!important;transform:none!important;display:block!important}
[data-testid="collapsedControl"]{display:none!important}
[data-testid="stSidebar"]::before{content:'';display:block;height:3px;background:linear-gradient(90deg,#00e5ff,#7c3aed,#00e5ff);background-size:200% 100%;animation:shimmer 3s linear infinite}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 8px #00c853}50%{opacity:.6;box-shadow:0 0 4px #00c853}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

div[data-testid="stSidebar"] .stButton>button{background:transparent!important;border:1px solid transparent!important;border-radius:8px!important;padding:10px 14px!important;color:#4a6080!important;font-size:13px!important;font-weight:600!important;letter-spacing:.5px!important;text-align:left!important;width:100%!important;transition:all .2s ease!important;text-transform:none!important;box-shadow:none!important;transform:none!important;margin-bottom:2px!important}
div[data-testid="stSidebar"] .stButton>button:hover{background:#131e30!important;color:#c8d6e8!important;border-color:#1e3050!important;transform:none!important;box-shadow:none!important}

.sidebar-brand{padding:28px 24px 20px;border-bottom:1px solid #1a2540;margin-bottom:12px}
.logo-hex{font-size:28px;color:#00e5ff;font-weight:800}
.logo-text{font-size:20px;font-weight:800;color:#fff;letter-spacing:1px}
.logo-sub{font-size:10px;font-family:'DM Mono',monospace;color:#3a4f6e;letter-spacing:3px;text-transform:uppercase;margin-top:2px}
.sidebar-status{margin:0 16px 16px;padding:10px 14px;background:#0a1a0a;border:1px solid #0d3d0d;border-radius:8px;display:flex;align-items:center;gap:8px}
.status-dot{width:7px;height:7px;border-radius:50%;background:#00c853;box-shadow:0 0 8px #00c853;animation:pulse 2s infinite;flex-shrink:0}
.status-text{font-family:'DM Mono',monospace;font-size:10px;color:#2e7d32;letter-spacing:1px}
.nav-section{font-family:'DM Mono',monospace;font-size:10px;color:#1e3050;letter-spacing:3px;text-transform:uppercase;padding:0 26px 8px}

.page-header{padding:32px 0 24px;border-bottom:1px solid #1a2540;margin-bottom:32px;animation:fadeIn .4s ease}
.page-tag{font-family:'DM Mono',monospace;font-size:11px;color:#00e5ff;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px}
.page-title{font-size:36px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1.1;margin:0}
.page-title span{color:#00e5ff}
.page-desc{font-size:14px;color:#3a4f6e;margin-top:8px;font-family:'DM Mono',monospace}

.section-header{display:flex;align-items:center;gap:12px;margin:28px 0 16px}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,#1a2d4a,transparent)}
.section-label{font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;letter-spacing:3px;text-transform:uppercase;white-space:nowrap}

.insight-badge{display:inline-flex;align-items:center;gap:6px;background:#0d1a2d;border:1px solid #1a2d4a;border-radius:20px;padding:6px 14px;font-family:'DM Mono',monospace;font-size:11px;color:#4a6080;letter-spacing:1px;margin-bottom:20px}
.insight-badge .dot{width:5px;height:5px;border-radius:50%;background:#00e5ff}

.kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}
.kpi-grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.kpi-card{background:#0d1a2d;border:1px solid #1a2d4a;border-radius:14px;padding:20px 24px;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#00e5ff,#7c3aed)}
.kpi-label{font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.kpi-value{font-size:32px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1}
.kpi-value.danger{color:#ff5252}
.kpi-value.warning{color:#ffab40}
.kpi-value.success{color:#69f0ae}

.result-churn{background:linear-gradient(135deg,#1a0a0a,#2d0d0d);border:1px solid rgba(231,76,60,.27);border-radius:16px;padding:28px;text-align:center;position:relative;overflow:hidden}
.result-churn::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#e74c3c,transparent)}
.result-stay{background:linear-gradient(135deg,#0a1a0a,#0d2d0d);border:1px solid rgba(0,200,83,.27);border-radius:16px;padding:28px;text-align:center;position:relative;overflow:hidden}
.result-stay::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00c853,transparent)}
.result-label{font-size:22px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin:4px 0}
.result-churn .result-label{color:#ff5252}
.result-stay .result-label{color:#69f0ae}
.result-sub{font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;letter-spacing:2px}

.seg-card{background:#0d1a2d;border:1px solid #1a2d4a;border-radius:14px;padding:22px;position:relative;overflow:hidden;margin-bottom:16px}
.seg-card:hover{border-color:rgba(0,229,255,.27)}
.seg-name{font-size:17px;font-weight:800;color:#fff;margin:10px 0 4px}
.seg-desc{font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e;line-height:1.6;margin-bottom:14px}

.alert-card{background:#0d1420;border:1px solid #1a2d4a;border-radius:12px;padding:16px 20px;margin-bottom:10px;display:flex;align-items:center;gap:16px}
.alert-card.critical{border-left:3px solid #ff5252}
.alert-card.high{border-left:3px solid #ffab40}
.alert-card.medium{border-left:3px solid #00e5ff}
.alert-avatar{width:40px;height:40px;border-radius:10px;background:#131e30;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:#c8d6e8;flex-shrink:0}
.alert-info{flex:1;min-width:0}
.alert-name{font-size:14px;font-weight:700;color:#fff;margin-bottom:2px}
.alert-meta{font-family:'DM Mono',monospace;font-size:11px;color:#3a4f6e}
.alert-score{text-align:right;flex-shrink:0}
.alert-score-val{font-family:'DM Mono',monospace;font-size:20px;font-weight:500}
.alert-score-val.critical{color:#ff5252}
.alert-score-val.high{color:#ffab40}
.alert-score-val.medium{color:#00e5ff}
.alert-score-label{font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px}
.action-pill{display:inline-flex;align-items:center;gap:4px;background:#0a1a2a;border:1px solid #1a3050;border-radius:6px;padding:4px 10px;font-family:'DM Mono',monospace;font-size:10px;color:#00e5ff;letter-spacing:.5px;margin:2px}

.tl-item{position:relative;margin-bottom:20px;padding-left:20px}
.tl-dot{position:absolute;left:-20px;top:4px;width:9px;height:9px;border-radius:50%;border:2px solid}
.tl-dot.cyan{border-color:#00e5ff;background:rgba(0,229,255,.2)}
.tl-dot.purple{border-color:#7c3aed;background:rgba(124,58,237,.2)}
.tl-dot.red{border-color:#ff5252;background:rgba(255,82,82,.2)}
.tl-dot.green{border-color:#69f0ae;background:rgba(105,240,174,.2)}
.tl-time{font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:3px}
.tl-text{font-size:13px;color:#c8d6e8;line-height:1.5}
.tl-tag{display:inline-block;font-family:'DM Mono',monospace;font-size:10px;padding:2px 7px;border-radius:4px;margin-top:4px}

.stNumberInput input,.stTextInput input{background:#070d18!important;border:1px solid #1a2d4a!important;border-radius:8px!important;color:#c8d6e8!important;font-family:'DM Mono',monospace!important;font-size:13px!important}
.stNumberInput input:focus,.stTextInput input:focus{border-color:#00e5ff!important;box-shadow:0 0 0 2px rgba(0,229,255,.08)!important}
label[data-testid="stWidgetLabel"]{color:#4a6080!important;font-size:11px!important;font-family:'DM Mono',monospace!important;letter-spacing:1px!important;text-transform:uppercase!important}
[data-testid="stFormSubmitButton"] button{background:linear-gradient(135deg,#00b4d8,#7c3aed)!important;border:none!important;border-radius:10px!important;color:#fff!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:14px!important;letter-spacing:1px!important;padding:14px 28px!important;text-transform:uppercase!important;width:100%!important}
[data-testid="stDataFrame"]{border:1px solid #1a2d4a!important;border-radius:12px!important;overflow:hidden!important}
[data-testid="stDataFrame"] th{background:#0d1a2d!important;color:#4a6080!important;font-family:'DM Mono',monospace!important;font-size:11px!important;letter-spacing:1px!important;text-transform:uppercase!important}
[data-testid="stDataFrame"] td{color:#c8d6e8!important;font-family:'DM Mono',monospace!important;font-size:12px!important}
[data-testid="stFileUploader"]{background:#0d1a2d!important;border:1px dashed #1e3a5a!important;border-radius:14px!important}
[data-testid="stDownloadButton"] button{background:transparent!important;border:1px solid rgba(0,229,255,.27)!important;color:#00e5ff!important;font-family:'DM Mono',monospace!important;border-radius:10px!important}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#0d1a2d}
::-webkit-scrollbar-thumb{background:#1e3050;border-radius:3px}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout (NO xaxis/yaxis to avoid conflict) ─────
def plot_layout(**kwargs):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Mono, monospace', color='#4a6080', size=11),
        legend=dict(bgcolor='#0d1a2d', bordercolor='#1a2d4a', borderwidth=1, font=dict(color='#c8d6e8')),
    )
    base.update(kwargs)
    return base

# ── Feature Engineering (mirrors notebook exactly) ───────────
@st.cache_data
def engineer_features(df_raw):
    df = df_raw.copy()

    # Drop Churn if present
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    # Binary encoding
    if df['International plan'].dtype == object:
        df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    if df['Voice mail plan'].dtype == object:
        df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})

    # One-hot Area code
    if 'Area code' in df.columns:
        df = pd.get_dummies(df, columns=['Area code'], prefix='Area')

    # Label encode State
    if 'State' in df.columns and df['State'].dtype == object:
        le = LabelEncoder()
        df['State'] = le.fit_transform(df['State'])

    # Feature engineering
    df['Total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
    df['Total_calls']   = df['Total day calls']   + df['Total eve calls']   + df['Total night calls']   + df['Total intl calls']
    df['Avg_min_per_call']    = (df['Total_minutes'] / df['Total_calls']).fillna(0)
    df['CS_calls_per_tenure'] = df['Customer service calls'] / (df['Account length'] + 1)
    df['Intl_percent']        = (df['Total intl minutes'] / df['Total_minutes']).fillna(0)

    # Drop charge columns
    cols_to_drop = [c for c in ['Total day charge','Total eve charge','Total night charge','Total intl charge'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Scale
    features_to_scale = [c for c in [
        'Account length','Number vmail messages','Total day minutes','Total day calls',
        'Total eve minutes','Total eve calls','Total night minutes','Total night calls',
        'Total intl minutes','Total intl calls','Customer service calls',
        'Total_minutes','Total_calls','Avg_min_per_call','CS_calls_per_tenure','Intl_percent'
    ] if c in df.columns]

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Ensure bool columns (from get_dummies) become int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

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

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("Single Prediction", "⬡  Single Prediction"),
        ("Batch Prediction",  "⬡  Batch Prediction"),
        ("Model Insights",    "⬡  Model Insights"),
        ("Customer Segments", "⬡  Customer Segments"),
        ("Alerts & Actions",  "⬡  Alerts & Actions"),
    ]
    for key, label in pages:
        if st.session_state.page == key:
            st.markdown(f"""
            <div style="background:#0f2040;border:1px solid rgba(0,229,255,.2);border-radius:8px;
                padding:10px 14px;color:#00e5ff;font-size:13px;font-weight:600;
                letter-spacing:.5px;margin:0 12px 2px;font-family:'Syne',sans-serif;">
                {label}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(label, key=f"nav_{key}"):
                nav(key); st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 24px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#1e3050;letter-spacing:1px;line-height:1.8;">
            ANTHROPIC · POWERED<br>RANDOM FOREST ENGINE<br>© 2025 CHURNIQ INC.
        </div>
    </div>""", unsafe_allow_html=True)

page = st.session_state.page

# ── Helpers ───────────────────────────────────────────────────
def section(label):
    st.markdown(f'<div class="section-header"><div class="section-label">{label}</div><div class="section-line"></div></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════
if page == "Single Prediction":
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYSIS · SINGLE RECORD</div>
        <h1 class="page-title">Customer <span>Churn</span> Scoring</h1>
        <p class="page-desc">Real-time inference · Random Forest · Feature Engineering Auto-Applied</p>
    </div>
    <div class="insight-badge"><div class="dot"></div>Upload a single-row CSV with raw customer data — feature engineering is applied automatically</div>
    """, unsafe_allow_html=True)

    uploaded_single = st.file_uploader("Upload a single customer CSV (raw format — same as training data)", type=["csv"], key="single_upload")

    if uploaded_single:
        df_raw = pd.read_csv(uploaded_single)
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:11px;color:#3a4f6e;margin-bottom:8px;">RAW INPUT PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(1), use_container_width=True)

        try:
            df_eng = engineer_features(df_raw)
            # Align columns to model
            missing_cols = set(feature_names) - set(df_eng.columns)
            for c in missing_cols:
                df_eng[c] = 0
            df_eng = df_eng[feature_names]

            proba = model.predict_proba(df_eng)[0][1]
            pred  = model.predict(df_eng)[0]

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
                        'axis': {'range': [0, 100], 'tickcolor': '#1a2d4a', 'tickfont': {'color': '#3a4f6e', 'size': 10}},
                        'bar': {'color': "#ff5252" if proba > 0.5 else "#00e5ff", 'thickness': 0.25},
                        'bgcolor': '#0a1220', 'bordercolor': '#1a2d4a',
                        'steps': [
                            {'range': [0,  40], 'color': '#0a2010'},
                            {'range': [40, 70], 'color': '#1a1a08'},
                            {'range': [70,100], 'color': '#2d0808'},
                        ],
                        'threshold': {'line': {'color': '#00e5ff', 'width': 2}, 'thickness': 0.8, 'value': 50}
                    }
                ))
                fig.update_layout(**plot_layout(height=260, margin=dict(t=50,b=10,l=20,r=20)))
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance chart
            section("Feature Importance — Model View")
            importances = model.feature_importances_
            fi = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance").tail(10)
            fig2 = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"], orientation='h',
                marker=dict(
                    color=list(fi["Importance"]),
                    colorscale=[[0,'#1a2d4a'],[0.5,'#7c3aed'],[1,'#00e5ff']],
                    line=dict(width=0)
                ),
            ))
            fig2.update_layout(**plot_layout(
                height=320,
                margin=dict(t=10,b=10,l=180,r=20),
                xaxis=dict(title="Importance Score", gridcolor='#1a2d4a', linecolor='#1a2d4a', tickfont=dict(color='#3a4f6e')),
                yaxis=dict(tickfont=dict(family='DM Mono', size=11, color='#c8d6e8')),
            ))
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du feature engineering : {e}")


# ════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYSIS · BATCH PROCESSING</div>
        <h1 class="page-title">Batch <span>Inference</span> Engine</h1>
        <p class="page-desc">Upload raw CSV · Feature engineering auto-applied · Export results</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload raw CSV (original format — Churn column optional)", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        section("Raw Data Preview")
        st.dataframe(df_raw.head(5), use_container_width=True)

        try:
            # 1. Feature Engineering
            df_eng = engineer_features(df_raw)
            
            # 2. Alignement des colonnes avec le modèle
            missing_cols = set(feature_names) - set(df_eng.columns)
            for c in missing_cols:
                df_eng[c] = 0
            df_input = df_eng[feature_names]

            # 3. Prédictions
            probas = model.predict_proba(df_input)[:, 1]
            preds  = model.predict(df_input)

            # 4. Construction du DataFrame de sortie
            df_out = df_raw.copy()
            if 'Churn' in df_out.columns:
                df_out = df_out.drop(columns=['Churn'])
            
            df_out["Churn_Probability"] = probas.round(4)
            df_out["Prediction"]        = preds
            df_out["Risk_Label"]        = pd.Series(preds).map({0:"Stay", 1:"Churn"}).values
            
            # --- CRUCIAL : Sauvegarde pour les autres pages ---
            st.session_state.df_out = df_out 
            # --------------------------------------------------

            churn_rate = preds.mean() * 100

            # 5. Affichage des KPIs
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">{len(df_out):,}</div></div>
                <div class="kpi-card"><div class="kpi-label">Predicted Churners</div><div class="kpi-value danger">{int(preds.sum()):,}</div></div>
                <div class="kpi-card"><div class="kpi-label">Churn Rate</div>
                    <div class="kpi-value {'danger' if churn_rate>30 else 'warning' if churn_rate>15 else 'success'}">{churn_rate:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # 6. Graphique : Distribution des probabilités
            section("Probability Distribution")
            fig = px.histogram(df_out, x="Churn_Probability", color="Risk_Label", nbins=40,
                color_discrete_map={"Stay":"#00e5ff","Churn":"#ff5252"}, opacity=0.9)
            
            fig.update_traces(marker_line_width=0)
            
            # Correction syntaxe Plotly (sans les **)
            fig.update_layout(plot_layout(
                height=300,
                margin=dict(t=10,b=40,l=40,r=20),
                bargap=0.05,
                xaxis=dict(title="Churn Probability", gridcolor='#1a2d4a', linecolor='#1a2d4a', tickfont=dict(color='#3a4f6e')),
                yaxis=dict(title="Count", gridcolor='#1a2d4a', linecolor='#1a2d4a', tickfont=dict(color='#3a4f6e'))
            ))
            st.plotly_chart(fig, use_container_width=True)

            # 7. Détails et Export
            c1, c2 = st.columns(2)
            with c1:
                section("Risk Breakdown")
                low    = int((probas < 0.4).sum())
                medium = int(((probas >= 0.4) & (probas < 0.7)).sum())
                high   = int((probas >= 0.7).sum())
                
                fig2 = go.Figure(go.Pie(
                    labels=["Low Risk","Medium Risk","High Risk"], 
                    values=[low,medium,high], 
                    hole=0.65,
                    marker=dict(colors=["#00e5ff","#ffab40","#ff5252"], line=dict(color='#090d13', width=3)),
                    textfont=dict(family='DM Mono', size=11, color='#c8d6e8'),
                ))
                fig2.update_layout(plot_layout(height=260, margin=dict(t=10,b=10,l=10,r=10)))
                st.plotly_chart(fig2, use_container_width=True)
                
            with c2:
                section("Scored Records (Top 20)")
                st.dataframe(df_out[["Churn_Probability","Risk_Label"]].head(20), use_container_width=True, height=240)

            # Bouton de téléchargement
            csv_out = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬡  Export Predictions · CSV",
                data=csv_out,
                file_name=f"churniq_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'analyse : {e}")
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS (SPÉCIFIQUE AU CSV CHARGÉ)
# ════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// MODEL · EXPLAINABILITY</div>
        <h1 class="page-title">Model <span>Intelligence</span></h1>
        <p class="page-desc">Feature importance · Data Specific Insights · Hyperparameters</p>
    </div>""", unsafe_allow_html=True)

    # Vérification si un fichier CSV a été traité
    has_data = 'df_out' in st.session_state
    
    if has_data:
        df_active = st.session_state.df_out
        n_records = len(df_active)
        churn_p = (df_active['Prediction'] == 1).mean() * 100
        
        st.markdown(f"""
        <div style="background: rgba(0, 229, 255, 0.05); border: 1px solid #00e5ff; padding: 15px; border-radius: 10px; margin-bottom: 25px;">
            <strong style="color: #00e5ff;">Analysis Context:</strong> Ces insights sont basés sur le fichier CSV chargé 
            contenant <b>{n_records:,} clients</b> avec un taux de churn prédit de <b>{churn_p:.1f}%</b>.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("💡 Note : Chargez un fichier dans 'Batch Prediction' pour voir les statistiques spécifiques à vos données.")

    # 1. FEATURE IMPORTANCE (Statique par rapport au modèle, mais visuel corrigé)
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)

    section("Top 15 Predictive Drivers")
    top15 = fi_df.tail(15)
    fig = go.Figure(go.Bar(
        x=list(top15["Importance"]),
        y=list(top15["Feature"]),
        orientation='h',
        marker=dict(
            color=list(top15["Importance"]),
            colorscale=[[0,'#1a2d4a'],[0.5,'#0077b6'],[1,'#00e5ff']],
            line=dict(width=0)
        ),
        text=[f"{v:.4f}" for v in top15["Importance"]],
        textfont=dict(family='DM Mono', size=10, color='#c8d6e8'),
        textposition='outside',
    ))
    fig.update_layout(plot_layout(
        height=450,
        margin=dict(t=10,b=40,l=160,r=80),
        xaxis=dict(title="Global Impact Score", gridcolor='#1a2d4a'),
        yaxis=dict(tickfont=dict(family='DM Mono', size=11, color='#c8d6e8')),
    ))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    
    with c1:
        # 2. DISTRIBUTION DES PROBABILITÉS DU CSV (S'il existe)
        section("Score Distribution (Last CSV)")
        if has_data:
            fig2 = px.violin(st.session_state.df_out, y="Churn_Probability", box=True, points="all",
                             color_discrete_sequence=['#00e5ff'])
            fig2.update_layout(plot_layout(height=300, margin=dict(t=10,b=10,l=10,r=10)))
            fig2.update_layout(yaxis=dict(gridcolor='#1a2d4a', title="Probability"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("En attente de données...")

    with c2:
        # 3. HYPERPARAMÈTRES RÉELS
        section("Model Architecture")
        params = model.get_params()
        # On ne garde que les paramètres clés pour plus de lisibilité
        key_params = ['n_estimators', 'max_depth', 'min_samples_split', 'criterion', 'random_state']
        filtered_params = {k: params[k] for k in key_params if k in params}
        
        st.markdown('<div style="background:#090d13; padding:10px; border-radius:5px; border:1px solid #1a2d4a;">', unsafe_allow_html=True)
        for k, v in filtered_params.items():
            st.markdown(f"**{k}**: `{v}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if hasattr(model, "oob_score_"):
            st.metric("OOB Score (Validation)", f"{model.oob_score_:.4f}")

    # 4. EXPLICATION DES VARIABLES TOP (Adapté à vos données télécom)
    if has_data:
        section("Key Variable Analysis")
        # Exemple : Analyse automatique de la variable la plus importante
        top_var = fi_df.iloc[-1]["Feature"]
        avg_val = st.session_state.df_out[top_var].mean() if top_var in st.session_state.df_out else 0
        
        st.write(f"La variable **{top_var}** est le levier n°1 de prédiction.")
        st.write(f"Moyenne sur le fichier chargé : `{avg_val:.2f}`")


# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTS (DYNAMIQUE VIA CSV)
# ════════════════════════════════════════════════════════════
elif page == "Customer Segments":
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// ANALYTICS · SEGMENTATION</div>
        <h1 class="page-title">Customer <span>Segments</span></h1>
        <p class="page-desc">Risk-based clustering · Behavioral profiles · Retention targeting</p>
    </div>""", unsafe_allow_html=True)

    if 'df_out' not in st.session_state:
        st.warning("⚠️ Aucune donnée disponible. Veuillez d'abord traiter un fichier dans 'Batch Prediction'.")
    else:
        df_active = st.session_state.df_out.copy()

        # 1. SEGMENTATION RÉELLE BASÉE SUR LE SCORE DU MODÈLE
        # On définit les segments selon les probabilités prédites
        def assign_segment(prob):
            if prob < 0.3: return "Champions"
            elif prob < 0.6: return "Hibernating"
            elif prob < 0.8: return "At Risk"
            else: return "Lost"

        df_active['Segment'] = df_active['Churn_Probability'].apply(assign_segment)
        
        # On simule un score d'engagement basé sur l'usage (ex: Total day minutes) 
        # pour le scatter plot, normalisé de 0 à 100
        if 'Total day minutes' in df_active.columns:
            max_min = df_active['Total day minutes'].max()
            df_active['Engagement'] = (df_active['Total day minutes'] / max_min) * 100
        else:
            df_active['Engagement'] = (1 - df_active['Churn_Probability']) * 100

        seg_colors = {"Champions":"#00e5ff","At Risk":"#ffab40","Hibernating":"#7c3aed","Lost":"#ff5252"}
        seg_descs  = {
            "Champions":   "High-value, very low churn probability. Maintain loyalty.",
            "Hibernating": "Low activity or engagement. Needs re-activation.",
            "At Risk":     "High probability of leaving. Immediate action required.",
            "Lost":        "Critical churn score. Last-chance intervention.",
        }

        # 2. AFFICHAGE DES CARTES DE SEGMENTS
        c1, c2 = st.columns(2)
        segments_list = ["Champions", "Hibernating", "At Risk", "Lost"]
        
        for i, seg in enumerate(segments_list):
            color = seg_colors[seg]
            seg_data = df_active[df_active["Segment"] == seg]
            count = len(seg_data)
            avg_p = seg_data['Churn_Probability'].mean() * 100 if count > 0 else 0
            
            card = f"""
            <div class="seg-card">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:{color};letter-spacing:2px;text-transform:uppercase;">SEGMENT</div>
                <div class="seg-name">{seg}</div>
                <div class="seg-desc">{seg_descs[seg]}</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:8px;">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:4px;">CUSTOMERS</div>
                        <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:#c8d6e8;">{count:,}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4f6e;letter-spacing:1px;margin-bottom:4px;">AVG RISK</div>
                        <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:{color};">{avg_p:.1f}%</div>
                    </div>
                </div>
            </div>"""
            if i % 2 == 0: c1.markdown(card, unsafe_allow_html=True)
            else: c2.markdown(card, unsafe_allow_html=True)

        # 3. SCATTER PLOT : RISK VS ENGAGEMENT
        section("Risk vs Engagement · Real Distribution")
        fig = px.scatter(df_active, x="Engagement", y="Churn_Probability", color="Segment",
            color_discrete_map=seg_colors, opacity=0.7,
            hover_data=['Churn_Probability'],
            labels={"Churn_Probability":"Churn Risk (%)","Engagement":"Usage Engagement Index"})
        
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_layout(plot_layout(
            height=400,
            xaxis=dict(title="Usage Score (Engagement)", gridcolor='#1a2d4a'),
            yaxis=dict(title="Churn Probability", gridcolor='#1a2d4a'),
        ))
        st.plotly_chart(fig, use_container_width=True)

        # 4. BAR CHART & BOX PLOT
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            section("Volume par Segment")
            sc = df_active["Segment"].value_counts().reindex(segments_list).reset_index()
            sc.columns = ["Segment","Count"]
            fig2 = go.Figure(go.Bar(
                x=list(sc["Segment"]),
                y=list(sc["Count"]),
                marker=dict(color=[seg_colors[s] for s in sc["Segment"]], line=dict(width=0)),
            ))
            fig2.update_layout(plot_layout(height=250))
            st.plotly_chart(fig2, use_container_width=True)

        with col_b2:
            section("Risk Spread per Segment")
            fig3 = go.Figure()
            for seg in segments_list:
                vals = df_active[df_active["Segment"]==seg]["Churn_Probability"] * 100
                if not vals.empty:
                    fig3.add_trace(go.Box(
                        y=vals, name=seg,
                        marker_color=seg_colors[seg],
                        line_color=seg_colors[seg],
                        fillcolor=f'rgba({int(seg_colors[seg][1:3],16)},{int(seg_colors[seg][3:5],16)},{int(seg_colors[seg][5:7],16)},0.13)',
                    ))
            fig3.update_layout(plot_layout(height=250, showlegend=False))
            st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — ALERTS & ACTIONS
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# PAGE 5 — ALERTS & ACTIONS (DYNAMIQUE)
# ════════════════════════════════════════════════════════════
elif page == "Alerts & Actions":
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">// OPERATIONS · REAL-TIME</div>
        <h1 class="page-title">Alerts <span>&</span> Actions</h1>
        <p class="page-desc">Critical churn signals · Intervention queue · Retention playbooks</p>
    </div>""", unsafe_allow_html=True)

    # 1. RÉCUPÉRATION DES DONNÉES DU BATCH (S'IL EXISTE)
    # On suppose que vous avez stocké 'df_out' dans st.session_state lors du Batch Prediction
    if 'df_out' not in st.session_state:
        st.warning("⚠️ Aucune donnée disponible. Veuillez d'abord traiter un fichier dans 'Batch Prediction'.")
        # Données de secours vides pour éviter le crash
        alerts = []
    else:
        df_final = st.session_state.df_out
        
        # On prépare les alertes à partir du vrai fichier CSV
        alerts = []
        for idx, row in df_final.iterrows():
            # On cherche un nom de colonne pour l'identité (ex: 'State' ou 'ID' ou index)
            # Si vous n'avez pas de colonne "Name", on utilise l'index ou une colonne existante
            customer_name = f"Customer {idx}"
            if 'State' in row: customer_name = f"User_{row['State']}_{idx}"
            
            score = row['Churn_Probability']
            
            # On ne garde que les risques significatifs (> 50%)
            if score > 0.5:
                level = "critical" if score > 0.80 else "high" if score > 0.65 else "medium"
                
                # Simulation de données complémentaires car elles ne sont pas tjs dans le CSV
                alerts.append({
                    "name": customer_name,
                    "score": score,
                    "level": level,
                    "segment": "At Risk" if score < 0.8 else "Lost",
                    "days": random.randint(1, 15), # Simulation activité
                    "value": int(row.get('Total day minutes', 0) * 10) # Valeur estimée par l'usage
                })

    # Tri par score décroissant
    alerts.sort(key=lambda x: x["score"], reverse=True)

    # 2. CALCUL DES KPI RÉELS
    crit = sum(1 for a in alerts if a["level"]=="critical")
    high = sum(1 for a in alerts if a["level"]=="high")
    med  = sum(1 for a in alerts if a["level"]=="medium")
    rev  = sum(a["value"] for a in alerts if a["level"]=="critical")

    st.markdown(f"""
    <div class="kpi-grid-4">
        <div class="kpi-card"><div class="kpi-label">Critical Alerts</div><div class="kpi-value danger">{crit}</div></div>
        <div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value warning">{high}</div></div>
        <div class="kpi-card"><div class="kpi-label">Medium Risk</div><div class="kpi-value">{med}</div></div>
        <div class="kpi-card"><div class="kpi-label">Revenue at Risk</div><div class="kpi-value danger">{rev:,} TND</div></div>
    </div>""", unsafe_allow_html=True)

    # 3. FILTRAGE
    col_f1, _ = st.columns([1, 3])
    with col_f1:
        filter_level = st.selectbox("Filter by level", ["All","Critical","High","Medium"])
    
    filtered = alerts if filter_level=="All" else [a for a in alerts if a["level"]==filter_level.lower()]

    # 4. AFFICHAGE DES CARTES D'INTERVENTION
    section("Intervention Queue")
    actions_map = {
        "critical": ["Retention call","Promo offer","Executive escalation"],
        "high":     ["Email campaign","Discount voucher","Check-in call"],
        "medium":   ["Newsletter","Product tip","Survey"],
    }

    if not filtered:
        st.info("No alerts found for this level.")
    else:
        for a in filtered:
            # Génération d'initiales propres
            parts = a["name"].split('_') if '_' in a["name"] else a["name"].split()
            initials = "".join([w[0].upper() for w in parts[:2]])
            
            actions_html = "".join([f'<span class="action-pill">▸ {act}</span>' for act in actions_map[a["level"]]])
            
            st.markdown(f"""
            <div class="alert-card {a['level']}">
                <div class="alert-avatar">{initials}</div>
                <div class="alert-info">
                    <div class="alert-name">{a['name']}</div>
                    <div class="alert-meta">{a['segment']} · Last active {a['days']}d ago · {a['value']:,} TND</div>
                    <div style="margin-top:6px;">{actions_html}</div>
                </div>
                <div class="alert-score">
                    <div class="alert-score-val {a['level']}">{a['score']*100:.1f}%</div>
                    <div class="alert-score-label">CHURN RISK</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # 5. TIMELINE (ACTIVITÉ PLATEFORME)
    section("Recent Platform Activity")
    now = datetime.now()
    # On rend la timeline un peu plus dynamique par rapport au dernier batch
    last_batch_time = "Just now" if 'df_out' in st.session_state else "No activity"
    
    events = [
        (now-timedelta(minutes=2),  "cyan",   "Batch scoring updated",    f"{len(alerts)} risks detected from latest CSV", "BATCH"),
        (now-timedelta(hours=1),    "purple", "Model sync",               "Random Forest Engine v2.4.1",                 "MODEL"),
    ]
    
    tag_colors = {"BATCH":"#00e5ff","ALERT":"#ff5252","MODEL":"#7c3aed","ACTION":"#69f0ae","SUCCESS":"#69f0ae"}

    for ts, dot, title, detail, tag in events:
        tc = tag_colors.get(tag, "#00e5ff")
        r,g,b = int(tc[1:3],16),int(tc[3:5],16),int(tc[5:7],16)
        st.markdown(f"""
        <div class="tl-item">
            <div class="tl-dot {dot}"></div>
            <div class="tl-time">{ts.strftime('%H:%M')}</div>
            <div class="tl-text"><strong style="color:#c8d6e8;">{title}</strong><br>{detail}</div>
            <span class="tl-tag" style="background:rgba({r},{g},{b},0.13);color:{tc};border:1px solid rgba({r},{g},{b},0.27);">{tag}</span>
        </div>""", unsafe_allow_html=True)
