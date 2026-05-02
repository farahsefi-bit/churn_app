import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model & features ────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("saved_model/best_rf_model.pkl")
    features = joblib.load("saved_model/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ── Sidebar navigation ───────────────────────────────────────
st.sidebar.title("📉 Churn Predictor")
page = st.sidebar.radio("Navigate", [
    "🔍 Single Prediction",
    "📂 Batch Prediction (CSV)",
    "📊 Model Insights"
])

# ════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":

    st.title("🔍 Single Customer Churn Prediction")
    st.markdown("Fill in the customer details below to get an instant churn probability.")

    with st.form("prediction_form"):
        st.subheader("Customer Features")
        cols = st.columns(3)
        user_input = {}

        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                user_input[feat] = st.number_input(
                    label=feat,
                    value=0.0,
                    format="%.4f"
                )

        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_input])

        proba = model.predict_proba(input_df)[0][1]
        pred  = model.predict(input_df)[0]

        col1, col2 = st.columns(2)

        with col1:
            color = "#e74c3c" if pred == 1 else "#2ecc71"
            label = "⚠️ CHURN RISK" if pred == 1 else "✅ LIKELY TO STAY"
            st.markdown(f"""
            <div style='background:{color};padding:20px;border-radius:10px;
            text-align:center;color:white;font-size:22px;font-weight:bold;'>
                {label}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if proba > 0.5 else "#2ecc71"},
                    'steps': [
                        {'range': [0,  40], 'color': "#d5f5e3"},
                        {'range': [40, 70], 'color': "#fef9e7"},
                        {'range': [70,100], 'color': "#fadbd8"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction (CSV)":

    st.title("📂 Batch Prediction from CSV")
    st.markdown("Upload a CSV file with the same columns as your training data (without the `Churn` column).")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"**Rows loaded:** {len(df_batch)}")
        st.dataframe(df_batch.head(), use_container_width=True)

        missing = set(feature_names) - set(df_batch.columns)
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            df_input = df_batch[feature_names]

            probas = model.predict_proba(df_input)[:, 1]
            preds  = model.predict(df_input)

            df_batch["Churn_Probability"] = probas.round(4)
            df_batch["Prediction"]        = preds
            df_batch["Risk_Label"]        = df_batch["Prediction"].map({0: "Stay", 1: "Churn"})

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers",    len(df_batch))
            c2.metric("Predicted Churners", int(preds.sum()))
            c3.metric("Churn Rate",         f"{preds.mean()*100:.1f}%")

            fig = px.histogram(
                df_batch, x="Churn_Probability", color="Risk_Label",
                nbins=30, title="Churn Probability Distribution",
                color_discrete_map={"Stay": "#2ecc71", "Churn": "#e74c3c"}
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df.tail(15),
        x="Importance", y="Feature",
        orientation="h",
        title="Top 15 Feature Importances (Random Forest)",
        color="Importance",
        color_continuous_scale="RdYlGn"
    )
    fig.update_layout(yaxis_title="", xaxis_title="Importance Score", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌲 Best Model Parameters")
    params = model.get_params()
    params_df = pd.DataFrame(
        {"Parameter": list(params.keys()), "Value": list(params.values())}
    )
    st.dataframe(params_df, use_container_width=True)