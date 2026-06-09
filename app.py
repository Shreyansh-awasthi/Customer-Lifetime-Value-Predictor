import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #070b12;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    max-width: 1280px;
}

.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.1rem 1.4rem;
    border: 1px solid #1e293b;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.85);
    margin-bottom: 1.5rem;
}

.logo {
    font-size: 1.05rem;
    font-weight: 800;
    color: #f8fafc;
}

.model-tag {
    font-size: 0.78rem;
    color: #94a3b8;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.hero-section {
    padding: 3rem 2.4rem;
    border-radius: 30px;
    border: 1px solid #1e293b;
    background:
        radial-gradient(circle at top left, rgba(234,179,8,0.20), transparent 30%),
        linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
    margin-bottom: 1.5rem;
}

.hero-eyebrow {
    color: #eab308;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 4rem;
    line-height: 1.02;
    font-weight: 800;
    color: #f8fafc;
}

.hero-title span {
    color: #eab308;
}

.hero-desc {
    color: #cbd5e1;
    font-size: 1.05rem;
    line-height: 1.8;
    max-width: 760px;
    margin-top: 1.2rem;
}

.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.7rem;
}

.stat-card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid #1e293b;
    border-radius: 22px;
    padding: 1.4rem;
}

.stat-num {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
}

.accent {
    color: #eab308;
}

.stat-desc {
    margin-top: 0.35rem;
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.panel-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}

.panel-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-bottom: 1.4rem;
}

.result-box {
    border-radius: 26px;
    border: 1px solid #334155;
    padding: 2rem;
    background:
        radial-gradient(circle at top right, rgba(234,179,8,0.22), transparent 32%),
        linear-gradient(135deg, #111827, #020617);
}

.result-label {
    color: #94a3b8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.clv-value {
    font-size: 3.4rem;
    font-weight: 800;
    color: #eab308;
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
}

.segment {
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.segment-vip {
    color: #f5a623;
}

.segment-high {
    color: #00b894;
}

.segment-medium {
    color: #6c5ce7;
}

.segment-low {
    color: #636e72;
}

.action-text {
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 0.95rem;
}

.waiting-box {
    height: 500px;
    border-radius: 26px;
    border: 1px dashed #334155;
    background: #0f172a;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #64748b;
}

.stButton > button {
    background: #eab308;
    color: #020617;
    font-weight: 800;
    border: none;
    border-radius: 14px;
    height: 3rem;
}

.stButton > button:hover {
    background: #facc15;
    color: #020617;
}

@media (max-width: 900px) {
    .stats-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title {
        font-size: 2.8rem;
    }

    .top-bar {
        flex-direction: column;
        gap: 0.5rem;
        align-items: flex-start;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load("clv_model.pkl"), True
    except Exception:
        return None, False


model, model_loaded = load_model()


def predict_clv(frequency, recency, T, monetary):
    input_data = np.array([[frequency, recency, T, monetary]])
    log_clv = model.predict(input_data)[0]
    predicted_clv = np.expm1(log_clv)
    return float(predicted_clv)


def get_segment(predicted_clv):
    if predicted_clv > 500:
        return (
            "VIP Customer",
            "segment-vip",
            "Offer exclusive loyalty rewards, priority service, and premium retention benefits."
        )
    elif predicted_clv > 200:
        return (
            "High Value",
            "segment-high",
            "Nurture with personalized offers. This customer has strong revenue potential."
        )
    elif predicted_clv > 100:
        return (
            "Medium Value",
            "segment-medium",
            "Send targeted promotions to increase purchase frequency and customer value."
        )
    else:
        return (
            "Low Value",
            "segment-low",
            "Use re-engagement campaigns with limited discounts. Do not overspend on this customer."
        )


st.markdown("""
<div class="top-bar">
    <div class="logo">Customer Lifetime Value Predictor</div>
    <div class="model-tag">BG/NBD · Gamma-Gamma · XGBoost</div>
</div>

<div class="hero-section">
    <div class="hero-eyebrow">Predictive Analytics</div>
    <div class="hero-title">Customer <span>Lifetime</span><br>Value</div>
    <div class="hero-desc">
        Predict the future revenue potential of a customer using Frequency, Recency,
        Customer Age and Monetary Value.
    </div>
</div>

<div class="stats-row">
    <div class="stat-card">
        <div class="stat-num accent">0.94</div>
        <div class="stat-desc">Model R² Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">1M+</div>
        <div class="stat-desc">Transactions</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">12M</div>
        <div class="stat-desc">Forecast Horizon</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">RFM</div>
        <div class="stat-desc">Feature Framework</div>
    </div>
</div>
""", unsafe_allow_html=True)


left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown("""
    <div class="panel-title">Customer Data</div>
    <div class="panel-subtitle">Enter customer purchase behavior values.</div>
    """, unsafe_allow_html=True)

    monetary = st.number_input(
        "Monetary Value",
        min_value=0.0,
        max_value=10000.0,
        value=250.0,
        step=10.0
    )

    T = st.number_input(
        "Customer Age / T",
        min_value=1.0,
        max_value=3000.0,
        value=365.0,
        step=1.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        max_value=500.0,
        value=8.0,
        step=1.0
    )

    recency = st.number_input(
        "Recency",
        min_value=0.0,
        max_value=3000.0,
        value=120.0,
        step=1.0
    )

    predict_btn = st.button("Predict Customer Value", use_container_width=True)


with right:
    st.markdown("""
    <div class="panel-title">Prediction Output</div>
    <div class="panel-subtitle">Customer value, segment, and business recommendation.</div>
    """, unsafe_allow_html=True)

    if predict_btn:
        if not model_loaded:
            st.error("Model file not found. Make sure xgboost_clv_model.pkl is in the same folder.")
        elif recency > T:
            st.error("Recency cannot be greater than T. Last purchase cannot be before first purchase.")
        else:
            predicted_clv = predict_clv(frequency, recency, T, monetary)
            segment, segment_class, recommendation = get_segment(predicted_clv)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Customer Lifetime Value</div>
                <div class="clv-value">€{predicted_clv:,.2f}</div>
                <div class="segment {segment_class}">{segment}</div>
                <div class="action-text">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            fig1 = go.Figure()

            fig1.add_trace(go.Bar(
                x=["Low", "Medium", "High", "VIP", "Customer"],
                y=[100, 200, 500, 700, predicted_clv],
                text=[
                    "€100",
                    "€200",
                    "€500",
                    "€700+",
                    f"€{predicted_clv:,.0f}"
                ],
                textposition="outside",
                marker=dict(
                    color=["#636e72", "#6c5ce7", "#00b894", "#f5a623", "#eab308"]
                )
            ))

            fig1.update_layout(
                title="Customer Position Against CLV Segments",
                xaxis_title="CLV Segment",
                yaxis_title="Customer Lifetime Value",
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=70, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )

            fig1.add_hline(
                y=100,
                line_dash="dot",
                line_color="#6c5ce7",
                annotation_text="Medium starts"
            )

            fig1.add_hline(
                y=200,
                line_dash="dot",
                line_color="#00b894",
                annotation_text="High starts"
            )

            fig1.add_hline(
                y=500,
                line_dash="dot",
                line_color="#f5a623",
                annotation_text="VIP starts"
            )

            st.plotly_chart(fig1, use_container_width=True)

            input_df = pd.DataFrame({
                "Frequency": [frequency],
                "Recency": [recency],
                "T": [T],
                "Monetary": [monetary],
                "Predicted_CLV": [predicted_clv],
                "Segment": [segment]
            })

            with st.expander("View model input"):
                st.dataframe(input_df, use_container_width=True)

    else:
        st.markdown("""
        <div class="waiting-box">
            <div>
                <h2>Awaiting Input</h2>
                <p>Enter customer data and run prediction.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
