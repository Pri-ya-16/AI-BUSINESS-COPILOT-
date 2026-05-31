import streamlit as st
import pandas as pd
import plotly.express as px
import time

from utils import (
    calculate_metrics,
    predict_future_revenue,
    genai_reasoning,
    what_if_analysis,
    autonomous_alert,
)

st.set_page_config(
    page_title="GenAI Business Decision Copilot",
    layout="wide"
)

st.title("🤖 GenAI Business Decision Copilot")
st.caption("Real-Time AI-Driven Analytics & Decision Intelligence")

uploaded_file = st.file_uploader(
    "📂 Upload Business Data (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=",")
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("📊 Data Preview")
    st.dataframe(df)

    metrics = calculate_metrics(df)

    st.subheader("📌 Key Business Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Revenue Change (%)", metrics["revenue_change"])
    col2.metric("Repeat Customer Change (%)", metrics["repeat_change"])

    st.subheader("📈 Revenue Trend")
    fig = px.line(df, x="date", y="revenue", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Revenue Forecast")
    future = predict_future_revenue(df)
    st.line_chart(future)

    st.subheader("🧠 Autonomous Business Alert")
    st.warning(autonomous_alert(metrics))

    st.subheader("💬 Ask the AI Business Analyst")
    question = st.text_input(
        "Example: Why did revenue drop last week?"
    )

    if question:
        st.info(genai_reasoning(question, metrics))

    st.subheader("🧪 What-If Scenario Simulator")
    discount = st.slider(
        "Discount Percentage (%)",
        5, 30, 10
    )
    st.success(what_if_analysis(discount, metrics))

    st.caption("🔄 Real-time monitoring active")
    time.sleep(1)
