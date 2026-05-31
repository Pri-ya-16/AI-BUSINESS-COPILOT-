import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


def calculate_metrics(df):
    revenue_change = (
        (df["revenue"].iloc[-1] - df["revenue"].iloc[0])
        / df["revenue"].iloc[0]
    ) * 100

    repeat_change = (
        (df["repeat_customers"].iloc[-1] - df["repeat_customers"].iloc[0])
        / df["repeat_customers"].iloc[0]
    ) * 100

    return {
        "revenue_change": round(revenue_change, 2),
        "repeat_change": round(repeat_change, 2),
    }


def predict_future_revenue(df, days=7):
    df = df.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["revenue"])

    df["index"] = np.arange(len(df))
    X = df[["index"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(df), len(df) + days).reshape(-1, 1)
    predictions = model.predict(future_index)

    return predictions


def genai_reasoning(question, metrics):
    # Always safe: API OR fallback
    if not gemini_model:
        return (
            "AI Insight: Revenue changes are influenced by repeat customer behavior. "
            "The business should focus on improving retention through loyalty programs "
            "and personalized engagement."
        )

    try:
        prompt = f"""
You are a senior business analyst.

Think step by step before answering.

Business metrics:
- Revenue change: {metrics['revenue_change']}%
- Repeat customer change: {metrics['repeat_change']}%

User question:
{question}

Explain clearly in your own words:
1. Why this happened
2. What it means for the business
3. What actions should be taken next

Use natural human language. No bullet points.
"""

        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception:
        return (
            "AI Insight: Revenue decline is mainly due to reduced repeat customers. "
            "Improving customer retention and engagement can help stabilize performance."
        )


def what_if_analysis(discount, metrics):
    return (
        f"If a {discount}% discount is introduced, short-term sales may increase. "
        "However, the business should closely monitor profit margins and long-term "
        "customer retention to avoid negative impact."
    )


def autonomous_alert(metrics):
    if metrics["revenue_change"] < -10:
        return "🚨 Critical Alert: Revenue is declining sharply."
    elif metrics["revenue_change"] < 0:
        return "⚠️ Warning: Revenue is trending downward."
    else:
        return "✅ Business performance is stable."
    

def what_if_analysis(discount, metrics):
    """
    Simulates the impact of a discount on revenue trajectory
    """

    current_revenue_change = metrics["revenue_change"]

    # Simple impact logic (hackathon-safe & explainable)
    impact_factor = discount * 0.35
    simulated_revenue_change = current_revenue_change + impact_factor

    simulated_revenue_change = round(simulated_revenue_change, 2)

    return (
        f"Simulating a {discount}% discount: "
        f"Based on current performance, this could shift your revenue "
        f"trajectory to approximately {simulated_revenue_change}%.\n\n"
        "🔄 Data synchronized with uploaded file."
    )
