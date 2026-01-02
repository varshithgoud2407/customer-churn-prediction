import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor (5 Features)", page_icon="📉")

API_URL = "http://127.0.0.1:8000/predict"

st.title("📉 Customer Churn Prediction (5 Features)")
st.caption("Model trained only on 5 key business features for a clean and realistic UI.")

threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

tenure = st.number_input("Tenure (months)", 0, 100, 12)
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 89.1)
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

payload = {
    "tenure": int(tenure),
    "Contract": Contract,
    "MonthlyCharges": float(MonthlyCharges),
    "InternetService": InternetService,
    "PaymentMethod": PaymentMethod,
}

if st.button("Predict Churn"):
    try:
        r = requests.post(API_URL, params={"threshold": threshold}, json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()

        st.metric("Churn probability", res["churn_probability"])
        st.metric("Prediction (1=Churn)", res["churn_prediction"])
        st.progress(min(max(res["churn_probability"], 0.0), 1.0))

    except Exception as e:
        st.error(f"API error: {e}")
        st.info("Make sure FastAPI is running at http://127.0.0.1:8000")
