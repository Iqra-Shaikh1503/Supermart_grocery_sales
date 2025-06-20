import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Load everything =====
model = joblib.load("src/Stacked_sales_profit_model.pkl")
scaler = joblib.load("src/scaler.pkl")
feature_names = joblib.load("src/model_input_features.pkl")
encoders = joblib.load("src/label_encoders.pkl")

# ===== Page Setup =====
st.set_page_config(page_title="🛍️ Sales & Profit Predictor", layout="centered")
st.title("📦 Supermart Sales & Profit Predictor")
st.markdown("Predict **Sales** and **Profit** using basic product and order details.")

# ===== Input Form =====
with st.form("prediction_form"):
    st.subheader("📥 Enter Input Details")
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", encoders['Category'].classes_)
        sub_category = st.selectbox("Sub Category", encoders['Sub Category'].classes_)
        city = st.selectbox("City", encoders['City'].classes_)
        region = st.selectbox("Region", encoders['Region'].classes_)

    with col2:
        order_month = st.slider("Order Month", 1, 12, 6)
        order_year = st.selectbox("Order Year", [2015, 2016, 2017, 2018])
        order_quarter = st.selectbox("Order Quarter", [1, 2, 3, 4])
        revenue_per_discount = st.number_input("Revenue per Discount", min_value=0.0)

    submitted = st.form_submit_button("📈 Predict")

# ===== Run Prediction =====
if submitted:
    # Encode selected options using training encoders
    input_row = pd.DataFrame([{
        'Category': encoders['Category'].transform([category])[0],
        'Sub Category': encoders['Sub Category'].transform([sub_category])[0],
        'City': encoders['City'].transform([city])[0],
        'Region': encoders['Region'].transform([region])[0],
        'Order_Month': order_month,
        'Order_Year': order_year,
        'Order_Quarter': order_quarter,
        'Revenue_per_Discount': revenue_per_discount
    }])

    # Fill in missing columns from training with 0
    full_input = pd.DataFrame(columns=feature_names)
    full_input.loc[0] = 0  # initialize all as 0
    full_input.update(input_row)

    # Scale input
    input_scaled = scaler.transform(full_input)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    st.success("✅ Prediction Complete!")
    st.metric("🛒 Predicted Sales", f"₹ {prediction[0][0]:,.2f}")
    st.metric("💰 Predicted Profit", f"₹ {prediction[0][1]:,.2f}")
