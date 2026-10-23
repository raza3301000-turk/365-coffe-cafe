import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# App Title & Header
st.set_page_config(page_title="365 Coffee Café Retention Predictor", layout="centered")
st.title("☕ 365 Coffee Café - Customer Return Predictor")
st.write("Predict whether a customer visit will result in a repeat visit within 14 days.")

@st.cache_resource
def train_model():
    # Load visit-level dataset
    visit_df = pd.read_csv("365_Coffee_Visit_Level.csv")
    
    # Target encoding
    visit_df['target'] = visit_df['returned_within_14_days'].apply(
        lambda x: 1 if str(x).strip().lower() in ['yes', 'true', '1'] else 0
    )
    
    drop_cols = ['basket_id', 'transaction_date', 'transaction_time', 'customer_id', 'returned_within_14_days', 'target']
    X = visit_df.drop(columns=[col for col in drop_cols if col in visit_df.columns])
    X = pd.get_dummies(X, drop_first=True)
    y = visit_df['target']
    
    # Train Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    return rf, X.columns

model, feature_columns = train_model()

# Sidebar / User Inputs
st.sidebar.header("Customer & Visit Details")

visit_number = st.sidebar.number_input("Visit Number", min_value=1, max_value=50, value=2)
days_since_prev = st.sidebar.number_input("Days Since Previous Visit", min_value=-1, max_value=100, value=7)
wait_time = st.sidebar.slider("Wait Time (Minutes)", min_value=1.0, max_value=20.0, value=5.0)
net_revenue = st.sidebar.number_input("Net Revenue (PKR)", min_value=1.0, max_value=5000.0, value=850.0)
customer_age = st.sidebar.slider("Customer Age", min_value=18, max_value=80, value=30)
service_rating = st.sidebar.selectbox("Service Rating", options=[1, 2, 3, 4, 5], index=3)
loyalty_tier = st.sidebar.selectbox("Loyalty Tier", options=["Basic", "Bronze", "Silver", "Gold", "Platinum"])

# Prediction Button
if st.button("Predict Return Likelihood"):
    # Create input row matching training feature columns
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Assign numerical values
    if 'visit_number' in input_data.columns: input_data['visit_number'] = visit_number
    if 'days_since_previous_visit' in input_data.columns: input_data['days_since_previous_visit'] = days_since_prev
    if 'wait_time_min' in input_data.columns: input_data['wait_time_min'] = wait_time
    if 'net_revenue' in input_data.columns: input_data['net_revenue'] = net_revenue
    if 'customer_age' in input_data.columns: input_data['customer_age'] = customer_age
    if 'service_rating' in input_data.columns: input_data['service_rating'] = service_rating
    
    # One-hot encoded feature column for loyalty tier
    loyalty_col = f"loyalty_tier_{loyalty_tier}"
    if loyalty_col in input_data.columns:
        input_data[loyalty_col] = 1

    # Get Prediction
    prob = model.predict_proba(input_data)[0][1] * 100
    pred = model.predict(input_data)[0]

    st.markdown("---")
    if pred == 1:
        st.success(f"**Prediction: Likely to Return within 14 days** 🎉")
        st.metric(label="Return Probability", value=f"{prob:.1f}%")
    else:
        st.warning(f"**Prediction: Unlikely to Return within 14 days** ⚠️")
        st.metric(label="Return Probability", value=f"{prob:.1f}%")
        st.info("💡 *Note: This estimate serves as a decision-support metric based on historical patterns.*")