import streamlit as st
import pandas as pd
import joblib

# 1. Page Config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# 2. Load Model and Columns
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model_final.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_artifacts()

# 3. Title
st.title("📱 Customer Churn Prediction")
st.markdown("Predict if a customer will leave the telecom company.")

# 4. Sidebar Inputs
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
payment = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['Yes', 'No'])
phone = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
paperless = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])

# 5. Create the raw input data (matching the original CSV structure)
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone],
    'MultipleLines': ['No'],
    'InternetService': [internet],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': [contract],
    'PaperlessBilling': [paperless],
    'PaymentMethod': [payment],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# 6. Apply One-Hot Encoding (Same as training)
input_encoded = pd.get_dummies(input_data, drop_first=True)

# 7. ALIGN COLUMNS: This is the magic step
# We add any missing columns (with value 0) and remove extra ones
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# 8. Predict
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk!** This customer is likely to CHURN. (Probability: {probability*100:.1f}%)")
        st.markdown("**Recommended Action:** Send a retention offer with a 20% discount.")
    else:
        st.success(f"✅ **Safe.** This customer is likely to STAY. (Probability of Churn: {probability*100:.1f}%)")
        st.markdown("**Recommended Action:** Keep engaging with loyalty rewards.")