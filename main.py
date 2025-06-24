import streamlit as st
from utils.model_loader import predict_subscription

st.title("Bank Subscription Predictor")

# Define user inputs
job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur',
 'student'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course',
 'unknown', 'university.degree', 'illiterate'])
default = st.selectbox("Default", ['yes', 'no', 'unknown' ])
contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep']
)
day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'success', 'failure'])

duration = st.number_input("Call Duration (seconds)", min_value=0, value=820)
nr_employed = st.number_input("Number Employed", value=5099.1)
euribor3m = st.slider(
    "Euribor 3 Month Rate",
    min_value=0.000,
    max_value=5.000,
    value=4.857,
    step=0.001
)
emp_var_rate = st.number_input("Employment Variation Rate", value=4991.6)
previous = st.number_input("Number of Previous Contacts [bool]", min_value=0, value=0)

# Construct input
sample_input = {
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'poutcome': poutcome,
    'duration': duration,
    'nr.employed': nr_employed,
    'euribor3m': euribor3m,
    'emp.var.rate': emp_var_rate,
    'previous': previous
}

if st.button("Predict Subscription"):
    with st.spinner("Predicting..."):
        try:
            prediction, probability = predict_subscription(sample_input=sample_input)

            label = "Yes" if prediction == 1 else "No"
            st.success(f"🎯 Will the client subscribe? **{label}**")
            # st.info(f"🧠 Model Confidence: **{round(float(probability) * 100, 2)}%**")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")