import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the input features in the same order as during training
input_features = [
    'feature_1', 'feature_2', 'feature_3',  # replace with real column names
    'feature_4', 'feature_5', 'feature_6',
    # add all 44 features here in the correct order
]

# Title
st.title("Binary Classification with Random Forest")
st.markdown("Enter the values below and get the model's prediction.")

# Collect user input
user_input = []
for feature in input_features:
    value = st.number_input(f"Enter value for {feature}", step=0.01)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    # Convert input to DataFrame for model
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0]

    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The model predicts: **Class 1** with probability {prediction_proba[1]:.2f}")
    else:
        st.info(f"The model predicts: **Class 0** with probability {prediction_proba[0]:.2f}")
