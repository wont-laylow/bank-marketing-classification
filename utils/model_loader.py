import pickle
import pandas as pd
from pathlib import Path

def load_files():
    # load the saved model files
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "model" / "random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    scaler_path = BASE_DIR / "model" / "scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load dummy columns
    dummy_col_path = BASE_DIR / "model" / "dummy_columns.pkl"
    with open(dummy_col_path, 'rb') as f:
        dummy_columns = pickle.load(f)

    return model, scaler, dummy_columns


def predict_subscription(sample_input):
    
    model, scaler, dummy_columns = load_files()

    # Convert to DataFrame
    input_df = pd.DataFrame([sample_input])

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df, columns=[
        'job', 'marital', 'education', 'default',
        'contact', 'month', 'day_of_week', 'poutcome'
    ])

    # Add missing dummy columns
    for col in dummy_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[dummy_columns]

    # Scale numerical columns
    numerical = ['duration', 'nr.employed', 'euribor3m', 'emp.var.rate', 'previous']
    input_encoded[numerical] = scaler.transform(input_encoded[numerical])

    # Predict
    pred = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    # Output
    prediction_text = "Subscribed (1)" if pred == 1 else "Not Subscribed (0)"
    print("Prediction:", prediction_text)
    print("Probability of subscribing:", round(proba, 4))

    return pred, proba
