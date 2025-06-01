import streamlit as st
import requests
import numpy as np
import json
import pandas as pd

# Configure the page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Credit Card Fraud Detection")
st.markdown("""
This application helps detect potentially fraudulent credit card transactions using machine learning.
The model analyzes transaction features and provides a fraud probability score.
""")

# API endpoint
API_ENDPOINT = "http://localhost:8000/predict"

def predict_fraud(features):
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"features": features},
            timeout=10  # Added a timeout for robustness
        )
        if response.ok: # Check if the request was successful (status code 2xx)
            return response.json()
        else:
            # Try to parse error message from API if available
            try:
                error_detail = response.json().get("detail", response.text)
            except json.JSONDecodeError: # If response is not JSON
                error_detail = response.text
            st.error(f"API Error (Status {response.status_code}): {error_detail}")
            return None  # Return None if API call failed or returned an error
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {str(e)}")
        return None

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Transaction Features")
    st.info("Enter the transaction features (V1-V30). These are PCA-transformed features of the original transaction data.")
    
    # Create input fields for features
    num_features = 30
    features = []
    
    # Create 3 rows of 10 columns each for better layout
    for i in range(0, num_features, 10):
        cols = st.columns(10)
        for j in range(10):
            if i + j < num_features:
                feature_index = i + j
                feature_val = cols[j].number_input(
                    f"V{feature_index + 1}",
                    value=0.0,
                    format="%.6f",
                    key=f"feature_v_{feature_index}"
                )
                features.append(feature_val)

    # Add a predict button
    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing transaction..."):
            result = predict_fraud(features)
            
            # More robust check for the result structure
            if result and isinstance(result, dict) and "fraud_probability" in result and "is_fraud" in result:
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Create a gauge-like visualization for fraud probability
                    fraud_prob = result["fraud_probability"]
                    is_fraud = result["is_fraud"]
                    
                    # Display probability with a progress bar
                    st.markdown("### Fraud Probability")
                    st.progress(fraud_prob)
                    st.markdown(f"### {fraud_prob:.2%}")
                    
                    # Display the decision
                    if is_fraud:
                        st.error("⚠️ High risk of fraud detected!")
                    else:
                        st.success("✅ Transaction appears legitimate")
                    
                    # Additional details
                    st.markdown("### Decision Details")
                    st.json({
                        "Fraud Probability": f"{fraud_prob:.4f}",
                        "Risk Level": "High" if is_fraud else "Low",
                        "Recommendation": "Investigate" if is_fraud else "Approve"
                    })
            elif result: # If result is not None but doesn't have the expected keys
                st.error("Received an unexpected response from the API. Check the backend logs for more details.")
                # st.json(result) # Optionally display the unexpected response for debugging
            # If result is None, predict_fraud has already displayed an error

# Add information about the features
with st.expander("ℹ️ About the Features"):
    st.markdown("""
    The features (V1-V30) are PCA transformations of the original transaction data. 
    This transformation is done to protect sensitive transaction information while 
    maintaining the mathematical properties needed for fraud detection.
    
    - Features V1-V28 are the principal components obtained from PCA
    - V29 is the transaction time
    - V30 is the transaction amount
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • MLflow • FastAPI") 