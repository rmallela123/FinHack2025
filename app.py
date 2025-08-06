import streamlit as st
import pandas as pd
from analysis import (
    load_data,
    preprocess_data,
    train_and_evaluate_model,
    create_feature_importance_plot,
    create_claim_amount_distribution_plot
)

# --- App Configuration ---
st.set_page_config(
    page_title="Healthcare Claims Analysis",
    page_icon="🏥",
    layout="wide"
)

# --- App Title and Description ---
st.title("🏥 Healthcare Claims Analysis & Data Quality Assessment")
st.markdown("""
This tool helps analyze healthcare claims data to identify key cost drivers and assess data quality.
The original analysis discovered that the initial dataset was likely **synthetic**, based on the uniform
distribution of claim amounts. Real-world claims data typically shows a **right-skewed distribution**
(many small claims, a few very large ones).

**Instructions:**
1. Upload your own healthcare claims CSV file.
2. The tool will automatically preprocess the data, train a model, and display the results.
3. Pay close attention to the **"Distribution of Claim Amounts"** chart to assess if your data appears realistic.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your healthcare claims CSV file", type="csv")

# --- Analysis Trigger ---
if uploaded_file is not None:
    st.info("File uploaded successfully! Running analysis...")

    # --- Data Loading and Analysis ---
    try:
        raw_data = load_data(uploaded_file)

        # Preprocess the data
        preprocessed_data = preprocess_data(raw_data.copy())

        # --- Handle Column Alignment for Robustness ---
        # Load dummy data to get the canonical column structure after dummification
        dummy_df_processed = preprocess_data(pd.read_csv('dummy_claims.csv'))

        # Align columns
        model_features = dummy_df_processed.drop(columns=['ClaimAmount']).columns
        current_features = preprocessed_data.drop(columns=['ClaimAmount']).columns

        # Add missing columns (features the model expects but are not in the new data)
        for col in model_features:
            if col not in preprocessed_data.columns:
                preprocessed_data[col] = 0

        # Drop extra columns (features in the new data that the model wasn't trained on)
        for col in current_features:
            if col not in model_features:
                preprocessed_data = preprocessed_data.drop(columns=[col])

        # Ensure the order of columns is the same
        preprocessed_data = preprocessed_data[['ClaimAmount'] + list(model_features)]

        # --- Model Training and Visualization ---
        st.header("Analysis Results")

        # Use a smaller subset if the data is too large, to keep the app responsive
        sample_size = min(len(preprocessed_data), 5000)
        analysis_df = preprocessed_data.sample(n=sample_size, random_state=42)

        # Train model and get metrics
        model, mse, r2, feature_names = train_and_evaluate_model(analysis_df)

        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model R-Squared (R²)", f"{r2:.3f}")
            st.caption("""
            Measures how well the model explains the variance in claim amounts.
            - **1** is a perfect fit.
            - **0** means the model does not perform better than random.
            - **Negative** values indicate the model is very poor, often a sign of data quality issues (like synthetic data).
            """)
        with col2:
            st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
            st.caption("The average of the squares of the prediction errors. Lower is better.")

        # Display Visualizations
        st.header("Visual Insights")

        col1_viz, col2_viz = st.columns(2)
        with col1_viz:
            st.subheader("Key Drivers of Cost")
            feature_plot = create_feature_importance_plot(model, feature_names)
            st.pyplot(feature_plot)

        with col2_viz:
            st.subheader("Distribution of Claim Amounts")
            distribution_plot = create_claim_amount_distribution_plot(analysis_df)
            st.pyplot(distribution_plot)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.warning("Please ensure your CSV file has the required columns: 'ClaimID', 'PatientID', 'ProviderID', 'ClaimAmount', 'ClaimDate', 'DiagnosisCode', 'ProcedureCode', 'PatientAge', 'PatientGender', 'ProviderSpecialty', 'ClaimStatus', 'PatientIncome', 'PatientMaritalStatus', 'PatientEmploymentStatus', 'ProviderLocation', 'ClaimType', 'ClaimSubmissionMethod'.")

else:
    st.info("Please upload a CSV file to begin the analysis.")
