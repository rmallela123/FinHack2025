import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(uploaded_file):
    """Loads data from an uploaded CSV file into a pandas DataFrame."""
    return pd.read_csv(uploaded_file)

def preprocess_data(df):
    """
    Preprocesses the raw claims data by cleaning, encoding, and feature engineering.
    """
    # Make a copy to avoid modifying the original dataframe
    hcdata = df.copy()

    # Store original diagnosis codes for frequency calculation
    diagnosis_codes = hcdata['DiagnosisCode'].copy()

    # Encoding gender
    hcdata['PatientGender'] = hcdata['PatientGender'].map({'M': 0, 'F': 1})

    # One-hot encode categorical variables
    categorical_cols = ['PatientMaritalStatus', 'PatientEmploymentStatus', 'ClaimType', 'ProviderSpecialty']
    hcdata = pd.get_dummies(hcdata, columns=categorical_cols)

    # Drop irrelevant columns
    cols_to_drop = ['ClaimID', 'PatientID', 'ProviderID', 'ClaimDate',
                    'ClaimSubmissionMethod', 'ProviderLocation', 'DiagnosisCode',
                    'ProcedureCode', 'ClaimStatus']
    hcdata_cleaned = hcdata.drop(columns=cols_to_drop)

    # Create frequency feature for diagnosis codes
    diag_freq = diagnosis_codes.value_counts().to_dict()
    hcdata_cleaned['DiagnosisFreq'] = diagnosis_codes.map(diag_freq)

    # Handle potential missing columns in dummy/user data after one-hot encoding
    # (This is important for robustness when user uploads their own data)
    # The model will be trained on a specific set of columns, so new data must match.
    # We will handle this alignment in the main app script before prediction.

    return hcdata_cleaned

def train_and_evaluate_model(df):
    """
    Trains a RandomForestRegressor model and evaluates its performance.
    Returns the model, mse, and r2 score.
    """
    features = df.drop(columns=['ClaimAmount'])
    target = df['ClaimAmount']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, features.columns

def create_feature_importance_plot(model, feature_names):
    """
    Generates a bar plot of feature importances from the trained model.
    """
    importances = model.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Key Drivers of Healthcare Cost')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    return plt

def create_claim_amount_distribution_plot(df):
    """
    Generates a histogram for the 'ClaimAmount' distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['ClaimAmount'], bins=50, edgecolor='k')
    plt.title('Distribution of Claim Amounts')
    plt.xlabel('Claim Amount')
    plt.ylabel('Frequency')

    # Add interpretation text
    textstr = '\n'.join((
        r'$\bf{Interpretation\ Guide:}$',
        r'- $\bf{Right-skewed}$: Typical for real-world claims (many small, few large).',
        r'- $\bf{Uniform}$: Suggests synthetic or artificial data.',
        r'- $\bf{Bell-shaped}$: Also suggests artificial data; real costs are rarely normal.'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.1, -0.1, textstr, fontsize=10, bbox=props, transform=plt.gca().transAxes)
    plt.subplots_adjust(bottom=0.3)

    return plt
