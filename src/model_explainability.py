import shap
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier  # Add this line for XGBClassifier
from sklearn.ensemble import RandomForestClassifier  # Add this line for RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# File paths
MODEL_DIR = 'models/'
DATA_PATH = 'data/preprocessed_data.csv'

# Load the preprocessed data and the trained models
def load_data(path):
    return pd.read_csv(path)

def load_model(model_name):
    return joblib.load(f"{MODEL_DIR}{model_name.lower()}_model.pkl")

def explain_model_shap(model, X_train, X_test):
    # Create a SHAP explainer for the model
    if isinstance(model, XGBClassifier) or isinstance(model, RandomForestClassifier):  # Tree-based models
        explainer = shap.TreeExplainer(model)
    else:  # For other models like LogisticRegression
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        
    # Get SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    # Visualize the SHAP summary plot (global explanation)
    shap.summary_plot(shap_values, X_test)
    
    # Visualize the SHAP force plot for the first test sample (local explanation)
    shap.initjs()  # Initialize JS for the plot
    shap.force_plot(shap_values[0], X_test.iloc[0], X_test.columns)

def explain_model_lime(model, X_train, X_test, y_train):
    # Create a LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values, 
        feature_names=X_train.columns, 
        class_names=["Not Churn", "Churn"], 
        mode="classification"
    )

    # Choose a test sample to explain
    idx = 0  # Change this to any index to explain other samples
    explanation = explainer.explain_instance(X_test.iloc[idx].values, model.predict_proba)

    # Visualize the explanation (local explanation)
    explanation.show_in_notebook()

def main():
    # Load the data and split into X_train, X_test
    df = load_data(DATA_PATH)
    X_train = df.drop('Churn', axis=1)
    y_train = df['Churn']
    X_test = X_train.sample(frac=0.2, random_state=42)
    y_test = y_train.sample(frac=0.2, random_state=42)
    
    # Load trained models
    logreg = load_model("LogReg")
    rf = load_model("RF")
    xgb = load_model("XGB")

    # Explain Logistic Regression using SHAP
    print("Explaining Logistic Regression Model with SHAP...")
    explain_model_shap(logreg, X_train, X_test)
    
    # Explain Random Forest using SHAP
    print("Explaining Random Forest Model with SHAP...")
    explain_model_shap(rf, X_train, X_test)

    # Explain XGBoost using SHAP
    print("Explaining XGBoost Model with SHAP...")
    explain_model_shap(xgb, X_train, X_test)
    
    # Explain Logistic Regression using LIME
    print("Explaining Logistic Regression Model with LIME...")
    explain_model_lime(logreg, X_train, X_test, y_train)

    # Explain Random Forest using LIME
    print("Explaining Random Forest Model with LIME...")
    explain_model_lime(rf, X_train, X_test, y_train)

    # Explain XGBoost using LIME
    print("Explaining XGBoost Model with LIME...")
    explain_model_lime(xgb, X_train, X_test, y_train)

if __name__ == "__main__":
    main()
