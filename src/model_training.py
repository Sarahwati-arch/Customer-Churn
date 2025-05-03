import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# File paths
DATA_PATH = 'data/preprocessed_data.csv'
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Label encode the target variable (Churn) to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), label_encoder

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, label_encoder):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    save_model(model, os.path.join(MODEL_DIR, f"{model_name.lower()}_model.pkl"))

    # Decode predictions back to original labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    print("Decoded Predictions:", y_pred_decoded)

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    df = load_data(DATA_PATH)
    (X_train, X_test, y_train, y_test), label_encoder = split_data(df)

    # Logistic Regression with class_weight='balanced' to handle class imbalance
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    train_and_evaluate_model(logreg, "LogReg", X_train, X_test, y_train, y_test, label_encoder)

    # Random Forest with class_weight='balanced' to handle class imbalance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    train_and_evaluate_model(rf, "RF", X_train, X_test, y_train, y_test, label_encoder)

    # XGBoost with scale_pos_weight to handle class imbalance
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=10)
    train_and_evaluate_model(xgb, "XGB", X_train, X_test, y_train, y_test, label_encoder)

if __name__ == "__main__":
    main()
