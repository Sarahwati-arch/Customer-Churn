import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# File paths
DATA_PATH = 'data/processed_data.csv'  # cleaned data
OUTPUT_DATA_PATH = 'data/preprocessed_data.csv'
SCALER_PATH = 'models/scaler.pkl'  # Save scaler as .pkl instead of .h5

os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def encode_categorical(df, cat_features):
    # encoding on categorical features
    label_encoders = {}  # saving encoder for each column

    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def scale_numerical(df, num_features):
    # scaling on numerical features
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    return df, scaler

def save_scaler_to_pkl(scaler, path):
   # saving parameter scaler with joblib
    joblib.dump(scaler, path)

def save_preprocessed_data(df, path):
    # saving preprocessed data
    df.to_csv(path, index=False)
    print(f"Preprocessed data saved to: {path}")

def main():
    # load cleaned dataset
    df = load_data(DATA_PATH)

    # drop column
    df.drop(columns=['customerID'], inplace=True)

    # Convert 'TotalCharges' to numeric, forcing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Replace empty strings with NaN and drop NaN values
    df.replace('', float('nan'), inplace=True)
    df.dropna(inplace=True)

    # choose categorical and numerical features
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # encoding categorical features
    df, encoders = encode_categorical(df, categorical_features)

    # scaling numerical features
    df, scaler = scale_numerical(df, numerical_features)

    # saving preprocessed data and scaler
    save_preprocessed_data(df, OUTPUT_DATA_PATH)
    save_scaler_to_pkl(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

if __name__ == "__main__":
    main()
