import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import h5py
import os

# File paths
DATA_PATH = 'data/processed_data.csv'  # cleaning result
OUTPUT_DATA_PATH = 'data/preprocessed_data.csv'
SCALER_PATH = 'models/scaler.h5'

os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def encode_categorical(df, cat_features):
    # encoding on categorical features
    label_encoders = {}  # saving encoder for each columns

    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def scale_numerical(df, num_features):
    #s caling on nmerical features
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    return df, scaler

def save_scaler_to_h5(scaler, path):
   # saving parameter scaler
    with h5py.File(path, 'w') as f:
        f.create_dataset('mean_', data=scaler.mean_)
        f.create_dataset('scale_', data=scaler.scale_)
        f.create_dataset('var_', data=scaler.var_)

def save_preprocessed_data(df, path):
    # saving preprocessed data
    df.to_csv(path, index=False)
    print(f"Preprocessed data saved to: {path}")

def main():
    # load cleaned dataset
    df = load_data(DATA_PATH)

    # drop kolom 
    df.drop(columns=['customerID'], inplace=True)

    # conversi TotalCharges to numeric 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # drop NaN
    df.dropna(inplace=True)

    # choosing fitur categorical dan numerical
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # encoding categori
    df, encoders = encode_categorical(df, categorical_features)

    # scaling numeric
    df, scaler = scale_numerical(df, numerical_features)

    # saving preprocessing dan scaler result
    save_preprocessed_data(df, OUTPUT_DATA_PATH)
    save_scaler_to_h5(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

if __name__ == "__main__":
    main()
