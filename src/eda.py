import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = 'data/processed_data.csv'
SAVE_PATH = 'reports/eda/'

# Ensure directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'churn_distribution.png')
    plt.close()

def plot_categorical_features(df, cat_features):
    for col in cat_features:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue='Churn')
        plt.title(f'{col} vs Churn')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(SAVE_PATH + f'{col}_vs_churn.png')
        plt.close()

def plot_numerical_features(df, num_features):
    for col in num_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue='Churn', kde=True, bins=30)
        plt.title(f'{col} Distribution by Churn')
        plt.tight_layout()
        plt.savefig(SAVE_PATH + f'{col}_distribution_by_churn.png')
        plt.close()

def main():
    df = load_data(DATA_PATH)

    # Drop irrelevant columns
    df.drop(columns=['customerID'], inplace=True)

    # Identify categorical and numerical features
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Ensure correct dtypes
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing numeric conversions (if any)
    df.dropna(subset=numerical_features, inplace=True)

    plot_churn_distribution(df)
    plot_categorical_features(df, categorical_features)
    plot_numerical_features(df, numerical_features)

    print("EDA plots saved to:", SAVE_PATH)

if __name__ == "__main__":
    main()
