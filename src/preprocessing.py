import pandas as pd

# Load dataset
RAW_DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'

def load_data(path):
    df = pd.read_csv(path)
    return df

def review_data(df):
    print("First 5 rows:\n", df.head())
    print("\nColumn Info:\n")
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())

    # Check if any cells have only whitespace
    print("\nWhitespace cells per column:\n", (df == ' ').sum())

    # Check duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())

def clean_data(df):
    # Replace empty strings or spaces with NaN
    df.replace(" ", pd.NA, inplace=True)

    # Drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"\nCleaned data saved to {path}")

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    review_data(df)
    cleaned_df = clean_data(df)
    save_data(cleaned_df, PROCESSED_DATA_PATH)
