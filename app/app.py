from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from tensorflow.keras.models import load_model # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the uploaded dataframe
df = None

# Load the pre-trained model and scaler
model = load_model('models/churn_model.h5')
scaler = joblib.load('models/scaler.pkl')  # assuming saved scaler

def preprocess_data(df):
    # Example preprocessing: handle missing values, encode categorical data
    df = df.dropna()  # Drop rows with missing values as an example
    # Encoding categorical features if necessary
    label_encoders = {}
    categorical_cols = ['gender', 'Partner', 'Dependents']  # Modify this based on your dataset
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Scaling the numerical features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']  # Modify as necessary
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    global df  # Access global df variable
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Read the uploaded CSV into DataFrame
            df = pd.read_csv(file)
            # Preprocess data for prediction
            processed_df = preprocess_data(df)
            
            # Make predictions
            predictions = model.predict(processed_df)
            df['Churn Prediction'] = predictions

            # Send data back to frontend for display
            return render_template('upload.html', data=df.to_html(classes='data', header=True))
    
    return render_template('index.html')

@app.route('/plot')
def plot():
    # Use the global df variable
    global df
    if df is None:
        return redirect(url_for('index'))  # Redirect to the index page if no data is uploaded
    
    # Create a plot and convert it to PNG
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn Prediction')
    plt.title('Churn Prediction Distribution')

    # Save plot to a PNG in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('plot.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
