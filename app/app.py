import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.preprocessing import preprocess_input_data
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from tensorflow.keras.models import load_model  # type: ignore
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__)

# Set secret key for session
app.secret_key = os.urandom(24)

# Global variable to store the uploaded dataframe
df = None

# Load the pre-trained models and scaler
model = joblib.load('models/logreg_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Assuming saved scaler for scaling features

def preprocess_data(df):
    # Drop the 'customerID' column if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Handle the 'TotalCharges' column (convert to numeric and drop missing values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.replace('', float('nan'), inplace=True)
    df.dropna(inplace=True)

    # Define the categorical and numerical features
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Re-encode categorical columns using Label Encoding
    for col in categorical_features:
        df[col] = df[col].astype('category').cat.codes

    # Scale the numerical features using the previously saved scaler
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    global df

    if request.method == 'POST':
        file = request.files['file']
        
        if not file or not file.filename.endswith('.csv'):
            return "Please upload a valid CSV file."
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            return "Error reading the CSV file. Please upload a valid CSV file."

        try:
            # ✅ Simpan customerID untuk ditambahkan kembali nanti
            customer_ids = df['customerID']

            # Preprocess tanpa menghapus customerID dari df asli
            processed_df = preprocess_data(df.copy())  # use a copy to avoid modifying original df

            expected_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ]
            processed_df = processed_df[expected_columns]

            # ✅ Prediksi
            predictions = model.predict(processed_df.values)
            predictions = (predictions > 0.5).astype(int)

            # ✅ Gabungkan kembali customerID dan hasil prediksi ke DataFrame asli
            df = df.loc[processed_df.index]  # Sesuaikan baris yang tersisa setelah dropna
            df['customerID'] = customer_ids.loc[processed_df.index]
            df['Churn Prediction'] = predictions

        except Exception as e:
            print(f"Error during processing or prediction: {e}")
            return "Error during processing or prediction. Please check the model and input data."

        return render_template('upload.html', data=df.to_html(classes='data', header=True))

    return render_template('index.html')


@app.route('/plot')
def plot():
    global df
    if df is None:
        return redirect(url_for('index'))

    # Cek apakah plot sudah ada dalam sesi
    if 'plot1_url' in session and 'plot2_url' in session:
        # Jika plot sudah ada dalam sesi, langsung tampilkan
        return render_template('plot.html', plot1_url=session['plot1_url'], plot2_url=session['plot2_url'])

    # === Countplot: Churn Distribution ===
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn Prediction')
    plt.title('Churn Prediction Distribution')
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot1_url = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # === Pie Chart: Churn vs Non-Churn ===
    plt.figure(figsize=(6, 6))
    churn_counts = df['Churn Prediction'].value_counts()
    labels = ['Not Churn', 'Churn']
    plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Churn Ratio')
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', dpi=150)  # Menurunkan DPI untuk gambar lebih kecil
    img2.seek(0)
    plot2_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    # Simpan URL plot ke sesi
    session['plot1_url'] = plot1_url
    session['plot2_url'] = plot2_url

    return render_template('plot.html', plot1_url=plot1_url, plot2_url=plot2_url)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data (for example, from form or file)
        input_data = {
            'gender': 'Male',  
            'SeniorCitizen': 0, 
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'No',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'tenure': 24,
            'MonthlyCharges': 80.75,
            'TotalCharges': 1938.00
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data before passing it to the model
        input_scaled = preprocess_input_data(input_df)

        # Make a prediction using the model
        prediction = model.predict(input_scaled)

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

@app.route('/download')
def download_csv():
    global df
    if df is None:
        return redirect(url_for('index'))

    # Buat file CSV sementara
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as tmp:
        df.to_csv(tmp.name, index=False)

        # Kirim file CSV sementara sebagai respons unduhan
        return send_file(tmp.name, as_attachment=True, download_name="processed_data.csv", mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)
