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
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model & scaler
model = joblib.load('models/logreg_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Global dataframe
df = None


def preprocess_data(df):
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.replace('', float('nan'), inplace=True)
    df.dropna(inplace=True)

    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in categorical_features:
        df[col] = df[col].astype('category').cat.codes

    df[numerical_features] = scaler.transform(df[numerical_features])

    return df


@app.route('/')
def index():
    print(">>> Rendering index.html")
    return render_template('index.html')

@app.route('/format')
def format_page():
    print(">>> /format route triggered")  # <--- Tambahkan ini
    return render_template('format.html')




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print(">>> Upload route triggered")  # Debug print
    global df
    if request.method == 'POST':
        file = request.files['file']
        if not file or not file.filename.endswith('.csv'):
            return "Please upload a valid CSV file."

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error reading CSV file: {e}"

        try:
            customer_ids = df['customerID']
            processed_df = preprocess_data(df.copy())
            expected_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ]
            processed_df = processed_df[expected_columns]

            predictions = model.predict(processed_df.values)
            predictions = (predictions > 0.5).astype(int)

            df = df.loc[processed_df.index]
            df['customerID'] = customer_ids.loc[processed_df.index]
            df['Churn Prediction'] = predictions

        except Exception as e:
            return f"Prediction error: {e}"

        return render_template('upload.html', data=df.to_html(classes='data', header=True))

    return render_template('upload.html')


@app.route('/plot')
def plot():
    global df
    if df is None:
        return redirect(url_for('upload'))

    if 'plot1_url' in session and 'plot2_url' in session:
        return render_template('plot.html', plot1_url=session['plot1_url'], plot2_url=session['plot2_url'])

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn Prediction')
    plt.title('Churn Prediction Distribution')
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot1_url = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(6, 6))
    churn_counts = df['Churn Prediction'].value_counts()
    labels = ['Not Churn', 'Churn']
    plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Churn Ratio')
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', dpi=150)
    img2.seek(0)
    plot2_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    session['plot1_url'] = plot1_url
    session['plot2_url'] = plot2_url

    return render_template('plot.html', plot1_url=plot1_url, plot2_url=plot2_url)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
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

        input_df = pd.DataFrame([input_data])
        input_scaled = preprocess_input_data(input_df)
        prediction = model.predict(input_scaled)

        return render_template('result.html', prediction=prediction[0])

    return redirect(url_for('index'))


@app.route('/download')
def download_csv():
    print(">>> MASUK DOWNLOAD ROUTE")
    global df
    if df is None:
        print(">>> df kosong, redirect ke /upload")
        return redirect(url_for('upload'))
    ...

    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv') as tmp:
        df.to_csv(tmp.name, index=False)
        return send_file(tmp.name, as_attachment=True, download_name="processed_data.csv", mimetype='text/csv')


if __name__ == '__main__':
    app.run(debug=True)
