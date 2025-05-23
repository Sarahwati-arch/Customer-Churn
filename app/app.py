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
import zipfile
import os


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

    if all(k in session for k in [
    'plot1_url', 'plot2_url', 'plot3_url', 'plot4_url', 'plot5_url',
    'plot6_url', 'plot7_url', 'plot8_url', 'plot9_url', 'plot10_url',
    'plot11_url', 'plot12_url',
    ]):
        return render_template('plot.html',
                           plot1_url=session['plot1_url'],
                           plot2_url=session['plot2_url'],
                           plot3_url=session['plot3_url'],
                           plot4_url=session['plot4_url'],
                           plot5_url=session['plot5_url'],
                           plot6_url=session['plot6_url'],
                           plot7_url=session['plot7_url'],
                           plot8_url=session['plot8_url'],
                           plot9_url=session['plot9_url'],
                           plot10_url=session['plot10_url'],
                           plot11_url=session['plot11_url'],
                           plot12_url=session['plot12_url'],)

    # Plot 1: Churn Prediction Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn Prediction', color='#6699cc')
    plt.title('Churn Prediction Distribution')
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot1_url = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # Plot 2: Churn Ratio
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

    # Plot 3: Internet Service Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='InternetService', palette='Set2')
    plt.title('Internet Service Types')
    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    plot3_url = base64.b64encode(img3.getvalue()).decode()
    plt.close()

    # Plot 4: Payment Method Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='PaymentMethod', palette='Set3')
    plt.title('Payment Method Distribution')
    plt.xticks(rotation=45)
    img4 = io.BytesIO()
    plt.savefig(img4, format='png')
    img4.seek(0)
    plot4_url = base64.b64encode(img4.getvalue()).decode()
    plt.close()

    # Plot 5: Contract Types
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Contract', palette='coolwarm')
    plt.title('Contract Type Distribution')
    img5 = io.BytesIO()
    plt.savefig(img5, format='png')
    img5.seek(0)
    plot5_url = base64.b64encode(img5.getvalue()).decode()
    plt.close()

    # Plot 6 - Monthly Charges (replacing previous tenure plot)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['MonthlyCharges'], kde=True, bins=30, color='skyblue')
    plt.title('Monthly Charges')
    img6 = io.BytesIO()
    plt.savefig(img6, format='png')
    img6.seek(0)
    plot6_url = base64.b64encode(img6.getvalue()).decode()
    plt.close()

    # Plot 7 - Monthly Charges by Churn (same as before)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn Prediction', y='MonthlyCharges', data=df, palette='pastel')
    plt.title('Monthly Charges by Churn')
    img7 = io.BytesIO()
    plt.savefig(img7, format='png')
    img7.seek(0)
    plot7_url = base64.b64encode(img7.getvalue()).decode()
    plt.close()

    # Plot 8 - Contract vs Churn (replacing previous dependents vs churn)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn Prediction', data=df, palette='muted')
    plt.title('Contract vs Churn')
    img8 = io.BytesIO()
    plt.savefig(img8, format='png')
    img8.seek(0)
    plot8_url = base64.b64encode(img8.getvalue()).decode()
    plt.close()


    # Plot 9: StreamingTV vs Churn
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='StreamingTV', hue='Churn Prediction', palette='Set1')
    plt.title('StreamingTV vs Churn')
    img9 = io.BytesIO()
    plt.savefig(img9, format='png')
    img9.seek(0)
    plot9_url = base64.b64encode(img9.getvalue()).decode()
    plt.close()

    # Plot 10: Gender vs Churn
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='gender', hue='Churn Prediction', palette='Set1')
    plt.title('Gender vs Churn')
    img10 = io.BytesIO()
    plt.savefig(img10, format='png')
    img10.seek(0)
    plot10_url = base64.b64encode(img10.getvalue()).decode()
    plt.close()


    # Plot 11: Customer Churn by Partner
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='Partner', hue='Churn Prediction', palette='coolwarm')
    plt.title('Customer Churn by Partner Status')
    plt.xlabel('Partner')
    plt.ylabel('Count')
    img11 = io.BytesIO()
    plt.savefig(img11, format='png')
    img11.seek(0)
    plot11_url = base64.b64encode(img11.getvalue()).decode()
    plt.close()


    # Plot 12: Customer Churn by Senior Citizen
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='SeniorCitizen', hue='Churn Prediction', palette='Set2')
    plt.title('Churn by Senior Citizen Status')
    plt.xlabel('Senior Citizen (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    img12 = io.BytesIO()
    plt.savefig(img12, format='png')
    img12.seek(0)
    plot12_url = base64.b64encode(img12.getvalue()).decode()
    plt.close()


    # Save all plots to session
    session['plot1_url'] = plot1_url
    session['plot2_url'] = plot2_url
    session['plot3_url'] = plot3_url
    session['plot4_url'] = plot4_url
    session['plot5_url'] = plot5_url
    session['plot6_url'] = plot6_url
    session['plot7_url'] = plot7_url
    session['plot8_url'] = plot8_url
    session['plot9_url'] = plot9_url
    session['plot10_url'] = plot10_url
    session['plot11_url'] = plot11_url
    session['plot12_url'] = plot12_url

    

    return render_template('plot.html',
                       plot1_url=plot1_url,
                       plot2_url=plot2_url,
                       plot3_url=plot3_url,
                       plot4_url=plot4_url,
                       plot5_url=plot5_url,
                       plot6_url=plot6_url,
                       plot7_url=plot7_url,
                       plot8_url=plot8_url,
                       plot9_url=plot9_url,
                       plot10_url=plot10_url,
                       plot11_url=plot11_url,
                       plot12_url=plot12_url,)




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
    
@app.route('/download_plots')
def download_plots():
    print(">>> ENTERED /download_plots ROUTE")
    print("session keys:", list(session.keys()))

    plot_keys = [
        'plot1_url', 'plot2_url', 'plot3_url', 'plot4_url', 'plot5_url',
        'plot6_url', 'plot7_url', 'plot8_url', 'plot9_url', 'plot10_url',
        'plot11_url', 'plot12_url'
    ]

    if not all(key in session for key in plot_keys):
        return redirect(url_for('plot'))  # Sesuaikan kalau route-nya beda

    # Buat temporary zip file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        with zipfile.ZipFile(tmp_zip, 'w') as zipf:
            temp_files = []
            try:
                for i, key in enumerate(plot_keys, start=1):
                    img_data = base64.b64decode(session[key])
                    plot_filename = f'plot{i}.png'

                    # Buat temp file untuk gambar
                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    try:
                        tmp_img.write(img_data)
                        tmp_img.flush()
                        tmp_img.close()  # PENTING supaya file bisa dibaca oleh zipf.write
                        temp_files.append(tmp_img.name)

                        # Tambahkan file ke zip, dengan nama di dalam zip
                        zipf.write(tmp_img.name, arcname=plot_filename)
                    except Exception as e:
                        print("Error writing image to temp file:", e)
                    finally:
                        tmp_img.close()

            finally:
                # Hapus semua temp image files setelah zip selesai dibuat
                for fpath in temp_files:
                    if os.path.exists(fpath):
                        os.remove(fpath)

        tmp_zip_path = tmp_zip.name

    # Kirim file zip sebagai attachment, hapus file setelah selesai kirim
    try:
        return send_file(tmp_zip_path, as_attachment=True, download_name='plots.zip', mimetype='application/zip')
    finally:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)

@app.route('/generate_plots')
def generate_plots():
    plots = {}

    for i in range(1, 13):
        fig = plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [i, i*2, i*3])  # Contoh plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots[f'plot{i}_url'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

    # Simpan ke session
    for key, val in plots.items():
        session[key] = val

    return redirect(url_for('show_plots'))

@app.route('/show_plots')
def show_plots():
    plot_data = {}
    for i in range(1, 13):
        key = f'plot{i}_url'
        plot_data[key] = session.get(key)

    return render_template('show_plots.html', plots=plot_data)



@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    # You can handle form data here
    theme = request.form.get('theme')
    notifications = 'notifications' in request.form
    print(f"Theme: {theme}, Notifications: {notifications}")
    
    # Normally you'd save these to a database or config
    return redirect(url_for('settings'))

    

if __name__ == '__main__':
    app.run(debug=True)
