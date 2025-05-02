import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/processed_data.csv')

# Preprocessing: Pisahkan fitur dan target
X = df.drop(['Churn'], axis=1)  # Fitur: Semua kolom kecuali 'Churn'
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Target: 1 untuk churn, 0 untuk tidak churn

# Mengonversi data kategorikal menjadi numerik jika perlu
X = pd.get_dummies(X, drop_first=True)

# Scaling: Standarkan fitur menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membangun model Keras (Neural Network sederhana)
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer dan hidden layer pertama
    Dense(32, activation='relu'),  # Hidden layer kedua
    Dense(1, activation='sigmoid')  # Output layer (Churn: 1 atau 0)
])

# Mengkompilasi model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Menyimpan model ke file churn_model.h5
model.save('models/churn_model.h5')

print("Model berhasil disimpan dalam 'models/churn_model.h5'")
