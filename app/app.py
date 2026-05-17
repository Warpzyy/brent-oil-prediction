import sys
import numpy as np

# Pendaftaran modul palsu untuk NumPy lama
sys.modules['numpy._core'] = np

import os
from flask import Flask, render_template, request
import joblib
import tensorflow as tf

app = Flask(__name__)

# Setup Path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load Semua Model & Scaler
lr_model = joblib.load(os.path.join(base_dir, '../models/linear_regression_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, '../models/scaler.pkl'))
ann_model = tf.keras.models.load_model(os.path.join(base_dir, '../models/ann_model.h5'))
lstm_model = tf.keras.models.load_model(os.path.join(base_dir, '../models/lstm_model.h5'))

@app.route('/')
def index():
    # Mengirim data dummy/statis untuk tabel evaluasi (sesuaikan dengan hasil notebook-mu)
    return render_template('index.html', 
                           mae_lr=0.85, rmse_lr=1.12,
                           mae_lstm=0.92, rmse_lstm=1.25,
                           mae_ann=1.05, rmse_ann=1.40,
                           mae_bp=2.10, rmse_bp=3.15)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        price_input = float(request.form['price'])
        input_data = np.array([[price_input]])
        
        # 1. Prediksi Linear Regression
        pred_lr = lr_model.predict(input_data)[0]
        
        # 2. Prediksi ANN & LSTM (Harus lewat Scaling)
        # Note: Karena LSTM butuh look_back, kita asumsikan input user adalah titik terakhir
        # Untuk demo, kita buat dummy sequence 30 hari yang isinya input user semua
        scaled_input = scaler.transform(input_data)
        dummy_seq = np.repeat(scaled_input, 30).reshape(1, 30, 1)
        
        pred_lstm_scaled = lstm_model.predict(dummy_seq, verbose=0)
        pred_lstm = scaler.inverse_transform(pred_lstm_scaled)[0][0]

        return render_template('index.html', 
                               original_input=price_input,
                               result_lr=round(float(pred_lr), 2),
                               result_lstm=round(float(pred_lstm), 2),
                               # Tetap kirim nilai tabel agar tidak hilang saat refresh prediksi
                               mae_lr=0.85, rmse_lr=1.12,
                               mae_lstm=0.92, rmse_lstm=1.25,
                               mae_ann=1.05, rmse_ann=1.40,
                               mae_bp=2.10, rmse_bp=3.15)

if __name__ == '__main__':
    app.run(debug=True)