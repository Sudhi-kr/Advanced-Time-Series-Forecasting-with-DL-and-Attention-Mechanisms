# ============================================================
# Advanced Time Series Forecasting with Seq2Seq + Attention
# ============================================================

# -----------------------------
# 1. Imports
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 2. Synthetic Multivariate Time Series Generation
# -----------------------------
def generate_synthetic_data(n_steps=1200):
    time = np.arange(n_steps)

    trend = 0.01 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 0.5, n_steps)

    feature_1 = trend + seasonal + noise
    feature_2 = 0.8 * feature_1 + np.random.normal(0, 0.3, n_steps)
    feature_3 = 0.5 * feature_1 + 0.3 * feature_2 + np.random.normal(0, 0.2, n_steps)

    data = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3
    })

    # Introduce missing values
    for col in data.columns:
        data.loc[data.sample(frac=0.02).index, col] = np.nan

    return data

data = generate_synthetic_data()

# -----------------------------
# 3. Preprocessing
# -----------------------------
data = data.interpolate().fillna(method="bfill")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# 4. Sequence Creation
# -----------------------------
def create_sequences(data, input_len=30, output_len=10):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, 0])
    return np.array(X), np.array(y)

INPUT_LEN = 30
OUTPUT_LEN = 10

X, y = create_sequences(scaled_data, INPUT_LEN, OUTPUT_LEN)

# -----------------------------
# 5. Rolling-Origin Cross Validation
# -----------------------------
def rolling_cv_splits(X, y, splits=3):
    fold_size = len(X) // splits
    for i in range(splits):
        train_end = (i + 1) * fold_size
        yield X[:train_end], y[:train_end], X[train_end:train_end+fold_size], y[train_end:train_end+fold_size]

# -----------------------------
# 6. Custom Attention Layer
# -----------------------------
class AttentionLayer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1), a

# -----------------------------
# 7. Seq2Seq Model with Attention
# -----------------------------
def build_attention_model():
    encoder_inputs = Input(shape=(INPUT_LEN, 3))
    encoder_lstm = LSTM(64, return_sequences=True)(encoder_inputs)

    context_vector, attention_weights = AttentionLayer()(encoder_lstm)

    decoder_input = RepeatVector(OUTPUT_LEN)(context_vector)
    decoder_lstm = LSTM(64, return_sequences=True)(decoder_input)
    decoder_output = TimeDistributed(Dense(1))(decoder_lstm)

    model = Model(encoder_inputs, decoder_output)
    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )

    return model

# -----------------------------
# 8. Baseline LSTM Model
# -----------------------------
def build_baseline_model():
    model = tf.keras.Sequential([
        LSTM(64, input_shape=(INPUT_LEN, 3)),
        RepeatVector(OUTPUT_LEN),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

# -----------------------------
# 9. Training + Evaluation
# -----------------------------
def evaluate_model(model_builder):
    rmses, maes, mapes = [], [], []

    for X_train, y_train, X_val, y_val in rolling_cv_splits(X, y):
        model = model_builder()
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        preds = model.predict(X_val).squeeze()

        rmse = np.sqrt(mean_squared_error(y_val.flatten(), preds.flatten()))
        mae = mean_absolute_error(y_val.flatten(), preds.flatten())
        mape = np.mean(np.abs((y_val.flatten() - preds.flatten()) / y_val.flatten())) * 100

        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)

    return np.mean(rmses), np.mean(maes), np.mean(mapes)

att_rmse, att_mae, att_mape = evaluate_model(build_attention_model)
base_rmse, base_mae, base_mape = evaluate_model(build_baseline_model)

print("Attention Model → RMSE:", att_rmse, "MAE:", att_mae, "MAPE:", att_mape)
print("Baseline LSTM → RMSE:", base_rmse, "MAE:", base_mae, "MAPE:", base_mape)

# -----------------------------
# 10. Train Best Model on Full Data
# -----------------------------
final_model = build_attention_model()
final_model.fit(X, y, epochs=15, batch_size=32, verbose=0)

# -----------------------------
# 11. Attention Weight Analysis
# -----------------------------
attention_extractor = Model(
    inputs=final_model.input,
    outputs=final_model.layers[2].output[1]
)

attention_weights = attention_extractor.predict(X[-1:])

plt.figure(figsize=(10, 4))
plt.plot(attention_weights[0])
plt.title("Attention Weights Over Input Timesteps")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.show()

# -----------------------------
# 12. Final Forecast (Last 50 Values)
# -----------------------------
last_input = scaled_data[-INPUT_LEN:]
last_input = last_input.reshape(1, INPUT_LEN, 3)

forecast_scaled = final_model.predict(last_input).flatten()

forecast = scaler.inverse_transform(
    np.c_[forecast_scaled, np.zeros((len(forecast_scaled), 2))]
)[:, 0]

final_output = forecast[:50]

print("\nFinal 50 Forecasted Values:\n")
print(final_output)
