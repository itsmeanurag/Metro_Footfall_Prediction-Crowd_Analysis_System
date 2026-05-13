import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
df = pd.read_csv("Hourly_Estimated_Footfall.csv")

# Extract start time and datetime
df['Start Time'] = df['Time Slot'].str.extract(r'(\d{2}:\d{2})')
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'])
df = df.sort_values('Datetime')
df['Month'] = df['Datetime'].dt.month

# Encode categorical features
label_encoders = {}
for col in ['Icon', 'Time Slot']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Use datetime index
df.set_index('Datetime', inplace=True)

# Target variable
target_series = df['Estimated Hourly Footfall'].values.reshape(-1, 1)

# Scale footfall
scaler = MinMaxScaler()
scaled = scaler.fit_transform(target_series)

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(scaled, sequence_length)

# Align datetime index with sequence output
df = df.iloc[sequence_length:]
df['Seq_Target'] = y

# Split by Jan–May (train) and June (test)
X_train = X[df['Month'] <= 5]
y_train = y[df['Month'] <= 5]
X_test = X[df['Month'] == 6]
y_test = y[df['Month'] == 6]
dt_test = df[df['Month'] == 6].index

# Build and train LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict
y_pred = model.predict(X_test)
y_test_rescaled = scaler.inverse_transform(y_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Accuracy metrics
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
accuracy = 100 - mape * 100

print(f"\n📊 LSTM Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape * 100:.2f}%")
print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100], label='Actual', marker='o')
plt.plot(y_pred_rescaled[:100], label='Predicted', linestyle='--')
plt.title("LSTM Footfall Prediction (First 100 June Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Estimated Hourly Footfall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
