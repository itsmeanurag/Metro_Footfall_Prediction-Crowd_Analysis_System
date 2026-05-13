import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess data
df = pd.read_csv('/Users/anuragtiwari/Desktop/Passenger_Flow/Hourly_Estimated_Footfall.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Start Time'] = df['Time Slot'].str.extract(r'(\d{2}:\d{2})')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Start Time'])
df = df.sort_values('Datetime')
df['Day'] = df['Datetime'].dt.day_name()
df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month

# Encode categorical variables
label_encoders = {}
for col in ['Icon', 'Day', 'Time Slot']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
features = ['Temperature', 'Cloud Cover', 'Icon', 'Day', 'Time Slot']
target = 'Estimated Hourly Footfall'

# Drop NaNs
df.dropna(subset=features + [target], inplace=True)

# Train on Jan–May, Test on June
train_df = df[df['Month'] <= 5]
test_df = df[df['Month'] == 6]

# Normalize and prepare data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[features + [target]])
scaled_test = scaler.transform(test_df[features + [target]])

X_train = scaled_train[:, :-1]
y_train = scaled_train[:, -1]
X_test = scaled_test[:, :-1]
y_test = scaled_test[:, -1]

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build 1D CNN model
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
    verbose=1
)

# Predict
y_pred = model.predict(X_test).flatten()

# Rescale predictions and actuals
y_test_rescaled = scaler.inverse_transform(np.hstack((X_test.squeeze(), y_test.reshape(-1, 1))))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.hstack((X_test.squeeze(), y_pred.reshape(-1, 1))))[:, -1]

# Accuracy metrics
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
accuracy = 100 - (mape * 100)

print("\n📊 1D CNN Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape * 100:.2f}%")
print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# Plot first 100 predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100], label='Actual', marker='o')
plt.plot(y_pred_rescaled[:100], label='Predicted', linestyle='--')
plt.title("1D CNN Footfall Prediction (First 100 June Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Footfall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
