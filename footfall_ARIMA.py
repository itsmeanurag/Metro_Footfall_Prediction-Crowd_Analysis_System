import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/Users/anuragtiwari/Desktop/Passenger_Flow/Hourly_Estimated_Footfall.csv")

# Fix Time Slot: Extract start time from "06:00 - 07:00"
df['Start Time'] = df['Time Slot'].str.extract(r'(\d{2}:\d{2})')

# Combine date and start time
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'])

# Sort by datetime
df = df.sort_values('Datetime')

# Extract time-based features
df['Day'] = df['Datetime'].dt.day_name()
df['Hour'] = df['Datetime'].dt.hour

# Encode categorical features
label_encoders = {}
for col in ['Icon', 'Day', 'Time Slot']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Rename columns to match
df.rename(columns={
    'Temperature': 'Temp',
    'Cloud Cover': 'Cloud Cover',
    'Estimated Hourly Footfall': 'Footfall'
}, inplace=True)

# Define target and exogenous variables
target = 'Footfall'
exog_cols = ['Temp', 'Cloud Cover', 'Icon', 'Day', 'Time Slot']

# Drop rows with missing values
df = df.dropna(subset=[target] + exog_cols)

# Use Jan to May as training, June as testing
df['Month'] = df['Datetime'].dt.month
train_df = df[df['Month'] <= 5]
test_df = df[df['Month'] == 6]

# Define series
y_train = train_df[target]
y_test = test_df[target]
exog_train = train_df[exog_cols]
exog_test = test_df[exog_cols]
datetime_test = test_df['Datetime']

# Fit ARIMA model
model = ARIMA(y_train, exog=exog_train, order=(2, 1, 2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(y_test), exog=exog_test)

# Accuracy metrics
r2 = r2_score(y_test, forecast)
mape = mean_absolute_percentage_error(y_test, forecast)
accuracy = 100 - (mape * 100)

# Print results
print(f"\nARIMA Model:")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape * 100:.2f}%")
print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# Plot first 100 samples
plt.figure(figsize=(12, 6))
plt.plot(datetime_test[:100], y_test.values[:100], label='Actual', color='blue', marker='o')
plt.plot(datetime_test[:100], forecast[:100], label='Forecast', color='red', linestyle='--')
plt.title(f'ARIMA Footfall Forecast vs Actual (First 100 Samples)\nAccuracy: {accuracy:.2f}%')
plt.xlabel('Datetime')
plt.ylabel('Estimated Hourly Footfall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
