import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Hourly_Estimated_Footfall.csv")

# Combine Date and Time Slot into a single datetime column
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time Slot"].str[:5])
df.sort_values("Datetime", inplace=True)
df.set_index("Datetime", inplace=True)
df['Month'] = df.index.month

# Select the target variable
ts = df["Estimated Hourly Footfall"]

# Split into Jan–May (train) and June (test)
train = ts[df['Month'] <= 5]
test = ts[df['Month'] == 6]

# Fit SARIMA model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 17),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# Forecast for June
forecast = results.forecast(steps=len(test))

# Evaluate accuracy
r2 = r2_score(test, forecast)
mape = mean_absolute_percentage_error(test, forecast) * 100
accuracy = 100 - mape

# Print metrics
print(f"\n📊 SARIMA Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# Plot first 100 actual vs forecast values
N = 100
plt.figure(figsize=(12, 6))
plt.plot(test.index[:N], test[:N], label='Actual', color='blue', marker='o')
plt.plot(test.index[:N], forecast[:N], label='Forecast', color='orange', linestyle='--')
plt.title(f"SARIMA Forecast vs Actual (First 100 June Samples)\nAccuracy: {accuracy:.2f}%")
plt.xlabel("Datetime")
plt.ylabel("Estimated Hourly Footfall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
