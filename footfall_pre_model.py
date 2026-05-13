import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/anuragtiwari/Desktop/Passenger_Flow/Dataset/Kalpur_Metro_Footfall_with_Weather.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Drop unnecessary column
df.drop(columns=['Station Name'], inplace=True)

# Encode categorical features
label_encoders = {}
for col in ['Day of Week', 'Time Slot', 'Icon']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Add 'Is Weekday' feature
df['Is Weekday'] = df['Date'].dt.dayofweek < 5

# Split into train (Jan–May) and test (June)
train_df = df[df['Date'].dt.month <= 5]
test_df = df[df['Date'].dt.month == 6]

# Define features and target
features = ['Day of Week', 'Time Slot', 'Temperature', 'Cloud Cover', 'Icon', 'Is Weekday']
target = 'Total Footfall'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# ---------- Linear Regression ----------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# ---------- Tuned Random Forest ----------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=21
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ---------- Evaluation Function ----------
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 100 - (mape * 100)

    print(f"\n{name} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# Evaluate both models
evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest (Tuned)", y_test, y_pred_rf)

# ---------- Plot First 100 Predictions ----------
plt.figure(figsize=(14,6))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred_lr[:100], label='Linear Regression', linestyle='--')
plt.plot(y_pred_rf[:100], label='Random Forest (Tuned)', linestyle='-.')
plt.title("Footfall Prediction (First 100 June Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Total Footfall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
