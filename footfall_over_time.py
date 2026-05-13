import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/Users/anuragtiwari/Desktop/Passenger_Flow/Dataset/Kalpur_Metro_Footfall_with_Weather.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Group by date to get total footfall per day
daily_footfall = df.groupby('Date')['Total Footfall'].sum().reset_index()

# Extract useful features
daily_footfall['Month'] = daily_footfall['Date'].dt.to_period('M')
daily_footfall['DayOfWeek'] = daily_footfall['Date'].dt.day_name()

# Set plot style
sns.set(style="whitegrid")

# 📆 Plot daily footfall for each month separately
months = daily_footfall['Month'].unique()

for month in months:
    plt.figure(figsize=(12, 6))
    df_month = daily_footfall[daily_footfall['Month'] == month]
    plt.plot(df_month['Date'], df_month['Total Footfall'], marker='o', color='darkblue')
    plt.title(f"📊 Daily Total Footfall - {month}")
    plt.xlabel("Date")
    plt.ylabel("Total Footfall")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 📈 Overall daily total footfall
plt.figure(figsize=(14, 6))
plt.plot(daily_footfall['Date'], daily_footfall['Total Footfall'], color='teal')
plt.title("📈 Overall Daily Total Footfall (Jan - June 2024)")
plt.xlabel("Date")
plt.ylabel("Total Footfall")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📊 Bar plot: Average footfall by day of the week
plt.figure(figsize=(10, 6))
avg_daywise = daily_footfall.groupby('DayOfWeek')['Total Footfall'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
sns.barplot(x=avg_daywise.index, y=avg_daywise.values, palette='viridis')
plt.title("🗓️ Average Total Footfall by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Total Footfall")
plt.tight_layout()
plt.show()
