import pandas as pd

# Load the daily total footfall data
df_daily = pd.read_csv("/Users/anuragtiwari/Desktop/Passenger_Flow/Daily_Total_Footfall_with_Weather.csv")

# Hourly data from 06:00 to 22:00 (excluding 23:00–00:00)
hourly_entry = [24, 101, 242, 186, 357, 102, 34, 45, 61, 27, 20, 106, 59, 52, 78, 106, 4]
hourly_exit  = [28, 76,  54,  63,  81,  61, 64, 83, 90, 77, 99, 169, 306, 277, 162, 91, 38]

# Calculate hourly total and ratios
hourly_total = [e + x for e, x in zip(hourly_entry, hourly_exit)]
total_weekly_sum = sum(hourly_total)
hourly_ratio = [val / total_weekly_sum for val in hourly_total]

# Time slot labels from 06:00 to 23:00 (excluding 23:00–00:00)
time_slots = [f"{str(6 + i).zfill(2)}:00 - {str(7 + i).zfill(2)}:00" for i in range(17)]

# Prepare hourly data
rows = []
for _, row in df_daily.iterrows():
    for i in range(17):  # Only loop over 17 hours now
        hourly_footfall = row['Total Footfall'] * hourly_ratio[i]
        rows.append({
            "Date": row['Date'],
            "Time Slot": time_slots[i],
            "Estimated Hourly Footfall": int(hourly_footfall),
            "Day of Week": row['Day of Week'],
            "Temperature": row['Temperature'],
            "Cloud Cover": row['Cloud Cover'],
            "Icon": row['Icon']
        })

# Create DataFrame and save to CSV
df_hourly = pd.DataFrame(rows)
df_hourly.to_csv("Hourly_Estimated_Footfall.csv", index=False)

print("✅ Hourly data saved as 'Hourly_Estimated_Footfall.csv' (without 23:00–00:00)")
