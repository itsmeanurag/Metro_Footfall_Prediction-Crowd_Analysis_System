import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
cctv_metadata_path = "/Users/anuragtiwari/Downloads/Kalpur_CCTV_Metadata.xlsx"
metro_footfall_path = "/Users/anuragtiwari/Downloads/Kalpur_Metro_Footfall.xlsx"

df_cctv = pd.read_excel(cctv_metadata_path, sheet_name="Sheet1")
df_footfall = pd.read_excel(metro_footfall_path, sheet_name="Sheet1")

# Convert Date column to datetime format
df_cctv["Date"] = pd.to_datetime(df_cctv["Date"])
df_footfall["Date"] = pd.to_datetime(df_footfall["Date"])

# Select a single day's data for CCTV analysis
latest_date_cctv = df_cctv["Date"].max()
df_cctv_day = df_cctv[df_cctv["Date"] == latest_date_cctv]

# Select last 7 days for metro footfall analysis
latest_dates_footfall = df_footfall["Date"].drop_duplicates().nlargest(7)
df_footfall_week = df_footfall[df_footfall["Date"].isin(latest_dates_footfall)]

# Set visualization style
sns.set_theme(style="darkgrid")

# Plot 1: CCTV Metadata - Crowd Density Across Hours for One Day
plt.figure(figsize=(12, 6))
sns.barplot(x="Hour", y="Detected Objects", hue="Location", data=df_cctv_day)
plt.title(f"📹 Crowd Density Analysis - {latest_date_cctv.strftime('%Y-%m-%d')}")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Detected Objects (Crowd Level)")
plt.xticks(rotation=45)
plt.legend(title="Location", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# Plot 2: Metro Footfall - Entry, Exit, and Total Footfall over 7 Days
plt.figure(figsize=(12, 6))
df_footfall_week_melted = df_footfall_week.melt(id_vars=["Date", "Time Slot"], 
                                                value_vars=["Entry Passengers", "Exit Passengers", "Total Footfall"], 
                                                var_name="Passenger Type", value_name="Count")

sns.barplot(x="Date", y="Count", hue="Passenger Type", data=df_footfall_week_melted)
plt.title("🚉 Passenger Flow Analysis (Entry, Exit & Total) - Last 7 Days")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend(title="Passenger Type")
plt.xticks(rotation=45)
plt.show()
