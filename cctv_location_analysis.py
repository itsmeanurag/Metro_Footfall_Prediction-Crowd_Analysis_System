import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
cctv_metadata_path = "/Users/anuragtiwari/Downloads/Kalpur_CCTV_Metadata.xlsx"
df_cctv = pd.read_excel(cctv_metadata_path, sheet_name="Sheet1")

# Convert Date column to datetime format
df_cctv["Date"] = pd.to_datetime(df_cctv["Date"])

# Select a single day's data for CCTV analysis
latest_date_cctv = df_cctv["Date"].max()
df_cctv_day = df_cctv[df_cctv["Date"] == latest_date_cctv]

# Convert Hour column to integer (extracting only the hour from time format)
df_cctv_day["Hour"] = df_cctv_day["Hour"].astype(str).str[:2].astype(int)

# Filter data for hours between 6:00 and 23:00
df_cctv_day = df_cctv_day[(df_cctv_day["Hour"] >= 6) & (df_cctv_day["Hour"] <= 23)]

# Set visualization style
sns.set_theme(style="darkgrid")

# Get unique locations
locations = df_cctv_day["Location"].unique()

# Create individual bar graphs for each location
for location in locations:
    plt.figure(figsize=(12, 6))
    df_location = df_cctv_day[df_cctv_day["Location"] == location]
    sns.barplot(x="Hour", y="Detected Objects", data=df_location, palette="viridis")
    plt.title(f"📹 Crowd Density at {location} - {latest_date_cctv.strftime('%Y-%m-%d')}")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Detected Objects (Crowd Level)")
    plt.xticks(rotation=45)
    plt.show()

# Overall daily trend across all locations
plt.figure(figsize=(12, 6))
sns.lineplot(x="Hour", y="Detected Objects", data=df_cctv_day, estimator="sum", ci=None, marker="o", color="b")
plt.title(f"📈 Overall Crowd Trend - {latest_date_cctv.strftime('%Y-%m-%d')}")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Detected Objects")
plt.xticks(rotation=45)
plt.show()

# Heatmap for location-wise crowd distribution
pivot_table = df_cctv_day.pivot_table(index="Hour", columns="Location", values="Detected Objects", aggfunc="sum")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".0f")
plt.title(f"🔥 Heatmap of Crowd Density - {latest_date_cctv.strftime('%Y-%m-%d')}")
plt.xlabel("Location")
plt.ylabel("Hour of the Day")
plt.show()

# Identify peak and low traffic locations
total_detections = df_cctv_day.groupby("Location")["Detected Objects"].sum().sort_values()
plt.figure(figsize=(12, 6))
total_detections.plot(kind="barh", color="teal")
plt.title("🏆 Locations with Peak and Low Crowd Density")
plt.xlabel("Total Detected Objects")
plt.ylabel("Location")
plt.show()
