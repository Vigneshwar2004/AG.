import pandas as pd
import numpy as np
from datetime import datetime
import random

np.random.seed(42)

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31, 23)
timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

data = {
    "Timestamp": timestamps,
    "Occupancy_Count": np.random.poisson(lam=50, size=len(timestamps)),
    "Outdoor_Temp_C": np.random.normal(loc=30, scale=5, size=len(timestamps)),
    "Humidity_%": np.random.uniform(low=50, high=90, size=len(timestamps)),
    "Wind_Speed_kph": np.random.uniform(low=0, high=20, size=len(timestamps)),
    "Day_Type": ['Weekend' if ts.weekday() >= 5 else 'Weekday' for ts in timestamps],
    "Hour": [ts.hour for ts in timestamps],
}

data["HVAC_Status"] = ["On" if (occ > 10 and temp > 25) else "Off"
                       for occ, temp in zip(data["Occupancy_Count"], data["Outdoor_Temp_C"])]

data["Indoor_Temp_C"] = [round(temp - 5 if hvac == "On" else temp - 2, 1)
                         for temp, hvac in zip(data["Outdoor_Temp_C"], data["HVAC_Status"])]

data["Energy_Consumption_kWh"] = [
    round(100 + occ * 2 + (temp - 22)**2 * 0.5 + random.uniform(-20, 20), 2)
    for occ, temp in zip(data["Occupancy_Count"], data["Indoor_Temp_C"])
]


df = pd.DataFrame(data)

print(df.head(10).to_markdown(index=False))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("smart_building_energy_efficiency.csv", parse_dates=['Timestamp'])

sns.set(style="whitegrid")


plt.figure(figsize=(10, 6))
hourly_avg = df.groupby('Hour')['Energy_Consumption_kWh'].mean()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o')
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy Consumption (kWh)')
plt.title('Average Energy Consumption by Hour of Day')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()
