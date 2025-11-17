import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Forces interactive backend for PyCharm
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2024.csv",
    low_memory=False
)

# --- Relevant delay columns ---
delay_cols = [
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]
delay_cols = [col for col in delay_cols if col in df.columns]

# --- Group by airline and sum delays ---
airline_delays = df.groupby('op_unique_carrier')[delay_cols].sum()

# Optional: sort airlines by total delays
airline_delays['total_delay'] = airline_delays.sum(axis=1)
airline_delays = airline_delays.sort_values('total_delay', ascending=False)
airline_delays.drop(columns='total_delay', inplace=True)

# --- Plot stacked bar chart ---
plt.figure(figsize=(12, 6))
airline_delays.plot(
    kind='bar',
    stacked=True,
    colormap='tab20',
    figsize=(12, 6)
)
plt.ylabel("Total Delay Minutes")
plt.xlabel("Airline")
plt.title("Total Delay Minutes by Airline and Delay Type")
plt.xticks(rotation=45)
plt.legend(title="Delay Type")
plt.tight_layout()
plt.show()

delay_cols = [col for col in delay_cols if col in df.columns]

# --- Group by airline and sum delays ---
airline_delays = df.groupby('op_unique_carrier')[delay_cols].sum()

# Sort airlines by total delays
airline_delays['total_delay'] = airline_delays.sum(axis=1)
airline_delays = airline_delays.sort_values('total_delay', ascending=True)  # smallest on top
airline_delays.drop(columns='total_delay', inplace=True)

# --- Normalize to percentages ---
airline_percent = airline_delays.div(airline_delays.sum(axis=1), axis=0) * 100

# --- Plot horizontal stacked bar chart ---
plt.figure(figsize=(12, 8))
bottoms = [0] * len(airline_percent)
colors = plt.cm.tab20.colors

for i, col in enumerate(airline_percent.columns):
    plt.barh(
        airline_percent.index,
        airline_percent[col],
        left=bottoms,
        color=colors[i % len(colors)],
        label=col
    )
    bottoms = [x + y for x, y in zip(bottoms, airline_percent[col])]

plt.xlabel("Percentage of Total Delay per Airline (%)")
plt.ylabel("Airline")
plt.title("Airline Delay Breakdown by Cause")
plt.legend(title="Delay Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()