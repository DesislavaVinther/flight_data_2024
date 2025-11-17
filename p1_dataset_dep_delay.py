import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Enables interactive plots in PyCharm
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2024.csv",
    low_memory=False
)

delay_cols = [
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay',
    'dep_delay'
]

delay_cols = [col for col in delay_cols if col in df.columns]

for col in delay_cols:
    df[col] = df[col].clip(lower=0)

total_delays = df[delay_cols].sum()


dep_delay_total = total_delays['dep_delay']
causal_sum_total = total_delays[['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']].sum()

missing = dep_delay_total - causal_sum_total
if missing < 0:
    missing = 0

plt.figure(figsize=(6, 6))

plt.bar('Departure Delay', dep_delay_total, color='gray', label='Total Departure Delay')

bottom = 0
colors = plt.cm.viridis_r([0.1, 0.3, 0.5, 0.7, 0.9])
for (cause, value, color) in zip(
    ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay'],
    total_delays[:-1],
    colors
):
    plt.bar('Sum of Causes', value, bottom=bottom, color=color, label=cause.replace('_', ' ').title())
    bottom += value

if missing > 0:
    plt.bar(
        'Sum of Causes',
        missing,
        bottom=bottom,
        color='none',
        edgecolor='black',
        hatch='///',
        linewidth=1.2,
        label='Unexplained Delay'
    )

plt.title("Total Departure Delay vs. Total Known Delay Causes", fontsize=14)
plt.ylabel("Total Delay (minutes)")
plt.legend(title="Delay Cause", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show(block=True)

