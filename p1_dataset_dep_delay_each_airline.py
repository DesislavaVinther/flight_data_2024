import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Enables interactive plots in PyCharm
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2024.csv",
    low_memory=False
)

# --- Relevant columns ---
delay_cols = [
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay',
    'dep_delay'
]

delay_cols = [col for col in delay_cols if col in df.columns]

# --- Clean negative values ---
for col in delay_cols:
    df[col] = df[col].clip(lower=0)

# --- Focus on interesting airlines ---
top_airlines = ['DL', 'AA', 'UA', 'WN', 'B6', 'AS']
df_filtered = df[df['op_unique_carrier'].isin(top_airlines)]

# --- Compute total delays per airline ---
airline_group = df_filtered.groupby('op_unique_carrier')[delay_cols].sum()

# --- Define colors for known causes ---
colors = plt.cm.viridis_r([0.1, 0.3, 0.5, 0.7, 0.9])

# --- Create one plot per airline ---
for airline, row in airline_group.iterrows():
    plt.figure(figsize=(6, 5))

    # Total departure delay
    dep_delay_total = row['dep_delay']

    # Sum of known causes
    causal_sum_total = row[['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']].sum()

    # Missing (unexplained)
    missing = max(dep_delay_total - causal_sum_total, 0)

    # --- Plot bars ---
    plt.bar('Departure Delay', dep_delay_total, color='gray', label='Total Departure Delay')

    # Stacked known causes
    bottom = 0
    for (cause, color) in zip(['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay'], colors):
        plt.bar('Sum of Causes', row[cause], bottom=bottom, color=color, label=cause.replace('_', ' ').title())
        bottom += row[cause]

    # Add striped unexplained delay
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

    # --- Titles and labels ---
    plt.title(f"{airline}: Total Departure Delay vs. Known Delay Causes", fontsize=13)
    plt.ylabel("Total Delay (minutes)")
    plt.legend(title="Delay Cause", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show(block=True)
