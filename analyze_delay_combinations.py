"""
Analyze delay type combinations for severely delayed flights (arr_delay > 180 minutes).

This script analyzes which combinations of delay reasons occur most frequently
for flights delayed more than 3 hours, comparing two time periods:
- Period 1: July 2023 - June 2024
- Period 2: July 2024 - June 2025

The 5 delay types analyzed are:
- carrier_delay: Delays caused by the airline (maintenance, crew issues, etc.)
- weather_delay: Delays caused by weather conditions
- nas_delay: Delays caused by the National Aviation System (air traffic control)
- security_delay: Delays caused by security issues
- late_aircraft_delay: Delays caused by the same aircraft arriving late from a previous flight
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import webbrowser  # with webbrowser, Python creates the file and opens it for automatically. Much more convenient once the script is done.

print("=" * 80) # visual text effect
print("DELAY COMBINATION ANALYSIS FOR SEVERELY DELAYED FLIGHTS (>180 minutes)")
print("=" * 80)

# Load the data
print("\nLoading flight data...")
df = pd.read_csv(r'C:\Users\desib\PycharmProjects\Flight data\data\flight_data_2023_2025_V_1.0.csv',
                 low_memory=False)
print(f"Total flights loaded: {len(df):,}")             #  f-strings are the modern, recommended way to format strings in Python. Easier to read, more powerful & faster

# Filter for severely delayed flights (arr_delay > 180 minutes)
print("\nFiltering for flights with arr_delay > 180 minutes...")
# Create a new DataFrame containing ONLY flights with arrival delay > 180 minutes (3 hours)
# How this works:
#   1. df['arr_delay'] > 180  →  Creates True/False for each row (True if delay > 180)
#   2. df[...]  →  Keeps only rows where the condition is True (filters the data)
#   3. .copy()  →  Creates an independent copy so changes don't affect the original df
severely_delayed = df[df['arr_delay'] > 180].copy()
# This filtering technique is called boolean indexing and it's one of the most powerful features of pandas.
print(f"Severely delayed flights (>180 min): {len(severely_delayed):,}")
print(f"Percentage of all flights: {100 * len(severely_delayed) / len(df):.3f}%")

# Convert fl_date to datetime
severely_delayed['fl_date'] = pd.to_datetime(severely_delayed['fl_date'])

# Define the two periods
period1_start = pd.Timestamp('2023-07-01')
period1_end = pd.Timestamp('2024-06-30')
period2_start = pd.Timestamp('2024-07-01')
period2_end = pd.Timestamp('2025-06-30')

# Split into two periods
period1 = severely_delayed[(severely_delayed['fl_date'] >= period1_start) &
                          (severely_delayed['fl_date'] <= period1_end)].copy()
period2 = severely_delayed[(severely_delayed['fl_date'] >= period2_start) &
                          (severely_delayed['fl_date'] <= period2_end)].copy()

print(f"\nPeriod 1 (2023-07-01 to 2024-06-30): {len(period1):,} flights")
print(f"Period 2 (2024-07-01 to 2025-06-30): {len(period2):,} flights")

# Define delay type columns
delay_cols = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']

# Short names for visualization
delay_short_names = {
    'carrier_delay': 'Carrier',
    'weather_delay': 'Weather',
    'nas_delay': 'NAS',
    'security_delay': 'Security',
    'late_aircraft_delay': 'Late Aircraft'
}

def identify_delay_combination(row):
    """
    Identify which delay types are present (have values > 0) for a flight.
    Returns a string describing the combination, e.g., "Carrier + Weather".

    This function takes one row (one flight) from the dataset and checks which
    types of delays contributed to that flight's total delay.
    """

    # Create an empty list to store the names of delay types that are present
    # Add delay types to this list as we find them
    active_delays = []

    # Loop through each delay type column (carrier_delay, weather_delay, etc.)
    # 'delay_cols' is a list defined earlier with all 5 delay type column names
    for col in delay_cols:
        # Check two conditions for this delay type:
        # 1. pd.notna(row[col]) - Is there a value? (not NaN/missing)
        # 2. row[col] > 0 - Is the delay value greater than 0 minutes?
        # Both must be True for this delay type to count as "active"
        if pd.notna(row[col]) and row[col] > 0:
            # If both conditions are met, add the short name of this delay type
            # to our list. For example, 'carrier_delay' becomes 'Carrier'
            active_delays.append(delay_short_names[col])

    # Now decide what to return based on how many delay types we found:

    # Case 1: No delay types found (empty list)
    # This happens when delay data is missing or all delay values are 0
    if len(active_delays) == 0:
        return "No delay breakdown"

    # Case 2: Exactly one delay type found
    # Return just that delay type name, e.g., "Carrier"
    elif len(active_delays) == 1:
        return active_delays[0]

    # Case 3: Multiple delay types found (2 or more)
    # Combine them into one string with " + " between each type
    # sorted() ensures they appear in alphabetical order for consistency
    # join() takes a list and combines it into a single string
    # Example: ['Weather', 'Carrier'] becomes "Carrier + Weather"
    else:
        return " + ".join(sorted(active_delays))

# Identify delay combinations for each period
print("\nIdentifying delay combinations...")
period1['delay_combination'] = period1.apply(identify_delay_combination, axis=1)
period2['delay_combination'] = period2.apply(identify_delay_combination, axis=1)

# Count combinations
combo_counts_p1 = period1['delay_combination'].value_counts()
combo_counts_p2 = period2['delay_combination'].value_counts()

print("\n" + "=" * 80)
print("PERIOD 1 (2023-07-01 to 2024-06-30) - Top 5 Delay Combinations")
print("=" * 80)
for i, (combo, count) in enumerate(combo_counts_p1.head(5).items(), 1):
    pct = 100 * count / len(period1)
    print(f"{i:2d}. {combo:40s} {count:>8,} flights ({pct:5.2f}%)")

print("\n" + "=" * 80)
print("PERIOD 2 (2024-07-01 to 2025-06-30) - Top 5 Delay Combinations")
print("=" * 80)
for i, (combo, count) in enumerate(combo_counts_p2.head(5).items(), 1):
    pct = 100 * count / len(period2)
    print(f"{i:2d}. {combo:40s} {count:>8,} flights ({pct:5.2f}%)")

# Create visualization
print("\nCreating visualizations...")  # Think of \n as an "invisible Enter key" in the terminal text! li

# Get top 5 combinations for each period
top_n = 5
top_combos_p1 = combo_counts_p1.head(top_n)
top_combos_p2 = combo_counts_p2.head(top_n)

# Calculate percentages
top_combos_p1_pct = (top_combos_p1 / len(period1)) * 100
top_combos_p2_pct = (top_combos_p2 / len(period2)) * 100

# Create consistent color mapping for all unique delay combinations
# Get all unique combinations from both periods
all_combos = list(set(top_combos_p1.index) | set(top_combos_p2.index))

# Define a color palette with distinct colors
color_palette = [
    '#3498DB',  # Blue
    '#E74C3C',  # Red
    '#2ECC71',  # Green
    '#F39C12',  # Orange
    '#9B59B6',  # Purple
    '#1ABC9C',  # Teal
    '#E67E22',  # Dark Orange
    '#34495E',  # Dark Gray
    '#16A085',  # Dark Teal
    '#D35400',  # Pumpkin
]

# Create a dictionary mapping each combination to a specific color
combo_color_map = {}
for i, combo in enumerate(sorted(all_combos)):
    combo_color_map[combo] = color_palette[i % len(color_palette)]

# Assign colors based on the combination name (consistent across both periods)
colors_p1 = [combo_color_map[combo] for combo in top_combos_p1.index]
colors_p2 = [combo_color_map[combo] for combo in top_combos_p2.index]

# Print the color mapping for reference
print("\nColor mapping for delay combinations:")
for combo in sorted(all_combos):
    print(f"  {combo:40s} -> {combo_color_map[combo]}")

# Create subplots (two horizontal bar charts side by side)
# 'fig' will be the complete visualization with two charts
fig = make_subplots(
    rows=1, cols=2,  # Create a grid: 1 row, 2 columns (side by side)
    # Titles that appear above each chart
    # <br> is HTML for line break (same as \n but for web pages)
    subplot_titles=(f'Period 1: July 2023 - June 2024<br>({len(period1):,} severely delayed flights)',
                   f'Period 2: July 2024 - June 2025<br>({len(period2):,} severely delayed flights)'),
    horizontal_spacing=0.15  # Space between the two charts (0.15 = 15% of total width)
)

# Period 1 bar chart
# add_trace() adds one chart (a "trace") to the figure
fig.add_trace(
    go.Bar(  # Create a bar chart using Plotly's Bar object
        # y-axis: The delay combination names (e.g., "Carrier + Weather")
        # [::-1] is Python's way to REVERSE a list (flips it upside down)
        # We reverse so the #1 combo appears at the TOP of the chart
        y=top_combos_p1.index[::-1],

        # x-axis: The percentage values (how common each combination is)
        x=top_combos_p1_pct.values[::-1],

        orientation='h',  # 'h' = horizontal bars (bars go left-to-right, not up-down)

        # Set the colors for each bar
        marker=dict(color=colors_p1[::-1]),

        # Text labels that appear next to each bar showing count and percentage
        # List comprehension creates a label for each bar
        # zip() pairs up counts with percentages so we can use both
        text=[f"{count:,}<br>({pct:.1f}%)" for count, pct in zip(top_combos_p1.values[::-1], top_combos_p1_pct.values[::-1])],

        textposition='outside',  # Put text labels outside the bars (to the right)

        # What appears when you hover your mouse over a bar
        # %{y} = the y-value (combination name), %{x} = the x-value (percentage)
        hovertemplate='<b>%{y}</b><br>Flights: %{x:.1f}%<extra></extra>',

        showlegend=False  # Don't show this in a legend (we don't need one)
    ),
    row=1, col=1  # Place this chart in row 1, column 1 (the LEFT chart)
)

# Period 2 bar chart
fig.add_trace(
    go.Bar(
        y=top_combos_p2.index[::-1],
        x=top_combos_p2_pct.values[::-1],
        orientation='h',
        marker=dict(color=colors_p2[::-1]),
        text=[f"{count:,}<br>({pct:.1f}%)" for count, pct in zip(top_combos_p2.values[::-1], top_combos_p2_pct.values[::-1])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Flights: %{x:.1f}%<extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_xaxes(title_text="Percentage of Severely Delayed Flights (%)", row=1, col=1)
fig.update_xaxes(title_text="Percentage of Severely Delayed Flights (%)", row=1, col=2)

fig.update_layout(
    title_text="Most Common Delay Type Combinations for Severely Delayed Flights (arr_delay > 180 min)",
    title_font_size=16,
    height=500,
    width=1800,
    showlegend=False
)

# Save the visualization
output_file = r'C:\Users\desib\PycharmProjects\Flight data\delay_combinations_analysis.html'
fig.write_html(output_file)
print(f"\nVisualization saved to: {output_file}")

# Open the visualization in the default browser
webbrowser.open(output_file)
print("Opening visualization in browser...")

# Additional analysis: Compare periods
print("\n" + "=" * 80)
print("COMPARISON BETWEEN PERIODS")
print("=" * 80)

# Get common combinations between both periods
common_combos = set(combo_counts_p1.head(5).index) & set(combo_counts_p2.head(5).index)

if common_combos:
    print("\nTop combinations appearing in both periods:")
    for combo in sorted(common_combos):
        p1_count = combo_counts_p1.get(combo, 0)
        p2_count = combo_counts_p2.get(combo, 0)
        p1_pct = 100 * p1_count / len(period1)
        p2_pct = 100 * p2_count / len(period2)
        change = p2_pct - p1_pct
        print(f"\n{combo}:")
        print(f"  Period 1: {p1_count:,} ({p1_pct:.2f}%)")
        print(f"  Period 2: {p2_count:,} ({p2_pct:.2f}%)")
        print(f"  Change: {change:+.2f} percentage points")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for period_name, period_data in [("Period 1", period1), ("Period 2", period2)]:
    print(f"\n{period_name}:")

    # Count how many delay types are present on average
    def count_delay_types(row):
        return sum(1 for col in delay_cols if pd.notna(row[col]) and row[col] > 0)

    period_data['num_delay_types'] = period_data.apply(count_delay_types, axis=1)

    print(f"  Average number of delay types per flight: {period_data['num_delay_types'].mean():.2f}")
    print(f"  Flights with no delay breakdown: {(period_data['num_delay_types'] == 0).sum():,}")
    print(f"  Flights with 1 delay type: {(period_data['num_delay_types'] == 1).sum():,}")
    print(f"  Flights with 2 delay types: {(period_data['num_delay_types'] == 2).sum():,}")
    print(f"  Flights with 3+ delay types: {(period_data['num_delay_types'] >= 3).sum():,}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
