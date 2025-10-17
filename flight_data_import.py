import pandas as pd

# Load CSV file
df = pd.read_csv('./data/flight_data_2024.csv', low_memory=False)

# Create on-time indicator
df['on_time'] = (df['arr_delay'] <= 15).astype(int)
            # Note: treating missing arr_delay (e.g., cancelled) as 0/on_time=0 counts them as not on time.
            # If we want to exclude cancellations from the Key Performance Indicator, we can change the logic.

# Airports performance
airport_perf = (
    df.groupby(['origin', 'origin_city_name'])
      .agg(
          flights=('origin', 'size'),
          otp=('on_time', 'mean'),
          avg_arr_delay=('arr_delay', 'mean')
      )
      .query('flights >= 5000')  # Keep only airports with enough data
      .sort_values('avg_arr_delay', ascending=False)  # airports with the highest average delay appear first (worst delays on top)
)

print("=== Top airports by average delay ===")
print(airport_perf.head(15))



# Routes performance
route = (
    df.groupby(['origin', 'origin_city_name', 'dest', 'dest_city_name'])
      .agg(
          flights=('distance', 'size'),
          avg_arr_delay=('arr_delay', 'mean'),
          otp=('on_time', 'mean'),
          avg_distance=('distance', 'mean')
      )
      .query('flights >= 3000')
      .sort_values('avg_arr_delay', ascending=False)
)

print("\n=== Top routes by average delay ===")
print(route.head(15))



# Define a flight as "delayed" if arrival delay > 15 minutes
df['is_delayed'] = (df['arr_delay'] > 15).astype(int)

# Group by day_of_week and count total and delayed flights
delay_stats = (
    df.groupby('day_of_week')
      .agg(
          total_flights=('arr_delay', 'size'),
          delayed_flights=('is_delayed', 'sum')
      )
)

# Calculate the percentage of delayed flights per weekday
delay_stats['delay_rate'] = (delay_stats['delayed_flights'] / delay_stats['total_flights']) * 100

# Reset the index
delay_stats = delay_stats.reset_index()    # takes the index day_of_week and turns it back into a normal column.

# Add weekday names for readability
weekday_map = {
    1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
    4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'
}
delay_stats['weekday'] = delay_stats['day_of_week'].map(weekday_map)

# Sort by delay rate (highest first)
delay_stats = delay_stats.sort_values('delay_rate', ascending=False)

print("=== Delayed Flights by Day of Week ===")
print(delay_stats)

# Find the day with the most delays
worst_day = delay_stats.iloc[0]
print(f"\nThe day with the most delayed flights is: {worst_day['weekday']} "
      f"with {worst_day['delay_rate']:.2f}% delayed flights.")
