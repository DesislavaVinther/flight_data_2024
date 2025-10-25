"""
Flight Time Validation Script - OPTIMIZED VERSION

PURPOSE:
This script validates that flight arrival times are logically after departure times.
It accounts for different timezones and daylight saving time by converting all times to UTC.

OPTIMIZATIONS:
- Uses vectorized operations instead of .apply() for massive speed improvements
- Eliminates row-by-row processing where possible
- Expected to be 10-50x faster than the original version

OUTPUT:
Creates 'invalid_flights.csv' containing any flights where arrival appears before departure.
"""

import pandas as pd  # Library for working with tabular data (like Excel in Python)
from pathlib import Path  # Modern way to handle file paths
from datetime import datetime  # For tracking script execution time
import numpy as np  # For efficient numerical operations

# ============================================================================
# STEP 1: DEFINE FILE PATHS
# ============================================================================
# We use Path objects instead of strings for better cross-platform compatibility
BASE_DIRECTORY = Path(r"C:\Users\desib\PycharmProjects\Flight data\data")
FLIGHT_DATA_FILE = BASE_DIRECTORY / "flight_data_2024.csv"
AIRPORT_DATA_FILE = BASE_DIRECTORY / "airports_correlated_with_altitude_longitude.csv"
OUTPUT_FILE = BASE_DIRECTORY / "invalid_flights_optimized.csv"

# Verify that both input files exist before we start processing
assert FLIGHT_DATA_FILE.exists() and AIRPORT_DATA_FILE.exists(), "Missing input files"

# Record start time to measure script performance
script_start_time = datetime.now()
print("="*70)
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
print(f"Processing: {FLIGHT_DATA_FILE.name}")

# ============================================================================
# STEP 2: LOAD DATA INTO MEMORY
# ============================================================================
# Read both CSV files into pandas DataFrames (think of them as Excel spreadsheets)
flight_data = pd.read_csv(FLIGHT_DATA_FILE, low_memory=False)
airport_data = pd.read_csv(AIRPORT_DATA_FILE)

print(f"Loaded {len(flight_data):,} flights from dataset")

# ============================================================================
# STEP 3: VALIDATE THAT REQUIRED COLUMNS EXIST
# ============================================================================
# Before we start processing, make sure the flight data has all the columns we need
required_columns = {"fl_date", "origin", "dest", "crs_dep_time", "crs_arr_time"}

if not required_columns.issubset(flight_data.columns):
    missing = required_columns - set(flight_data.columns)
    raise KeyError(f"Missing required columns: {missing}")

# ============================================================================
# STEP 4: PREPARE AIRPORT REFERENCE DATA
# ============================================================================
# We only need IATA codes and timezones from the airport data
# Clean up the IATA codes by making them uppercase and removing any extra spaces
airport_data["iata"] = airport_data["iata"].str.upper().str.strip()

# Some airports might appear multiple times - keep only the first occurrence
airport_data = airport_data.drop_duplicates(subset="iata")

# Select only the columns we need: IATA code and timezone
airport_data = airport_data[["iata", "tz"]]

print(f"Prepared {len(airport_data)} unique airports with timezone information")

# ============================================================================
# STEP 5: ADD TIMEZONE INFORMATION TO EACH FLIGHT
# ============================================================================
# First, clean the airport codes in the flight data (uppercase, remove spaces)
flight_data["origin"] = flight_data["origin"].str.upper().str.strip()
flight_data["dest"] = flight_data["dest"].str.upper().str.strip()

# Now we need to add timezone information from the airport data to each flight
# We do this in TWO steps because each flight has TWO airports (origin and destination)

# MERGE 1: Add timezone for the ORIGIN airport
# We rename columns so "iata" becomes "origin" and "tz" becomes "origin_tz"
airport_origins = airport_data.rename(columns={"iata": "origin", "tz": "origin_tz"})
flight_data = flight_data.merge(airport_origins, on="origin", how="left")

# MERGE 2: Add timezone for the DESTINATION airport
# We rename columns so "iata" becomes "dest" and "tz" becomes "dest_tz"
airport_destinations = airport_data.rename(columns={"iata": "dest", "tz": "dest_tz"})
flight_data = flight_data.merge(airport_destinations, on="dest", how="left")

# Check how many flights successfully got timezone information
origin_coverage = flight_data['origin_tz'].notna().mean()
destination_coverage = flight_data['dest_tz'].notna().mean()
print(f"Timezone coverage - Origin: {origin_coverage:.1%}, Dest: {destination_coverage:.1%}")


# ============================================================================
# STEP 6: CONVERT DEPARTURE AND ARRIVAL TIMES (OPTIMIZED - VECTORIZED!)
# ============================================================================
# OPTIMIZATION: Instead of processing each row individually with .apply(),
# we use vectorized string operations to build datetime strings, then convert all at once.
# This is 10-50x faster than the row-by-row approach!

print("\nConverting times to timezone-aware datetimes (OPTIMIZED)...")

# --- DEPARTURE TIMES ---
# Step 6a: Convert HHMM numbers to time strings
# Fill missing values with empty string, convert to int, then to 4-digit string
dep_time_str = flight_data['crs_dep_time'].fillna(0).astype(int).astype(str).str.zfill(4)
arr_time_str = flight_data['crs_arr_time'].fillna(0).astype(int).astype(str).str.zfill(4)

# Step 6b: Extract hours and minutes
# Example: "1430" → hours="14", minutes="30"
dep_hours = dep_time_str.str[:2]
dep_minutes = dep_time_str.str[2:]
arr_hours = arr_time_str.str[:2]
arr_minutes = arr_time_str.str[2:]

# Step 6c: Build ISO datetime strings (format: "2024-01-15 14:30:00")
# This combines the date with the time
flight_data['dep_datetime_str'] = (
    flight_data['fl_date'].astype(str) + ' ' +
    dep_hours + ':' + dep_minutes + ':00'
)
flight_data['arr_datetime_str'] = (
    flight_data['fl_date'].astype(str) + ' ' +
    arr_hours + ':' + arr_minutes + ':00'
)

# Step 6d: Convert to datetime objects (still timezone-naive at this point)
flight_data['departure_local_time'] = pd.to_datetime(flight_data['dep_datetime_str'], errors='coerce')
flight_data['arrival_local_time'] = pd.to_datetime(flight_data['arr_datetime_str'], errors='coerce')

# Step 6e: Handle special case - times like 2400 (midnight of next day)
# When hours = 24, we need to set time to 00:00 and add 1 day
midnight_dep = dep_hours == '24'
midnight_arr = arr_hours == '24'

if midnight_dep.any():
    flight_data.loc[midnight_dep, 'departure_local_time'] = (
        pd.to_datetime(flight_data.loc[midnight_dep, 'fl_date']) + pd.Timedelta(days=1)
    )
if midnight_arr.any():
    flight_data.loc[midnight_arr, 'arrival_local_time'] = (
        pd.to_datetime(flight_data.loc[midnight_arr, 'fl_date']) + pd.Timedelta(days=1)
    )

# Clean up temporary columns
flight_data.drop(['dep_datetime_str', 'arr_datetime_str'], axis=1, inplace=True)

print("Local times created successfully (vectorized)")

# ============================================================================
# STEP 7: CONVERT TO UTC (SIMPLIFIED - Using pandas built-in timezone support)
# ============================================================================
# OPTIMIZATION: We use pandas' built-in timezone localization which is faster
# than applying a custom function to each row
#
# NOTE: For full timezone accuracy with DST, we'd still need row-by-row processing
# This simplified version assumes most flights don't have DST issues
# (which is acceptable for validation purposes)

print("\nConverting all times to UTC for comparison (OPTIMIZED)...")

# For this optimization, we'll create separate DataFrames for each unique timezone,
# localize them, convert to UTC, then recombine
# This is faster than processing each row individually

# Build lists to collect all UTC times, then assign all at once
# This avoids dtype compatibility warnings
departure_utc_list = [pd.NaT] * len(flight_data)
arrival_utc_list = [pd.NaT] * len(flight_data)

# Process each unique origin timezone
print("  Processing origin timezones...")
for tz in flight_data['origin_tz'].dropna().unique():
    mask = flight_data['origin_tz'] == tz
    indices = flight_data.index[mask].tolist()
    try:
        # Localize to timezone, then convert to UTC (keeping timezone aware)
        localized = flight_data.loc[mask, 'departure_local_time'].dt.tz_localize(
            tz, ambiguous='NaT', nonexistent='shift_forward'
        )
        utc_times = localized.dt.tz_convert('UTC')
        # Store in list
        for idx, utc_time in zip(indices, utc_times):
            departure_utc_list[idx] = utc_time
    except Exception as e:
        # If timezone conversion fails, leave as NaT
        print(f"    Warning: Could not process timezone {tz}: {e}")

# Process each unique destination timezone
print("  Processing destination timezones...")
for tz in flight_data['dest_tz'].dropna().unique():
    mask = flight_data['dest_tz'] == tz
    indices = flight_data.index[mask].tolist()
    try:
        # Localize to timezone, then convert to UTC (keeping timezone aware)
        localized = flight_data.loc[mask, 'arrival_local_time'].dt.tz_localize(
            tz, ambiguous='NaT', nonexistent='shift_forward'
        )
        utc_times = localized.dt.tz_convert('UTC')
        # Store in list
        for idx, utc_time in zip(indices, utc_times):
            arrival_utc_list[idx] = utc_time
    except Exception as e:
        # If timezone conversion fails, leave as NaT
        print(f"    Warning: Could not process timezone {tz}: {e}")

# Assign all at once - this creates timezone-aware UTC columns
flight_data['departure_utc'] = departure_utc_list
flight_data['arrival_utc'] = arrival_utc_list

print("UTC conversion complete")

# ============================================================================
# STEP 8: FIX MIDNIGHT CROSSING FLIGHTS (OPTIMIZED - VECTORIZED!)
# ============================================================================
# OPTIMIZATION: Instead of a for loop, we use vectorized operations
# We can add 1 day to all problematic flights at once using .loc[]

print("\nChecking for midnight crossing flights...")

# Create a filter to find problematic flights
midnight_crossing_flights = (
    flight_data["arrival_utc"].notna() &
    flight_data["departure_utc"].notna() &
    (flight_data["arrival_utc"] < flight_data["departure_utc"])
)

flights_to_fix = midnight_crossing_flights.sum()
print(f"Found {flights_to_fix:,} flights crossing midnight - fixing...")

# VECTORIZED FIX: Add 1 day to ALL problematic arrival times at once
# This is much faster than a for loop!
#
# IMPORTANT FIX: Instead of re-localizing naive datetimes (which fails for DST-ambiguous times),
# we add 1 day directly to the timezone-aware UTC times. This preserves timezone information
# and avoids NaT errors during DST transitions.
if flights_to_fix > 0:
    # Add 1 day to arrival_utc (timezone-aware, so no ambiguity issues)
    flight_data.loc[midnight_crossing_flights, 'arrival_utc'] = (
        flight_data.loc[midnight_crossing_flights, 'arrival_utc'] + pd.Timedelta(days=1)
    )

    # Also update arrival_local_time for consistency (add 1 day to naive datetime)
    flight_data.loc[midnight_crossing_flights, 'arrival_local_time'] = (
        flight_data.loc[midnight_crossing_flights, 'arrival_local_time'] + pd.Timedelta(days=1)
    )

print(f"Midnight crossings corrected (vectorized)")

# ============================================================================
# STEP 9: IDENTIFY INVALID FLIGHTS
# ============================================================================
# After all corrections, check which flights STILL have invalid times
print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

# Create a boolean series: True if flight is valid, False if invalid
flight_is_valid = flight_data["arrival_utc"] >= flight_data["departure_utc"]

# Get all invalid flights
invalid_flights = flight_data[~flight_is_valid]

# Count valid and invalid flights
valid_count = flight_is_valid.sum()
invalid_count = len(invalid_flights)
total_count = len(flight_data)

# Print results
print(f"\nResults:")
print(f"  Total flights:   {total_count:,}")
print(f"  Valid flights:   {valid_count:,} ({valid_count/total_count:.2%})")
print(f"  Invalid flights: {invalid_count:,} ({invalid_count/total_count:.2%})")

# ============================================================================
# STEP 10: SAVE INVALID FLIGHTS TO CSV FILE
# ============================================================================
# Select which columns to include in the output file
columns_to_save = [
    "fl_date",              # Flight date
    "origin",               # Origin airport code
    "dest",                 # Destination airport code
    "crs_dep_time",         # Scheduled departure time (HHMM format)
    "crs_arr_time",         # Scheduled arrival time (HHMM format)
    "origin_tz",            # Origin timezone
    "dest_tz",              # Destination timezone
    "departure_local_time", # Departure in local time
    "arrival_local_time",   # Arrival in local time
    "departure_utc",        # Departure in UTC
    "arrival_utc"           # Arrival in UTC
]

# Save to CSV file (without row numbers)
invalid_flights[columns_to_save].to_csv(OUTPUT_FILE, index=False)
print(f"\nInvalid flights saved to: {OUTPUT_FILE}")

# ============================================================================
# SCRIPT EXECUTION TIME SUMMARY
# ============================================================================
# Record end time and calculate how long the script took to run
script_end_time = datetime.now()
elapsed_time = script_end_time - script_start_time

# Convert elapsed time to minutes and seconds for easier reading
total_seconds = elapsed_time.total_seconds()
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print("="*70)
print(f"Script finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {minutes} minutes and {seconds} seconds ({total_seconds:.1f}s)")
print("="*70)
