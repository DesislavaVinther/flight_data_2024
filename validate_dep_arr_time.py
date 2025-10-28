"""
Flight Time Validation Script

PURPOSE:
This script validates that flight arrival times are logically after departure times.
It accounts for different timezones and daylight saving time by converting all times to UTC.

OUTPUT:
Creates 'invalid_flights.csv' containing any flights where arrival appears before departure.
"""

import pandas as pd  # Library for working with tabular data (like Excel in Python)
from pathlib import Path  # Modern way to handle file paths
from zoneinfo import ZoneInfo  # Python's built-in timezone handling library
from datetime import datetime  # For tracking script execution time

# ============================================================================
# STEP 1: DEFINE FILE PATHS
# ============================================================================
# We use Path objects instead of strings for better cross-platform compatibility
BASE_DIRECTORY = Path(r"C:\Users\desib\PycharmProjects\Flight data\data")
FLIGHT_DATA_FILE = BASE_DIRECTORY / "flight_data_2024.csv"
AIRPORT_DATA_FILE = BASE_DIRECTORY / "airports_correlated_with_altitude_longitude.csv"
OUTPUT_FILE = BASE_DIRECTORY / "invalid_flights_v2.csv"

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
# HELPER FUNCTION: CONVERT HHMM FORMAT TO TIMEZONE-AWARE DATETIME
# ============================================================================
def convert_hhmm_to_datetime(date, time_in_hhmm_format, timezone_name):
    """
    Converts a date and time (in HHMM format) to a timezone-aware datetime object.

    PARAMETERS:
    - date: The flight date (e.g., "2024-01-15")
    - time_in_hhmm_format: Time as a number (e.g., 1430 means 2:30 PM, 45 means 12:45 AM)
    - timezone_name: Timezone string (e.g., "America/New_York")

    RETURNS:
    - A timezone-aware datetime object, or NaT (Not a Time) if data is missing/invalid

    EXAMPLE:
    convert_hhmm_to_datetime("2024-01-15", 1430, "America/New_York")
    → 2024-01-15 14:30:00 EST (Eastern Standard Time)
    """

    # If the time or timezone is missing, we can't create a valid datetime
    if pd.isna(time_in_hhmm_format) or pd.isna(timezone_name):
        return pd.NaT  # NaT = "Not a Time" (like NaN but for dates/times)

    # Convert the HHMM number to a 4-character string
    # Examples: 45 → "0045" (12:45 AM), 1430 → "1430" (2:30 PM)
    time_string = str(int(time_in_hhmm_format)).zfill(4)

    # Extract hours and minutes from the string
    hours = int(time_string[:2])      # First 2 characters
    minutes = int(time_string[2:])    # Last 2 characters

    # Special case: 2400 means midnight of the NEXT day
    # We change it to 0000 (00:00) and add 1 day to the date
    if hours == 24:
        hours = 0
        date = pd.to_datetime(date) + pd.Timedelta(days=1)

    # Create a basic timestamp by combining date and time
    timestamp = pd.to_datetime(date) + pd.Timedelta(hours=hours, minutes=minutes)

    # Now add timezone information to make it "timezone-aware"
    try:
        # tz_localize attaches timezone info to the timestamp
        # ambiguous="NaT": When clocks "fall back" for DST, 1:30 AM happens twice - mark as NaT
        # nonexistent="shift_forward": When clocks "spring forward", 2:30 AM doesn't exist - shift forward
        return timestamp.tz_localize(ZoneInfo(timezone_name),
                                     ambiguous="NaT",
                                     nonexistent="shift_forward")
    except Exception:
        # If anything goes wrong (invalid timezone, etc.), return NaT
        return pd.NaT


# ============================================================================
# STEP 6: CONVERT DEPARTURE AND ARRIVAL TIMES TO TIMEZONE-AWARE DATETIMES
# ============================================================================
print("\nConverting times to timezone-aware datetimes...")

# Convert DEPARTURE times
# For each row in the dataset, we call our conversion function
# .apply(axis=1) means "apply this function to each ROW"
flight_data["departure_local_time"] = flight_data.apply(
    lambda row: convert_hhmm_to_datetime(row["fl_date"], row["crs_dep_time"], row["origin_tz"]),
    axis=1
)

# Convert ARRIVAL times
# Same process, but for arrival times at the destination
flight_data["arrival_local_time"] = flight_data.apply(
    lambda row: convert_hhmm_to_datetime(row["fl_date"], row["crs_arr_time"], row["dest_tz"]),
    axis=1
)

print("Local times created successfully")

# ============================================================================
# STEP 7: CONVERT ALL TIMES TO UTC (COORDINATED UNIVERSAL TIME)
# ============================================================================
# WHY? Because we can't compare times in different timezones directly!
# Example: A flight departing at 3 PM in NYC and arriving at 3 PM in LA
#          Actually arrives LATER (6 PM NYC time) because of the 3-hour time difference
#
# UTC is the universal reference time - like a common language for all timezones
print("\nConverting all times to UTC for comparison...")

# Convert departure times to UTC
flight_data["departure_utc"] = flight_data["departure_local_time"].apply(
    lambda time_value: time_value.astimezone(ZoneInfo("UTC")) if pd.notna(time_value) else pd.NaT
)

# Convert arrival times to UTC
flight_data["arrival_utc"] = flight_data["arrival_local_time"].apply(
    lambda time_value: time_value.astimezone(ZoneInfo("UTC")) if pd.notna(time_value) else pd.NaT
)

print("UTC conversion complete")

# ============================================================================
# STEP 8: FIX MIDNIGHT CROSSING FLIGHTS
# ============================================================================
# PROBLEM: Some flights depart late at night and arrive after midnight the next day
# Example: Depart 11:30 PM on Jan 15, Arrive 1:30 AM on Jan 16
#          Our code assumes same date, so arrival appears 22 hours EARLIER than departure!
#
# SOLUTION: Find flights where arrival < departure (impossible!) and add 1 day to arrival
print("\nChecking for midnight crossing flights...")

# Create a filter to find problematic flights
# We look for flights where arrival appears BEFORE departure (which is impossible)
midnight_crossing_flights = (
    flight_data["arrival_utc"].notna() &           # Arrival time exists, AND
    flight_data["departure_utc"].notna() &         # Departure time exists, AND
    (flight_data["arrival_utc"] < flight_data["departure_utc"])  # Arrival before departure (wrong!)
)

flights_to_fix = midnight_crossing_flights.sum()
print(f"Found {flights_to_fix:,} flights crossing midnight - fixing...")

# Fix each problematic flight by adding 1 day to the arrival time
for flight_index in flight_data[midnight_crossing_flights].index:
    # Get the current (incorrect) arrival time
    old_arrival_time = flight_data.at[flight_index, "arrival_local_time"]

    # Add 1 day to correct it
    corrected_arrival_time = old_arrival_time + pd.Timedelta(days=1)

    # Update both the local time and UTC time
    flight_data.at[flight_index, "arrival_local_time"] = corrected_arrival_time
    flight_data.at[flight_index, "arrival_utc"] = corrected_arrival_time.astimezone(ZoneInfo("UTC"))

print(f"Midnight crossings corrected")

# ============================================================================
# STEP 9: IDENTIFY INVALID FLIGHTS
# ============================================================================
# After all corrections, check which flights STILL have invalid times
# Valid flight: arrival_utc >= departure_utc (arrival is at or after departure)
# Invalid flight: arrival_utc < departure_utc (arrival is before departure - impossible!)
print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

# Create a boolean series: True if flight is valid, False if invalid
flight_is_valid = flight_data["arrival_utc"] >= flight_data["departure_utc"]

# Get all invalid flights (where flight_is_valid is False)
# The ~ symbol means "NOT" - it flips True to False and False to True
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