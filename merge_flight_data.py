# Merge Multiple Flight Data CSV Files

# This script combines 13 CSV files spanning from July 2023 to June 2025 into a single file.
# The files are merged in chronological order to create one comprehensive dataset.

import pandas as pd
from pathlib import Path

# Define base directory for the project
BASE = Path(r"C:\Users\desib\PycharmProjects\Flight data")

# List all CSV files in the exact order they should be merged
# These represent monthly flight data from July 2023 through June 2025
csv_files = [
    BASE / "data" / "month_data" / "2023_juli.csv",
    BASE / "data" / "month_data" / "2023_august.csv",
    BASE / "data" / "month_data" / "2023_september.csv",
    BASE / "data" / "month_data" / "2023_oktober.csv",
    BASE / "data" / "month_data" / "2023_november.csv",
    BASE / "data" / "month_data" / "2023_december.csv",
    BASE / "data" / "flight_data_2024.csv",
    BASE / "data" / "month_data" / "2025_januar.csv",
    BASE / "data" / "month_data" / "2025_februar.csv",
    BASE / "data" / "month_data" / "2025_marts.csv",
    BASE / "data" / "month_data" / "2025_april.csv",
    BASE / "data" / "month_data" / "2025_maj.csv",
    BASE / "data" / "month_data" / "2025_juni.csv",
]

# Create an empty list to store each DataFrame (data table)
# I read each CSV file and add it to this list
dataframes = []

# Loop through each file and read it
for i, file_path in enumerate(csv_files, start=1):
    # Check if the file exists before trying to read it
    if not file_path.exists():
        print(f"WARNING: File not found - {file_path.name}")
        continue

    print(f"[{i}/{len(csv_files)}] Reading {file_path.name}...", end=" ")

    try:
        # Read the CSV file into a pandas DataFrame
        # low_memory=False helps with large files to avoid data type warnings
        df = pd.read_csv(file_path, low_memory=False)

        # Add this DataFrame to my list
        dataframes.append(df)

        # Print how many rows (flights) were in this file
        print(f"OK ({len(df):,} rows)")

    except Exception as e:
        # If there's an error reading the file, print it but continue
        print(f"ERROR: {e}")

# Check if we successfully read any files
if not dataframes:
    print("\nERROR: No files were successfully read. Exiting.")
    exit(1)

print(f"\n{'='*60}")
print("Merging all dataframes together...")

# Stack all DataFrames on top of each other
# ignore_index=True creates a new continuous index starting from 0
merged_df = pd.concat(dataframes, ignore_index=True)

print(f"DONE - Merge complete!")
print(f"  Total rows in merged dataset: {len(merged_df):,}")
print(f"  Total columns: {len(merged_df.columns)}")
print(f"{'='*60}\n")

# Define output file path
output_file = BASE / "data" / "flight_data_2023_2025_merged.csv"

# Save the merged DataFrame to a new CSV file
print(f"Saving merged data to: {output_file.name}...")
merged_df.to_csv(output_file, index=False)

# Check the file size to confirm it was saved
file_size_mb = output_file.stat().st_size / (1024 * 1024)
print(f"DONE - File saved successfully! ({file_size_mb:.1f} MB)\n")

print("SUCCESS - Merge operation completed!")
print(f"   Output file: {output_file}")
