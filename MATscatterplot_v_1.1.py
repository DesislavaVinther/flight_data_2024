# Import Python libraries
# pandas: for working with data tables (DataFrames)
# matplotlib.pyplot: for creating visualizations/plots
import pandas as pd
import matplotlib.pyplot as plt

# Load flight data from CSV file
# usecols: only load these specific columns to save memory (we don't need all columns)
# Loading all rows to ensure both time periods have sufficient data
usecols = ["fl_date", "dep_delay", "arr_delay", "diverted", "distance"]
df = pd.read_csv(r"C:\Users\desib\PycharmProjects\Flight data\data\flight_data_2023_2025_V_1.5.csv", usecols=usecols)

# Split the dataset into two time periods
# This allows us to compare flight patterns before and after June 25, 2024
# Comparing before/after the law implementation)
# Adf: Period 1 - flights BEFORE 2024-06-25
# Bdf: Period 2 - flights FROM 2024-06-25 and onwards
# .copy() creates independent copies so changes to one don't affect the other
Adf = df[df["fl_date"] < "2024-06-25"].copy()
Bdf = df[df["fl_date"] >= "2024-06-25"].copy()

# Function to prepare data for logarithmic plotting
# Logarithmic scales can't handle zero or negative numbers
# This function "shifts" the data to make all values positive:
# - Finds the minimum value in the column
# - Subtracts it from all values (making the minimum = 0)
# - Adds 1 (making the minimum = 1, which works with log scales)
# Example: if delays range from -20 to 100, they become 1 to 121
def shift_for_log(df, col):
    # If dataframe is empty, just return the empty column
    if df.empty:
        return df[col]
    # Shift values: subtract minimum, then add 1 to ensure all values are ≥ 1
    shifted = df[col] - df[col].min() + 1
    return shifted

# Create new columns with shifted delay values for logarithmic plotting
# We check "if not empty" to avoid errors if one time period has no data
# For each time period (A and B), we create two new columns:
if not Adf.empty:
    # Period 1: shift departure delays to positive values
    Adf["dep_shift"] = shift_for_log(Adf, "dep_delay")
    # Period 1: shift arrival delays to positive values
    Adf["arr_shift"] = shift_for_log(Adf, "arr_delay")
if not Bdf.empty:
    # Period 2: shift departure delays to positive values
    Bdf["dep_shift"] = shift_for_log(Bdf, "dep_delay")
    # Period 2: shift arrival delays to positive values
    Bdf["arr_shift"] = shift_for_log(Bdf, "arr_delay")

# Function to create and save a log-log scatter plot
# Parameters:
#   df: DataFrame containing the flight data
#   filename: name of the PNG file to save (e.g., "scatter_p1_loglog.png")
#   title: title to display on the plot (e.g., "P1 log-log")
def plot_loglog(df, filename, title):
    # Safety check: if there's no data, skip plotting and print a message
    if df.empty or df["dep_shift"].empty or df["arr_shift"].empty:
        print(f"{title}: Ingen data til log-log plot, springer over.")
        return

    # Create a new figure (blank canvas for the plot)
    plt.figure()

    # Create scatter plot with departure delay on x-axis, arrival delay on y-axis
    # s=5: small point size for better visibility with many data points
    # alpha=0.5: semi-transparent points (50%) to see overlapping patterns
    plt.scatter(df["dep_shift"], df["arr_shift"], s=5, alpha=0.5)

    # Set LOGARITHMIC scales on both axes
    # This helps visualize data that spans multiple orders of magnitude
    # (e.g., delays from 1 minute to 1000+ minutes)
    plt.xscale("log")  # X-axis: logarithmic scale
    plt.yscale("log")  # Y-axis: logarithmic scale

    # Add labels to explain what each axis represents
    plt.xlabel("Departure Delay (log)")
    plt.ylabel("Arrival Delay (log)")

    # Add title to the plot
    plt.title(title)

    # Save the plot as a PNG image file
    plt.savefig(filename)

    # Display the plot in a window (will open automatically)
    plt.show()

# Create and save the scatter plots for both time periods
# This will generate two PNG files and display two plot windows

# Plot Period 1 (before 2024-06-25)
# - Creates: scatter_p1_loglog.png
# - Shows: relationship between departure and arrival delays before the date cutoff
plot_loglog(Adf, "scatter_p1_loglog.png", "P1 log-log")

# Plot Period 2 (from 2024-06-25 onwards)
# - Creates: scatter_p2_loglog.png
# - Shows: relationship between departure and arrival delays after the date cutoff
# - Compare this with P1 to see if patterns changed over time
plot_loglog(Bdf, "scatter_p2_loglog.png", "P2 log-log")
