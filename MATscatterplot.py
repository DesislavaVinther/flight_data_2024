import pandas as pd
import matplotlib.pyplot as plt

# Indlæs kun relevante kolonner
usecols = ["fl_date", "dep_delay", "arr_delay", "diverted", "distance"]
df = pd.read_csv(r"C:\Users\madsh\Downloads\flight_data_2023_2025_V_1.3.csv", usecols=usecols, nrows=10000)

# Split datasættet uden loops
Adf = df[df["fl_date"] < "2024-06-25"].copy()
Bdf = df[df["fl_date"] >= "2024-06-25"].copy()

# Shift-funktion for log-log
def shift_for_log(df, col):
    if df.empty:
        return df[col]
    shifted = df[col] - df[col].min() + 1
    shifted = shifted[shifted > 0]
    return shifted

# Tilføj shifted kolonner
if not Adf.empty:
    Adf["dep_shift"] = shift_for_log(Adf, "dep_delay")
    Adf["arr_shift"] = shift_for_log(Adf, "arr_delay")
if not Bdf.empty:
    Bdf["dep_shift"] = shift_for_log(Bdf, "dep_delay")
    Bdf["arr_shift"] = shift_for_log(Bdf, "arr_delay")

# Log-log scatterplot funktion
def plot_loglog(df, filename, title):
    if df.empty or df["dep_shift"].empty or df["arr_shift"].empty:
        print(f"{title}: Ingen data til log-log plot, springer over.")
        return
    plt.figure()
    plt.scatter(df["dep_shift"], df["arr_shift"], s=5, alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Departure Delay (log)")
    plt.ylabel("Arrival Delay (log)")
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# Tegn scatterplots
plot_loglog(Adf, "scatter_p1_loglog.png", "P1 log-log")
plot_loglog(Bdf, "scatter_p2_loglog.png", "P2 log-log")
