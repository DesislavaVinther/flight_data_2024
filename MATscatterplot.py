import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Indlæs kun relevante kolonner
usecols = ["fl_date", "dep_delay", "arr_delay", "diverted", "distance"]
df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2023_2025_V_1.3.csv",
    usecols=usecols
)

# Datatype
df["fl_date"] = pd.to_datetime(df["fl_date"])

# Fjern kun punkter <180 i begge akser
df = df[~((df["dep_delay"] < 180) & (df["arr_delay"] < 180))]

# Periode-split
P1 = df[df["fl_date"] < "2024-06-25"].copy()
P2 = df[df["fl_date"] >= "2024-06-25"].copy()

# Shift-funktion til log-log
def shift_for_log(df, col):
    return df[col] - df[col].min() + 1

# Shift kolonner
for d in [P1, P2]:
    d["dep_shift"] = shift_for_log(d, "dep_delay")
    d["arr_shift"] = shift_for_log(d, "arr_delay")

# 180-min grænser efter shift
p1_dep_thr = 180 - P1["dep_delay"].min() + 1
p1_arr_thr = 180 - P1["arr_delay"].min() + 1

p2_dep_thr = 180 - P2["dep_delay"].min() + 1
p2_arr_thr = 180 - P2["arr_delay"].min() + 1

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# ---------- P1 ----------
axes[0].scatter(
    P1["dep_shift"],
    P1["arr_shift"],
    s=6,
    alpha=0.6
)

axes[0].axvline(p1_dep_thr, linestyle="--", linewidth=1)
axes[0].axhline(p1_arr_thr, linestyle="--", linewidth=1)

# Rød markering nederst til venstre
axes[0].add_patch(
    Rectangle(
        (100, 100),
        max(p1_dep_thr - 100, 0),
        max(p1_arr_thr - 100, 0),
        color="red",
        alpha=0.15
    )
)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(left=100)
axes[0].set_ylim(bottom=100)
axes[0].set_xlabel("Departure Delay (log)")
axes[0].set_ylabel("Arrival Delay (log)")

# ---------- P2 ----------
axes[1].scatter(
    P2["dep_shift"],
    P2["arr_shift"],
    s=6,
    alpha=0.6,
    color="tab:orange"
)

axes[1].axvline(p2_dep_thr, linestyle="--", linewidth=1)
axes[1].axhline(p2_arr_thr, linestyle="--", linewidth=1)

axes[1].add_patch(
    Rectangle(
        (100, 100),
        max(p2_dep_thr - 100, 0),
        max(p2_arr_thr - 100, 0),
        color="red",
        alpha=0.15
    )
)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlim(left=100)
axes[1].set_ylim(bottom=100)
axes[1].set_xlabel("Departure Delay (log)")

plt.tight_layout()
plt.savefig("scatter_loglog_P1_P2_filtered_low_delays_100start.png", dpi=300)
plt.show()

# ---------- Nedre højre kvadrant ----------
# definer kvadranten: dep >=180, arr <180
def lower_right_share(df):
    total = len(df)
    lr = df[(df["dep_delay"] >= 180) & (df["arr_delay"] < 180)]
    count = len(lr)
    share = count / total if total > 0 else 0
    return count, share

p1_count, p1_share = lower_right_share(P1)
p2_count, p2_share = lower_right_share(P2)

print("Nedre højre kvadrant")
print(f"P1: {p1_count} ud af {len(P1)} flights ({p1_share:.2%})")
print(f"P2: {p2_count} ud af {len(P2)} flights ({p2_share:.2%})")

# ---------- Øvre kvadrant ----------
# definer kvadranten: dep >=180, arr <180
def upper_left_share(df):
    total = len(df)
    lr = df[(df["dep_delay"] < 180) & (df["arr_delay"] >= 180)]
    count = len(lr)
    share = count / total if total > 0 else 0
    return count, share

p1_count, p1_share = upper_left_share(P1)
p2_count, p2_share = upper_left_share(P2)

print("Øvre venstre kvadrant")
print(f"P1: {p1_count} ud af {len(P1)} flights ({p1_share:.2%})")
print(f"P2: {p2_count} ud af {len(P2)} flights ({p2_share:.2%})")