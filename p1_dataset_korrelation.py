import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Forces interactive backend for PyCharm
import matplotlib.pyplot as plt
import seaborn as sns

# pandas (pd) bruges til at håndtere data i tabelform (DataFrames).
# matplotlib og seaborn bruges til at lave grafer.

df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2024.csv",
    low_memory=False
)
# Læser en CSV-fil med flydata.

# Vælg kun kolonner, der giver mening for ankomstforsinkelse
relevant_cols = [
    'arr_delay',          # ankomstforsinkelse
    'dep_delay',          # afgangsforsinkelse
    'carrier_delay',      # forsinkelse pga. flyselskab
    'weather_delay',      # forsinkelse pga. vejr
    'nas_delay',          # forsinkelse pga. lufttrafiksystem
    'security_delay',     # forsinkelse pga. sikkerhed
    'late_aircraft_delay',# forsinkelse pga. forsinket fly
    'taxi_out',           # tid brugt på taxi før start
    'taxi_in',            # tid brugt på taxi efter landing
    'air_time',           # tid i luften
    'distance',           # distance mellem lufthavne
    'actual_elapsed_time',# faktisk flyvetid
    'crs_elapsed_time'    # planlagt flyvetid
]

# Behold kun de kolonner, der findes i datasættet
relevant_cols = [col for col in relevant_cols if col in df.columns]
df_subset = df[relevant_cols]

# Beregn korrelationer kun for disse kolonner
corr = df_subset.corr(numeric_only=True)
corr_arr = corr['arr_delay'].dropna().sort_values(ascending=False)

# df.corr() beregner korrelationen mellem de udvalgte numeriske kolonner.
# corr['arr_delay'] henter kun korrelationen for ankomstforsinkelsen.
# .dropna() fjerner evt. tomme værdier.
# .sort_values(ascending=False) sorterer, så de stærkeste positive sammenhænge vises øverst.

plt.figure(figsize=(8, 5))
sns.barplot(
    x=corr_arr.values,
    y=corr_arr.index,
    hue=corr_arr.index,
    palette='coolwarm',
    legend=False
)
plt.title("Correlation with Arrival Delay", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Variable")
plt.tight_layout()
plt.show(block=True)

# Her tegnes et søjlediagram (vandret) med seaborn:
# x=corr_arr.values: korrelationskoefficienterne.
# y=corr_arr.index: de tilsvarende variabelnavne.
# palette='coolwarm': farveskema, hvor røde/blå farver viser styrke/retning på korrelationen.
# plt.tight_layout() sørger for at teksten ikke overlapper.
# plt.show(block=True) viser grafen (og holder den åben indtil du lukker vinduet).

#TEST(am)

