import pandas as pd
#Importer pandas biblioteket til datahåndtering og analyse.

import matplotlib
#Importer matplotlib biblioteket til visualiseringer.

matplotlib.use("TkAgg")
#Sæt backend til TkAgg for matplotlib (nødvendigt til visning i visse miljøer).

import matplotlib.pyplot as plt
#Importer pyplot submodulet til nem plotting af figurer og diagrammer.

from matplotlib.patches import Patch
#Importer Patch-klassen til at skabe farvede symboler til legend.

df = pd.read_csv(
    r"C:\Users\madsh\Downloads\flight_data_2023_2025.csv",
    low_memory=False
)
#Læs CSV-data ind i en pandas DataFrame. low_memory=False sikrer korrekt datatypehåndtering.

# --- Parse dates ---
df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')
#Konverter kolonnen 'fl_date' til datetime-format. Ukorrekte datoer bliver NaT.

df = df.dropna(subset=['fl_date'])
#Fjern rækker hvor datoen ikke kunne parses (NaT).

p1_start = "2023-07-01"
p1_end = "2024-06-30"
p2_start = "2024-07-01"
p2_end = "2025-06-30"
#Definer to tidsperioder til sammenligning.

p1 = df[(df['fl_date'] >= p1_start) & (df['fl_date'] <= p1_end)]
p2 = df[(df['fl_date'] >= p2_start) & (df['fl_date'] <= p2_end)]
#Filtrer DataFrame til de to perioder.

delay_cols = [
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]
#Liste over kolonner med forskellige forsinkelsestyper.

delay_cols = [c for c in delay_cols if c in df.columns]
#Sikre at kun kolonner, som findes i datasættet, medtages.

p1_air = p1.groupby('op_unique_carrier')[delay_cols].mean()
p2_air = p2.groupby('op_unique_carrier')[delay_cols].mean()
#grupper data per flyselskab og beregn gennemsnit af hver forsinkelsestype.

common = p1_air.index.intersection(p2_air.index)
#Find de flyselskaber, der findes i begge perioder.

p1_air = p1_air.loc[common]
p2_air = p2_air.loc[common]
#Begræns begge datasæt til de fælles flyselskaber.

p1_air['total'] = p1_air.sum(axis=1)
#Beregn total forsinkelse per flyselskab i periode 1.

sort_order = p1_air['total'].sort_values(ascending=False).index
#Sortér flyselskaber efter total forsinkelse i faldende rækkefølge.

p1_air = p1_air.loc[sort_order].drop(columns='total')
p2_air = p2_air.loc[sort_order]
#Anvend sorteringsrækkefølgen på begge perioder.

def normalize(df):
    #Normaliser rækkerne, så summen af forsinkelser per flyselskab bliver 1.
    return df.div(df.sum(axis=1), axis=0)

p1_norm = normalize(p1_air)
p2_norm = normalize(p2_air)
#Anvend normalisering på begge perioder.

x_positions = []
values = {col: [] for col in delay_cols}
spacing = 1.0  # gap between airline groups
bar_width = 0.5  # thicker bars
period_gap = 0.05  # gap between periods
#Opsæt variabler til positionsberegning og visualisering.

pos = 0
for airline in sort_order:
    # Period 1
    for col in delay_cols:
        values[col].append(p1_norm.loc[airline, col])
    x_positions.append(pos)
#Tilføj normaliserede værdier og positions for periode 1.

    # Period 2
    for col in delay_cols:
        values[col].append(p2_norm.loc[airline, col])
    x_positions.append(pos + bar_width + period_gap)
    #Tilføj normaliserede værdier og positions for periode 2.

    pos += 1 + spacing
    #Opdater position til næste flyselskabgruppe.


plt.figure(figsize=(14, 8))
#Opret figur med specificeret størrelse.

bottom = [0] * len(x_positions)
#Hold styr på hvor hver stakbar starter (for akkumulerede stakbare søjler).

base_colors = plt.cm.tab20.colors
#Farvepalette til søjlerne.

for i, col in enumerate(delay_cols):
    for j in range(len(x_positions)):
        plt.barh(
            x_positions[j],
            values[col][j],
            left=bottom[j],
            color=base_colors[i % len(base_colors)],
            height=bar_width,
            edgecolor='black',
            linewidth=1.2
        )
        bottom[j] += values[col][j]
        #Plot horisontale stakbare søjler for hver forsinkelsestype og opdater 'bottom'.

# Y-axis labels: one per airline group
plt.yticks(
    [(x_positions[i] + x_positions[i + 1]) / 2 for i in range(0, len(x_positions), 2)],
    sort_order
)
#Placér y-akse labels centralt mellem periode 1 og 2 for hver airline.

plt.xlabel("Proportion of Total Delay")
plt.title("Normalized Delay Breakdown per Airline, Both Periods (Side by Side)")
#Sæt labels og titel.

# Legend for delay types
delay_type_patches = [Patch(facecolor=base_colors[i % len(base_colors)], label=col) for i, col in enumerate(delay_cols)]
plt.legend(handles=delay_type_patches, title="Delay Type", bbox_to_anchor=(1.05, 1), loc='upper left')
#Opret legend for forsinkelstyper med farvekoder.

# Add period labels above or inside bars, avoiding overlap
for idx, x in enumerate(x_positions):
    period_label = '23–24' if idx % 2 == 0 else '24–25'
    bar_length = bottom[idx]
    # if bar is large enough, place label inside; else, place outside
    if bar_length > 0.15:
        plt.text(
            bar_length / 2,
            x,
            period_label,
            va='center',
            ha='center',
            fontsize=9,
            color='white',
            fontweight='bold'
        )
    else:
        plt.text(
            bar_length + 0.01,
            x,
            period_label,
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
    #Tilføj periode-label på hver bar, placeret indvendigt eller udenfor afhængig af barens størrelse.

plt.tight_layout()
#Automatisk tilpas layout så elementer ikke overlapper.

plt.show()
#Vis figuren.