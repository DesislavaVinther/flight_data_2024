import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Indlæs data
# ===============================
columns = ["binary_delay","op_unique_carrier","year","month"]
df = pd.read_csv("C:/Users/desib/PycharmProjects/Flight data/data/flight_data_2023_2025_V_1.3.csv", low_memory=False, usecols=columns)

# ===============================
# 2. Funktion til at filtrere perioder
# ===============================
def get_period(df, start_year, start_month, end_year, end_month):
    """
    Returnerer rækker fra df, der ligger mellem start_year/start_month og end_year/end_month.
    """
    return df[
        ((df["year"] > start_year) | ((df["year"] == start_year) & (df["month"] >= start_month))) &
        ((df["year"] < end_year) | ((df["year"] == end_year) & (df["month"] <= end_month)))
    ]

# ===============================
# 3. Definér perioder
# ===============================
p1 = get_period(df, 2023, 7, 2024, 6)
p2 = get_period(df, 2024, 7, 2025, 6)

# ===============================
# 4. Funktion til at køre logistisk regression
# ===============================
def run_logit_period(df_period, period_name):
    print(f"\n===== Resultater for {period_name} =====")

    # ---- 4a. One-hot encode flyselskaber
    X = pd.get_dummies(df_period[["op_unique_carrier"]], drop_first=False)

    # ---- 4b. Afhængig variabel
    y = df_period["binary_delay"]

    # ---- 4c. Train/test split med stratify for at håndtere ubalancer
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ---- 4d. Initialiser og træn modellen
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # ---- 4e. Forudsigelser
    y_pred = model.predict(X_test)

    # ---- 4f. Evaluering
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---- 4g. Koeficienter pr. carrier
    coef_df = pd.DataFrame({
        "carrier": X.columns,
        "coef": model.coef_[0]
    }).sort_values(by="coef", ascending=False)
    print("\nTop koefficienter (log-odds) for hver carrier:")
    print(coef_df)

    # ---- 4h. Konverter til Odds Ratios og Sandsynligheder
    print("\n" + "="*80)
    print("UDVIDET ANALYSE: Koefficienter -> Odds Ratios -> Sandsynligheder")
    print("="*80)

    # Beregn faktisk forsinkelsesrate i perioden (baseline)
    actual_delay_rate = y.mean()
    print(f"\n[INFO] Baseline forsinkelsesrate (180+ min) i denne periode: {actual_delay_rate:.4f} ({actual_delay_rate*100:.2f}%)")
    print(f"       Total antal flyvninger: {len(y):,}")
    print(f"       Antal forsinkede (180+ min): {y.sum():,}")
    print(f"       Antal ikke-forsinkede: {(~y.astype(bool)).sum():,}")

    # Beregn odds ratios (e^koefficient)
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])

    # For hvert luftselskab: beregn sandsynlighed
    # Vi laver en "prototype" flyvning for hvert luftselskab
    probabilities = []
    for carrier in X.columns:
        # Lav en dummy observation hvor kun dette luftselskab er aktiv
        dummy_flight = pd.DataFrame(0, index=[0], columns=X.columns)
        dummy_flight[carrier] = 1

        # Predict probability for denne flyvning
        prob = model.predict_proba(dummy_flight)[0, 1]  # Sandsynlighed for class 1 (delay)
        probabilities.append(prob)

    coef_df["probability"] = probabilities
    coef_df["probability_pct"] = coef_df["probability"] * 100

    # Rens carrier navne (fjern "op_unique_carrier_" prefix)
    coef_df["carrier_code"] = coef_df["carrier"].str.replace("op_unique_carrier_", "")

    # Sorter efter sandsynlighed (højest til lavest)
    coef_df_sorted = coef_df.sort_values(by="probability", ascending=False)

    # Print pæn tabel
    print("\n[ANALYSE] FULD ANALYSE AF LUFTSELSKABER:")
    print("-" * 100)
    print(f"{'Carrier':<8} {'Log-Odds':<10} {'Odds Ratio':<12} {'Sandsynlighed':<15} {'Fortolkning'}")
    print("-" * 100)

    for _, row in coef_df_sorted.iterrows():
        carrier_code = row["carrier_code"]
        log_odds = row["coef"]
        odds_ratio = row["odds_ratio"]
        prob = row["probability"]
        prob_pct = row["probability_pct"]

        # Fortolkning
        if odds_ratio > 1.5:
            interpretation = "[!!!] Hoej risiko"
        elif odds_ratio > 1.1:
            interpretation = "[!] Over gennemsnit"
        elif odds_ratio > 0.9:
            interpretation = "[=] Omkring gennemsnit"
        elif odds_ratio > 0.7:
            interpretation = "[+] Under gennemsnit"
        else:
            interpretation = "[++] Lav risiko"

        print(f"{carrier_code:<8} {log_odds:>9.3f}  {odds_ratio:>11.3f}  {prob_pct:>13.2f}%  {interpretation}")

    print("-" * 100)

    # Sammenligning: højeste vs laveste
    best_carrier = coef_df_sorted.iloc[-1]
    worst_carrier = coef_df_sorted.iloc[0]

    print(f"\n[BEDSTE] {best_carrier['carrier_code']} med {best_carrier['probability_pct']:.2f}% sandsynlighed")
    print(f"[VAERSTE] {worst_carrier['carrier_code']} med {worst_carrier['probability_pct']:.2f}% sandsynlighed")
    print(f"[FORSKEL] {worst_carrier['probability_pct'] - best_carrier['probability_pct']:.2f} procentpoint")
    print(f"          (Odds ratio: {worst_carrier['odds_ratio'] / best_carrier['odds_ratio']:.2f}x hoejere risiko)")

    return coef_df_sorted  # Returnerer dataframe så vi kan sammenligne perioder senere

# ===============================
# 5. Kør modellen for begge perioder
# ===============================
results_p1 = run_logit_period(p1, "Periode 1: juli 2023 - juni 2024")
results_p2 = run_logit_period(p2, "Periode 2: juli 2024 - juni 2025")

# ===============================
# 6. Sammenlign de to perioder
# ===============================
print("\n" + "="*80)
print("SAMMENLIGNING MELLEM PERIODE 1 OG PERIODE 2")
print("="*80)

# Merge de to resultater
comparison = results_p1[["carrier_code", "probability_pct"]].merge(
    results_p2[["carrier_code", "probability_pct"]],
    on="carrier_code",
    suffixes=("_p1", "_p2")
)

# Beregn ændring
comparison["change"] = comparison["probability_pct_p2"] - comparison["probability_pct_p1"]
comparison["change_pct"] = (comparison["change"] / comparison["probability_pct_p1"]) * 100

# Sorter efter største ændring
comparison_sorted = comparison.sort_values(by="change", ascending=False)

print("\n[INFO] AENDRINGER I SANDSYNLIGHED FOR 180+ MIN FORSINKELSE:")
print("-" * 100)
print(f"{'Carrier':<8} {'Periode 1':<12} {'Periode 2':<12} {'Aendring':<15} {'Trend'}")
print("-" * 100)

for _, row in comparison_sorted.iterrows():
    carrier = row["carrier_code"]
    p1_prob = row["probability_pct_p1"]
    p2_prob = row["probability_pct_p2"]
    change = row["change"]

    # Trend indikator
    if change > 0.1:
        trend = f"[UP] FORVAERRET (+{change:.2f}pp)"
    elif change < -0.1:
        trend = f"[DOWN] FORBEDRET ({change:.2f}pp)"
    else:
        trend = f"[=] Stabil ({change:+.2f}pp)"

    print(f"{carrier:<8} {p1_prob:>10.2f}%  {p2_prob:>10.2f}%  {change:>13.2f}pp  {trend}")

print("-" * 100)

# Top forbedringer og forværringer
print("\n[+++] TOP 3 FORBEDRINGER (stoerste fald i forsinkelsesrate):")
best_improvements = comparison_sorted.tail(3)
for i, (_, row) in enumerate(best_improvements.iterrows(), 1):
    print(f"   {i}. {row['carrier_code']}: {row['probability_pct_p1']:.2f}% -> {row['probability_pct_p2']:.2f}% "
          f"({row['change']:.2f}pp)")

print("\n[---] TOP 3 FORVAERRINGER (stoerste stigning i forsinkelsesrate):")
worst_declines = comparison_sorted.head(3)
for i, (_, row) in enumerate(worst_declines.iterrows(), 1):
    print(f"   {i}. {row['carrier_code']}: {row['probability_pct_p1']:.2f}% -> {row['probability_pct_p2']:.2f}% "
          f"(+{row['change']:.2f}pp)")

print("\n" + "="*80)
print("ANALYSE FAERDIG!")
print("="*80)