import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Indlæs data
# ===============================
columns = ["log_reg","op_unique_carrier","year","month"]
df = pd.read_csv("flight_data_2023_2025_V_1.2.csv", low_memory=False, usecols=columns)

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
    y = df_period["log_reg"]

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

# ===============================
# 5. Kør modellen for begge perioder
# ===============================
run_logit_period(p1, "Periode 1: juli 2023 - juni 2024")
run_logit_period(p2, "Periode 2: juli 2024 - juni 2025")