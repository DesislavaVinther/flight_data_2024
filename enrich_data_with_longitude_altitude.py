
# importer værktøjer

import sys                      # giver info om Python og filer
import pandas as pd             # “super-Excel” i Python til at læse/skriv tabeller (CSV)
from pathlib import Path        # smart måde at arbejde med filstier
import os                       # giver info om Python og filer

# ========== 1) DEFINÉR BASISMAPPEN ÉN GANG ==========

BASE = Path(r"C:\Users\desib\PycharmProjects\Flight data\data")     # grundmappen med min datafile

MAIN = BASE / "flight_data_2024.csv"                                # flyvningerne med rækker med origin/dest
HELP = BASE / "airports_correlated_with_altitude_longitude.csv"     # lufthavnsliste med IATA + koordinater + højde
OUT  = BASE / "flight_data_2024_with_l_l_a.csv"                     # output, altså hvor resultatet skal gemmes

# lidt debug-udskrifter (kan hjælpe hvis noget går galt)
print("Python exe:", sys.executable)
print("CWD       :", Path.cwd())
print("MAIN      :", MAIN)
print("HELP      :", HELP)
print("OUT       :", OUT)

# ========== 2) TJEK AT INPUT-FILER FINDES ==========
assert MAIN.exists(), f"Fandt ikke MAIN: {MAIN}"   # stopper programmet med en klar fejlbesked, hvis en fil mangler
assert HELP.exists(), f"Fandt ikke HELP: {HELP}"   # sørger for at output-mappen findes (ellers bliver den lavet)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ========== 3) LÆS DATA ==========
# Læser begge CSV-filer ind som tabeller (DataFrames)
flights = pd.read_csv(MAIN)
airports = pd.read_csv(HELP)

# printer kolonnenavne, så jeg kan se hvad der er til rådighed.
print("MAIN columns:", list(flights.columns))
print("HELP columns:", list(airports.columns))


# ========== 4) KLARGØR HJÆLPETABEL (IATA + koord) ==========
lower_map = {c.lower(): c for c in airports.columns}  # laver et case-insensitivt opslag på kolonnenavne (så “IATA” og “iata” begge findes
need = ["iata", "latitude", "longitude", "altitude_ft"] # sikrer at de nødvendige kolonner findes
for k in need:
    assert k in lower_map, f"Hjælpefilen mangler kolonnen: {k}"

air = (airports.rename(columns={lower_map[k]: k for k in need})
               .loc[:, need]                                        # beholder kun de kolonner, vi skal bruge
               .dropna(subset=["iata"])                             # smider rækker væk hvor IATA mangler
               .copy())
air["iata"] = air["iata"].astype(str).str.strip().str.upper()       # rydder IATA-koder op (tekst, trim, UPPERCASE)
air = air.drop_duplicates(subset="iata", keep="first")              # fjerner dobbeltposter (hvis samme IATA står flere gange)

# Nu har vi en ren opslagstabel “IATA → (lat, lon, altitude)”.

# ========== 5) KLARGØR NØGLER I MAIN ==========
# tjekker at flytabellen har origin og dest kolonner
assert "origin" in flights.columns, "MAIN mangler kolonnen 'origin'"
assert "dest"   in flights.columns, "MAIN mangler kolonnen 'dest'"

# laver rensede match-nøgler _origin_iata og _dest_iata (samme rens som for hjælpetabellen)
fl = flights.copy()
fl["_origin_iata"] = fl["origin"].astype(str).str.strip().str.upper()
fl["_dest_iata"]   = fl["dest"].astype(str).str.strip().str.upper()

# vi skal slå op to gange i samme tabel (en gang for origin, en gang for dest)
# derfor laver vi to kopier med forskellige kolonnenavne (så de ikke overskriver hinanden)
# to kopier af 'air' med omdøbte kolonner for at undgå kollision
air_origin = air.rename(columns={
    "latitude": "origin_latitude",
    "longitude": "origin_longitude",
    "altitude_ft": "origin_altitude_ft"
})
air_dest = air.rename(columns={
    "latitude": "dest_latitude",
    "longitude": "dest_longitude",
    "altitude_ft": "dest_altitude_ft"
})

# ========== 6) MERGE ORIGIN ==========
# første sammenfletning: tilføj origin_latitude, origin_longitude, origin_altitude_ft ud fra fl._origin_iata == air_origin.iata.
merged = fl.merge(
    air_origin,
    left_on="_origin_iata",
    right_on="iata",
    how="left"  # behold alle flyrækker, og fyld data på når der findes et match; ellers får du tomme felter (NaN).
).drop(columns=["iata"])

# ========== 7) MERGE DEST ==========
# anden sammenfletning: gør det samme for destinationen
merged = merged.merge(
    air_dest,
    left_on="_dest_iata",
    right_on="iata",
    how="left"
).drop(columns=["iata", "_origin_iata", "_dest_iata"])  # rydder op ved at fjerne midlertidige kolonner

# ========== 8) TJEK & GEM ==========
# match-rate = hvor stor en andel der fik udfyldt koordinater (ikke tomt). 0,951 → 95,1%.
o_rate = merged["origin_latitude"].notna().mean()
d_rate = merged["dest_latitude"].notna().mean()
print(f"Match-rate  origin: {o_rate:.1%}   |   dest: {d_rate:.1%}")

# viser hvilke nye kolonner der blev lagt til:
print("Nye kolonner:", [c for c in merged.columns if c.startswith(("origin_", "dest_"))])

# gemmer resultatet som CSV uden ekstra række-indeks:
merged.to_csv(OUT, index=False)
print("OK! Skrev:", OUT, "størrelse:", os.path.getsize(OUT), "bytes")
# printer filstørrelsen som en lille bekræftels