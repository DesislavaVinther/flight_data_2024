# henter værktøjskassen pandas og giver den kælenavnet pd
import pandas as pd


# liste med kolonnenavne
# den fil du jeg skal læse har ikke selv en overskriftslinje. Der bliver skrevet claypå forhånd, hvad kolonnerne skal hedde

cols = ["airport_id","name","city","country","iata","icao",
        "latitude","longitude","altitude_ft","timezone","dst","tz","type","source"]

# definere stien (adressen) til min .txt datafil på computeren
# bogstavet r foran strengens anførselstegn betyder raw string = Python skal tage backslashes \ bogstaveligt.
src = r"C:\Users\desib\OneDrive\Dokument\Data Science 1 sem\P_1\Airport _rute_map\airports.dat.txt"

# læs filen ind i en “DataFrame” (tænk: et regneark i hukommelsen)
df = pd.read_csv(                   # læs en komma-separeret tekstfil
    src,
    header=None, names=cols,        # fortæller, at første linje i filen ikke er overskrifter men data # bruger listen som kolonnenavne
    na_values="\\N",                # hvis der står \N i filen, så skal det behandles som manglende værdi (som tom celle i Excel)
    quotechar='"',                  # tekst, der står i anførselstegn, holdes samlet som ét felt – også selvom der er kommaer inde i teksten
    skipinitialspace=True           # ignorerer mellemrum lige efter kommaer
)

# gem den rensede tabel som en ny CSV-fil
# skriv en lille statusbesked i konsollen
df.to_csv("airports_correlated_with_altitude_longitude.csv", index=False)
print("Skrev airports_correlated_with_altitude_longitude.csv med", len(df), "rækker")
