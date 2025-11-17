import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# dataimport fra roden af lokaldisk
df = pd.read_csv('C:/flight_data_2024_sample.csv', low_memory=False)
#df = pd.read_csv('C:/flight_data_2024.csv', low_memory=False)
#df = pd.read_csv('C:/flight_data_2023_2025_merged.csv', low_memory=False)

print(df)
# test
# print(df.columns)

# Scatterplot uden sortering
# Det er dette plot vi vil lave med betingelser
plt.scatter(df['dep_delay'], df['arr_delay'])
plt.show()



# 2x Dict til dagsberegning ift 25 juni 2024 - TALLENE ER VIST IKKE HELT KORREKT men det er godt nok til test.
dict_year={"2023": -545, "2024":-180, "2025": 185}
dict_month={"0":0, "1":31, "2":59, "3":90, "4":120, "5":150, "6":180, "7":210, "8":240, "9":270, "10":300, "11":330}

# her er en kontrol af at kombinationen af dict synes at virke
# print(dict_year["2023"]+ dict_month["11"]+2)

# init af tomme A & B til opdeling

for row in df.itertuples():
  if (dict_year[str(row.year)]+ dict_month[str(row.month)] + row.day_of_month ) < 0:
     # Append til A
  else:
     # Append til B

# syntax-hjælp til 2x plot
# plt.scatter(df['dep_delay'], df['arr_delay'])
# plt.show()

# En løkke til efter