import pandas as pd
import matplotlib.pyplot as plt

# dataimport af minimal data fra roden af lokaldisk
usecols = ["fl_date", "dep_delay", "arr_delay", "diverted", "distance"]
df = pd.read_csv('C:/flight_data_2023_2025_V_1.3.csv', usecols=usecols)
# df = pd.read_csv('C:/flight_data_2024_sample.csv', usecols=usecols)

print(df.head())
A_list =[]
B_list =[]
for row in df.itertuples():
        ## Indsætte loop med liste af betingelser
        if  str(row.fl_date) < "2024-06-25":
            A_list.append(row)
        else:
            B_list.append(row)

Adf = pd.DataFrame(A_list)
Bdf = pd.DataFrame(B_list)

plt.xlim(-100, 6000)
plt.ylim(-700, 700)
plt.xlabel('Afstand')
plt.ylabel('Indhentet')
plt.title('P1')
plt.scatter(Adf['distance'], Adf['dep_delay'] - Adf['arr_delay'], c='b', marker='o')
plt.savefig('dist_indhent_foer.png')
#plt.show()

plt.xlim(-100, 6000)
plt.ylim(-700, 700)
plt.xlabel('Afstand')
plt.ylabel('Indhentet')
plt.title('P2')
plt.scatter(Bdf['distance'], Bdf['dep_delay'] - Bdf['arr_delay'], c='b', marker='o')
plt.savefig('dist_indhent_Efter.png')
#plt.show()