import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# dataimport af minimal data fra roden af lokaldisk
usecols = ["fl_date", "dep_delay", "arr_delay", "diverted", "distance"]
df = pd.read_csv('C:/flight_data_2023_2025_V_1.3.csv', usecols=usecols)
# df = pd.read_csv('C:/flight_data_2024_sample.csv', usecols=usecols)

# f tomme A & B .... skal loopes ud fra dictionay
A_list =[]
B_list =[]
C_list =[]
# Initialisering af datakotol


def pre(row):
 return(row.fl_date < "2024-06-25")

# En funktion der kan lave binne en række så der kan kan laves goodness of fit

for row in df.itertuples():
  if  pre(row):
      A_list.append(row)
  else:
      B_list.append(row)

print(len(A_list), len(B_list), len(C_list))

Adf = pd.DataFrame(A_list)
Bdf = pd.DataFrame(B_list)
Cdf = pd.DataFrame(C_list)

# plottet bør egentlig være loglog



plt.xlim(-400, 4500)
plt.ylim(-400, 4500)
plt.xlabel('Departue Delay')
plt.ylabel('Arrival Delay')
plt.title('P1')
plt.scatter((Adf['dep_delay']), (Adf['arr_delay']))
# plt.xscale('log')
# plt.yscale('log')
plt.savefig('scatter_p1.png')
plt.show()

plt.xlim(-400, 4500)
plt.ylim(-400, 4500)
plt.xlabel('Departue Delay')
plt.ylabel('Arrival Delay')
plt.title('P2')
plt.scatter(Bdf['dep_delay'], Bdf['arr_delay'])
plt.savefig('scatter_p2.png')
plt.show()




i=1
plt.title('Før')
plt.xlabel('Departue Delay')
plt.ylabel('Logaritmisk  forekomst')
plt.hist(Adf.dep_delay, bins=50, color='skyblue', edgecolor='black',range=(-200,4500), log=True)
plt.savefig('dep_foer.png')
plt.show()
print(i)
i +=1

plt.title('Før')
plt.xlabel('Arrival Delay')
plt.ylabel('Logaritmisk  forekomst')
plt.hist(Adf.arr_delay, bins=50, color='skyblue', edgecolor='black',range=(-200,4500), log=True)
plt.savefig('arr_foer.png')
plt.show()
print(i)
i +=1

plt.title('Efter')
plt.xlabel('Departue Delay')
plt.ylabel('Logaritmisk  forekomst')
plt.hist(Bdf.dep_delay, bins=50, color='skyblue', edgecolor='black',range=(-200,3600), log=True)
plt.savefig('dep_efter.png')
plt.show()
print(i)
i +=1

plt.title('Efter')
plt.xlabel('Arrival Delay')
plt.ylabel('Logaritmisk  forekomst')
plt.hist(Bdf.arr_delay, bins=50, color='skyblue', edgecolor='black',range=(-200,3600), log=True)
plt.savefig('arr_efter.png')
plt.show()
print(i)
i +=1

