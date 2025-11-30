
import matplotlib.pyplot as plt
import numpy as np
# Generates grouped data
data_group1 = [np.random.normal(0, 1, 100), np.random.normal(1, 2, 100), np.random.normal(2, 1.5, 100)]
data_group2 = [np.random.normal(0, 1, 100), np.random.normal(1, 2, 100), np.random.normal(2, 1.5, 100)]
# Combines two data groups into a dataset
data = data_group1 + data_group2
# Creates grouped boxplots
plt.boxplot(data, positions=[1, 2, 3, 5, 6, 7], tick_labels=['G1-D1', 'G1-D2', 'G1-D3', 'G2-D1', 'G2-D2', 'G2-D3'])
plt.title('Grouped Boxplots')
plt.xlabel('Group-Dataset')
plt.ylabel('Value')
plt.show()


 