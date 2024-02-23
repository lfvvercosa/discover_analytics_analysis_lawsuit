import matplotlib.pyplot as plt
import numpy as np

# Generate random data for the histograms
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(0, 2, 1000)

# Plot the histograms with different scales
fig, ax1 = plt.subplots()

ax1.hist(data1, alpha=0.5, label='Data 1', color='skyblue')
ax1.set_xlabel('X label')
ax1.set_ylabel('Y label for Data 1', color='skyblue')

ax2 = ax1.twinx()
ax2.hist(data2, alpha=0.5, label='Data 2', color='salmon')
ax2.set_ylabel('Y label for Data 2', color='salmon')

plt.show()