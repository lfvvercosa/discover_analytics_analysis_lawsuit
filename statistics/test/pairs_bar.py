import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['A', 'B', 'C', 'D']
values1 = [20, 35, 30, 35]
values2 = [250, 320, 340, 200]

color1 = (0.16696655132641292, 0.48069204152249134, 0.7291503267973857)
color2 = (0.5356862745098039, 0.746082276047674, 0.8642522106881968)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values1, width, label='Values 1', color=color1)

ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, values2, width, label='Values 2222', color=color2)

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Scores')
ax2.set_ylabel('Scores2')

ax.set_title('Scores by group and values')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax2.legend()

plt.show()