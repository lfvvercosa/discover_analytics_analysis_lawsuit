import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib import cycler
from matplotlib.lines import Line2D

# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = (np.geomspace(1, 10, 100) + np.random.randn(N, 100)).T
cmap = plt.cm.coolwarm
mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

custom_lines = [Line2D([0], [0], color='#FFFFFF', lw=1, linewidth=0.01),
                Line2D([0], [0], color='#FFFFFF', lw=1, linewidth=0.01),
                Line2D([0], [0], color='#FFFFFF', lw=1, linewidth=0.01)]

fig, ax = plt.subplots()
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])
# ax.legend(['A','B','C'], ['Cold', 'Medium', 'Hot'])
# ax.legend(['Cold', 'Medium', 'Hot'])

plt.show()