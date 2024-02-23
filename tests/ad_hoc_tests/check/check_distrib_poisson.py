import numpy as np
import matplotlib.pyplot as plt


a = []

v = 70 - 40
m = v/4


for i in range(30000):
    s = np.random.poisson(3, 1)
    
    # s *= 7
    a.append(s[0])
    print()


# s = np.random.poisson(3, 30000)
count, bins, ignored = plt.hist(a, 14, density=True)
plt.show()
