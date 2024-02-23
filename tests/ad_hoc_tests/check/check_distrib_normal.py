import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 32, 5 # mean and standard deviation
s = np.random.normal(mu, sigma, 10000)


# s = np.random.poisson(3, 30000)
count, bins, ignored = plt.hist(s, 14, density=True)
plt.show()
