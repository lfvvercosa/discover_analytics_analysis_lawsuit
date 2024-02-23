import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Plotting the data
x = [1, 2, 3, 4, 5]
y = [1000000, 2000000, 3000000, 4000000, 5000000]
plt.plot(x, y)

# Format the y-axis ticks to remove the 'e' symbol
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

# Display the plot
plt.show()