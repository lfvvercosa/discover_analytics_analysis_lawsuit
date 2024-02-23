import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create some example data
data = {'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
df = pd.DataFrame(data)

# Create the boxplot
sns.boxplot(
    x = "Category",
    y = "Value",
    showmeans=True,  # Show means as diamonds
    data=df
)

# Customize the plot
plt.title("Boxplot of Value by Category")
plt.xlabel("Category")
plt.ylabel("Value")
plt.xticks(rotation=45)  # Rotate category labels for better readability
plt.tight_layout()
plt.show()