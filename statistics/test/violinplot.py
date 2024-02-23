import seaborn as sns 
import matplotlib.pyplot as plt


df = sns.load_dataset("titanic")
sns.violinplot(data=df, x="age", y="class")

plt.tight_layout()
plt.show()