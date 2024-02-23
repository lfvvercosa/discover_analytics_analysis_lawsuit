import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
data = [
    [5665, 686, 44, 6],
    [677, 1453, 328, 42],
    [18, 332, 486, 284],
    [4, 26, 309, 1948]
]

confusion_mat = np.array(data)

plt.figure(figsize=(8, 6))
sns.set(font_scale=2.0)
sns.heatmap(confusion_mat, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            annot_kws={"size": 22}, # Increase font size to 16
            # linewidths=0.25
           )  
plt.xlabel('Predicted', fontsize=22, labelpad=20)
plt.ylabel('Actual', fontsize=22, labelpad=20)
# plt.title('Confusion Matrix')

# Increase font size of color scale
# plt.colorbar().ax.tick_params(labelsize=16)

# plt.show()
labels = ['Fast', 'Medium', 'Critical', 'Urgent']
plt.xticks(np.arange(len(labels)) + 0.5, labels, fontsize=20)  # Increase font size to 16
plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=20)
# plt.yticks(rotation=90)  # Rotate y-axis labels by 90 degrees
plt.tight_layout()
plt.savefig('temp/confusion_matrix.png', dpi=400, bbox_inches='tight')