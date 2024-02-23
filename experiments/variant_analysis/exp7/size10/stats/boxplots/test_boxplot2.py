import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['Col1', 'Col2', 'Col3', 'Col4'])
# boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3']) 

plt.boxplot(df[['Col1', 'Col2', 'Col3']], labels=['Col1', 'Col2', 'Col3'])
plt.show()
