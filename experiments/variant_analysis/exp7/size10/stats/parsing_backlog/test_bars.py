# Library Import(numpy and matplotlib)
import matplotlib.pyplot as plot
import pandas as pd

# Make a data definition
_data = {'Women': [11, 19, 26, 34, 39],
        'Men': [20, 31, 29, 64, 30]}
_df = pd.DataFrame(_data,columns=['Women', 'Men'], 
                   index = ['MTECH', 'BEd', 'MBA', 'BTECH', 'MS'])

# Multiple bar chart
_df.plot.bar()

# Display the plot
plot.show()