import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


out_path = 'dataset/tribunais_trabalho/mini/results_boxplots.png'

df = pd.DataFrame({'UJ': [
                          0.2709852573533027, 
                          0.25418720426207453, 
                          0.41199700935093775, 
                          0.3088203339048269, 
                          0.38916240456058704, 
                          0.3453861638508485, 
                          0.2620096207437075, 
                          0.24159369593180857, 
                          0.4248243310025228, 
                          0.3647106326453404
                         ], 
                   'LAWSUIT': [
                               0.4578473322722707,
                               0.6370759378538257,
                               0.6587689428188614,
                               0.6301367500542323,
                               0.5601363524705711,
                               0.59483346319669,
                               0.5281647513440344,
                               0.5852298946289939,
                               0.5768884227761812,
                               0.6800401376650502
                              ],
                   'CLUSTERING': [
                                  0.5939191228928254,
                                  0.5870375610747715,
                                  0.5985603046600083,
                                  0.6488813939070182,
                                  0.6567193654512427,
                                  0.6617975213831534,
                                  0.6846449241910896,
                                  0.6701056781722066,
                                  0.6848025687875455,
                                  0.6307057593423248
                                 ],
                   'MOVEMENTS': [
                                 0.7093700063512649,
                                 0.727628503836651,
                                 0.7031126173776151,
                                 0.7715573923196367,
                                 0.7170520434851336,
                                 0.7376974568162389,
                                 0.7239081700761962,
                                 0.7103887076626102,
                                 0.7423111327330896,
                                 0.7248556499859166],
                   'ALL':[
                       0.7788011345714634,
                       0.8046124683677969,
                       0.8235530763864246,
                       0.8419953170258218,
                       0.7702343957581272,
                       0.8127229830837178,
                       0.7979588299208069,
                       0.7979830708667023,
                       0.8008023762235867,
                       0.8005981660836596
                   ]
                  })
data = df.values


# Add labels for the datasets
labels = np.array(['UJ', 'Lawsuit', 'Clustering', 'Movements','All'])

# Add a title
plt.title("R2-score by Feature Group")

# Change colors and fill boxes
plt.boxplot(data, 
            labels=labels,
            patch_artist=True, 
            boxprops=dict(facecolor='lightblue'), 
            medianprops=dict(linewidth=2, color='red'))
plt.ylim(bottom=0, top=1)

# Show the plot
plt.tight_layout()
plt.savefig(out_path)

print('done!')


