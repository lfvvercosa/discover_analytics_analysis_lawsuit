import json
import pandas as pd
from os import listdir
from os.path import isfile, join, exists

base_dir = 'experiments/results/metrics/'
folders = ['IMf/', 'IMd/', 'ETM/']
dataframes = []

for fol in folders:
    curr_dir = base_dir + fol
    my_files = [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]

    for fil in my_files:
        file_path = curr_dir + fil
        with open(file_path) as json_file:
            data = json.load(json_file)
            dataframes.append(pd.DataFrame.from_dict(data))
            
df = pd.concat(dataframes)
df = df.drop_duplicates(subset=['event_log','algorithm'])
df.to_csv('experiments/results/metrics_mixed_dataset.csv', sep='\t')

print('done!')


