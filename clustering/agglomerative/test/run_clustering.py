import datetime
import pandas as pd
from clustering.agglomerative.Agglomerative import Agglomerative

d1 = datetime.date(2022, 12, 15)
d2 = datetime.date(2022, 12, 20)
d3 = datetime.date(2022, 12, 25)
d4 = datetime.date(2022, 12, 26)
d5 = datetime.date(2022, 12, 27)
d6 = datetime.date(2022, 12, 28)
d7 = datetime.date(2022, 12, 29)
d8 = datetime.date(2022, 12, 30)
d9 = datetime.date(2023, 1, 1)
d10 = datetime.date(2023, 1, 2)


d = {'case:concept:name':['A1','A1','A1','A2','A2','A3','A3','A3','A4','A4'],
     'concept:name':['Ca','Ôn','Ca','Mo','Mo','Ca','Ca','Ca','Mo','Je'],
     'time:timestamp':[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]}

df = pd.DataFrame.from_dict(d)
df = df.sort_values(by='time:timestamp')

print(df)
print(df.groupby('case:concept:name').agg(movs=('concept:name',list)))



agglom = Agglomerative()
df_clus = agglom.run_agglomerative(df, 'concept:name', 'levenshtein','average', 2)

print(df_clus)