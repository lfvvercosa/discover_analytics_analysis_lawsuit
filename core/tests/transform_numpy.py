import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

d1 = datetime.date(2022, 12, 18)
d2 = datetime.date(2022, 12, 20)
d3 = datetime.date(2022, 12, 28)
d4 = datetime.date(2022, 12, 28)
d5 = datetime.date(2023, 12, 21)
d6 = datetime.date(2023, 12, 22)
d7 = datetime.date(2023, 12, 25)
d8 = datetime.date(2023, 8, 20)
d9 = datetime.date(2024, 12, 21)
d10 = datetime.date(2024, 1, 10)
d11 = datetime.date(2025, 2, 21)


d = {'id':['1','1','1','1','2','2','2','3','3','4','4'],
     'mov':['A','B','B','C','D','B','C','B','E','E','A'],
     'time':[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]}

df = pd.DataFrame.from_dict(d)
df = df.sort_values(by=['id','time'])

print(df)

X = df.to_numpy()