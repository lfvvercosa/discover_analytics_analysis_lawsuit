import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

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

t1 = datetime.timedelta(days=0)
t2 = datetime.timedelta(days=60)
t3 = datetime.timedelta(days=(365*2))
bins = pd.IntervalIndex.from_tuples([(t1,t2),
                                     (t2,t3),
                                    ]
                                   )

df_time = df.groupby('id').agg(max_time=('time','max'),
                               min_time=('time','min'),
                               mov=('mov','first'))
df_time['total_time'] = df_time['max_time'] - df_time['min_time']
df_time = df_time.drop(columns=['max_time','min_time'])
df_time['cat'] = pd.cut(df_time['total_time'], bins, labels=[0,1])
df_time['cat'] = df_time['cat'].cat.rename_categories(['fast','slow'])

X = df_time[['mov','total_time']].to_numpy()
y_cat = df_time[['cat']].to_numpy()

X_train, X_test, _, _ = train_test_split(X, y_cat, test_size = 0.5, stratify=y_cat)

print()