import datetime
import pandas as pd
import numpy as np


def add_time(time, add):
    return time + datetime.timedelta(seconds=add)


ts = 1617295943.17321
datetime_str = '09/19/18 13:55:26'

d1 = datetime.datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
d2 = datetime.date(2022, 12, 20)
d3 = datetime.date(2022, 12, 28)
d4 = datetime.date(2022, 12, 28)
d5 = datetime.date(2023, 12, 21)
d6 = datetime.date(2023, 12, 21)
d7 = datetime.date(2023, 12, 21)
d8 = datetime.date(2024, 12, 20)
# d9 = datetime.date(2024, 12, 21)


d = {'id':['1','1','1','1','2','2','2','3'],
     'mov':['A','B','B','C','D','B','C','B'],
     'time':[d1,d2,d3,d4,d5,d6,d7,d8]}

df = pd.DataFrame.from_dict(d)
df = df.sort_values(by=['id','time'])

print(df)

print(df.groupby('id').agg(my_list=('mov',list)))