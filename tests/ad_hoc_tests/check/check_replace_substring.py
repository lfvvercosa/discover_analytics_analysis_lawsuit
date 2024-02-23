import pandas as pd


data = {'COL_1':['.xespart.xes', 'a.xes.gz', 'xesgzba.xes'], 
        'COL_2':[0.2,0.1,0.11]}

df = pd.DataFrame.from_dict(data)


df['COL_1'] = df['COL_1'].str.replace(r'\.xes\.gz$', '')
df['COL_1'] = df['COL_1'].str.replace(r'\.xes$', '')

print(df)