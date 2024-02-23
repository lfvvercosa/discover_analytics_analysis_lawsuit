import datetime
import pandas as pd


d1 = datetime.date(2022, 12, 25)
d2 = datetime.date(2022, 12, 20)
d3 = datetime.date(2022, 12, 28)
d4 = datetime.date(2022, 12, 21)


d = {'A':['Limão','Laranja','Limão','Limão'],
     'B':['Carro','Ônibus','Caminhão Grande','Moto'],
     'C':[d1,d2,d3,d4]}

df = pd.DataFrame.from_dict(d)
df = df.sort_values(by='C')

print(df)
print(df.groupby('A').agg(movs=('B',list)))

