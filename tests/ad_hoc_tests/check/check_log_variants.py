import pm4py
import pandas as pd


def update_labels(label1, label2):
    if ord(label2) > 122:
        return chr(ord(label1) + 1), chr(97)
    else:
        return chr(ord(label1)), chr(ord(label2) + 1)


def create_labels_acts(act_list):
    label1 = 'A'
    label2 = 'a'
    m = {'concept:name':[],'label':[]}

    for act in act_list:
        m['concept:name'].append(act)
        m['label'].append(label1 + label2)
        label1, label2 = update_labels(label1, label2) 


    return pd.DataFrame.from_dict(m)


path = '/home/vercosa/Insync/doutorado/artigos/artigo_alignment/'+\
       'BPI2015/BPI2015Reduced2014.xml'
log = pm4py.read_xes(path)
log = log[~log['case:concept:name'].isna()]

act_list = log.drop_duplicates('concept:name')['concept:name'].to_list()
df_label = create_labels_acts(act_list)

log = log.merge(df_label,on='concept:name',how='left')
log = log.drop(columns=['concept:name'])
log = log.rename(columns={'label':'concept:name'})
var = pm4py.get_variants(log)


path2 = '/home/vercosa/Insync/doutorado/artigos/artigo_alignment/'+\
        'BPI2015/frequencyLog.xml'
log2 = pm4py.read_xes(path2)
log2 = log2[~log2['case:concept:name'].isna()]

log2 = log2.merge(df_label,on='concept:name',how='left')
log2 = log2.drop(columns=['concept:name'])
log2 = log2.rename(columns={'label':'concept:name'})
var2 = pm4py.get_variants(log2)

count = 0

for v in var2:
    if v in var:
        count += 1

print(log)