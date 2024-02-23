import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from os import listdir
from os.path import isfile, join

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF


base_path = 'xes_files/test_variants/exp4/actitrac/config2/'
onlyfiles = [f for f in listdir(base_path) if isfile(join(base_path, f))]
utils = Utils()
count_id = 0
all_y_true = []
all_y_pred = []
all_ids_clus = []


for f in onlyfiles:
    log_path = join(base_path, f)
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    df = convert_to_dataframe(log)

    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]

    y_true = utils.get_ground_truth(ids_clus)
    y_pred = [count_id] * len(ids_clus)

    # df = df.drop_duplicates('case:concept:name')
    # ids_clus = df['case:concept:name'].to_list()

    all_y_true += y_true
    all_y_pred += y_pred
    all_ids_clus += ids_clus

    count_id += 1

print('size ids_clus list: ' + str(len(all_ids_clus)))
print('size ids_clus set: ' + str(len(set(all_ids_clus))))


ARI = adjusted_rand_score(all_y_true, all_y_pred)
Vm = v_measure_score(all_y_true, all_y_pred)


print('ARI: ' + str(ARI))
print('Vm: ' + str(Vm))
