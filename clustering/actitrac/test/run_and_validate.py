from clustering.actitrac.ActiTraCConnector import ActiTracConnector
from pm4py.objects.log.importer.xes import importer as xes_importer

import pm4py

from clustering.agglomerative.Agglomerative import Agglomerative
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


def load_log(path, name):
    log = xes_importer.apply(path, 
                             variant=xes_importer.Variants.LINE_BY_LINE)
    log._attributes['concept:name'] = name
    pm4py.write_xes(log, path)


    return log


def get_dendrogram(df_log):
    agglom = Agglomerative(df_log, 
                           'single',
                           'levenshtein',
                           1,
                           1,
                           1,
                           is_map_act=False)

    dist_matrix = agglom.create_dist_matrix()
    Z = agglom.create_dendrogram(dist_matrix,
                                 method='single',
                                 metric='levenshtein')
    a = hierarchy.dendrogram(Z,
                             labels=agglom.variants,
                             leaf_rotation=90)
    plt.tight_layout()
    plt.savefig('temp/dendrograms/example_agglomerative_wl.png', dpi=300)


def get_readable_traces(df_log):
    return df_log.groupby('case:concept:name').agg(trace=('concept:name',list))



if __name__ == "__main__": 
    path = 'clustering/test/test_dendrogram2.xes'
    path_valid = 'clustering/test/test_dendrogram2_valid.xes'
    saving_path = '/home/vercosa/git/analysis_labor_lawsuits/temp/actitrac/train/'
    saving_path_valid = '/home/vercosa/git/analysis_labor_lawsuits/temp/actitrac/valid/'

    log = load_log(path, 'train')
    log_valid = load_log(path_valid, 'test')

    df_log = pm4py.convert_to_dataframe(log)
    df_log_valid = pm4py.convert_to_dataframe(log_valid)

    number_of_clusters = 3
    target_fit = 0.8
    is_greedy = True
    dist_greed = 0.25
    heu_miner_threshold = 0.9
    heu_miner_long_dist = False
    heu_miner_rel_best_thrs = 0.025
    heu_miner_and_thrs = 0.1
    min_clus_size = round((1/(2*number_of_clusters)),5)

    get_dendrogram(df_log)
    traces = get_readable_traces(df_log)
    traces_valid = get_readable_traces(df_log_valid)


    actitrac = ActiTracConnector(
                    is_greedy,
                    dist_greed,
                    target_fit,
                    min_clus_size,
                    heu_miner_threshold,
                    heu_miner_long_dist,
                    heu_miner_rel_best_thrs,
                    heu_miner_and_thrs,
               )
    
    df_clus = actitrac.run(number_of_clusters, 
                           path, 
                           saving_path, 
                           True)

    df_clus_valid = actitrac.validate(log_valid)
    

    print()