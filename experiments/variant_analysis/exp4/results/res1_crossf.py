from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
from pm4py.algo.discovery.inductive.variants.im_clean import algorithm as IM
from pm4py.algo.discovery.inductive.variants.im_f import algorithm as IMf
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import variants
from pm4py.visualization.petri_net import visualizer as pn_visualizer



def get_dict_var(dict_gd):
    dict_var = {}

    for k in dict_gd:
        dict_var[k] = []

        for i in dict_gd[k]:
            dict_var[k] += dict_gd[k][i]
    

    return dict_var


def get_df_distrib(y_true, y_pred):
    distrib = {}
    pred_map = {}
    pred_list = list(set(y_pred))
    pred_list.sort()
    count = 0
    total = len(pred_list)

    for i in pred_list:
        pred_map[i] = count
        count += 1

    for (true, pred) in zip(y_true, y_pred):
        if true not in distrib:
            distrib[true] = [0]*total

        distrib[true][pred_map[pred]] += 1
    
    df_distrib = pd.DataFrame.from_dict(distrib)
    df_distrib.index = pred_list

    return df_distrib


def get_df_gd(y_true, y_pred, traces):
    distrib = {}
    pred_map = {}
    pred_list = list(set(y_pred))
    pred_list.sort()
    count = 0
    total = len(pred_list)

    for i in pred_list:
        pred_map[i] = count
        count += 1

    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true not in distrib:
            distrib[true] = {}

        if pred not in distrib[true]:
            distrib[true][pred] = []

        distrib[true][pred].append(traces[idx])
    

    return distrib


def only_one_cluster(l, y_true, c):
    for e in l:
        if y_true[e] != c:
            return False
    
    return True


def get_first_merge_diff_clusters(c1, c2, dend, y_true):
    
    for e in dend:
        if only_one_cluster(e[0], y_true, c1) and \
           only_one_cluster(e[1], y_true, c2):
            return e
        if only_one_cluster(e[0], y_true, c2) and \
           only_one_cluster(e[1], y_true, c1):
            return e


    return None


def get_petri_nets(c1,c2):
    log_path = 'xes_files/test_variants/exp4/exp4.xes'
    log = xes_importer.apply(log_path)

    # convert to df
    df = convert_to_dataframe(log)

    # get traces
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    y_true = utils.get_ground_truth(ids_clus)

    # make each cluster individual to apply only agglomerative
    size = len(traces)
    cluster_labels = list(range(0,size))
    cluster_labels = np.array(cluster_labels)

    print(list(set(cluster_labels)))
    print(cluster_labels)

    # get df-log with cluster labels
    split_join.join_df(cluster_labels)
    df = split_join.df
    df_variants = split_join.join_df_uniques(cluster_labels)

    df_log1 = df_variants[df_variants['cluster_label'].isin(c1)]
    df_log2 = df_variants[df_variants['cluster_label'].isin(c2)]

    log1 = pm4py.convert_to_event_log(df_log1)
    log2 = pm4py.convert_to_event_log(df_log2)

    thresh = 0.8

    params_ind = {IMf.Parameters.NOISE_THRESHOLD:thresh}
    net1, im1, fm1 = IMf.apply(log1, params_ind)
    net2, im2, fm2 = IMf.apply(log2, params_ind)
    
    # gviz = pn_visualizer.apply(net1, im1, fm1)
    # pn_visualizer.view(gviz)

    align_var = variants.dijkstra_less_memory
    params_ali = {replay_fitness.Parameters.ALIGN_VARIANT:align_var,
                #   alignment_based.Parameters.MULTIPROCESSING:True,
                    }

    v1 = replay_fitness.apply(log2, net1, im1, fm1, 
                                variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                parameters=params_ali)
    
    v2 = replay_fitness.apply(log1, net2, im2, fm2, 
                                variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                parameters=params_ali)
    
    from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust

    agglomClust = CustomAgglomClust()
    df_log3 = df_variants[df_variants['cluster_label'].isin([85,87,84])]
    agglomClust.fit_log_alignment(df_log1, df_log3)

    print()

# t_y_true = [11, 11, 12, 12, 13, 13, 13]
# t_y_pred = [10, 10, 20, 20, 20, 30, 10]

# print(get_df_distrib(t_y_true, t_y_pred))


traces = ['ABCDEFGLMGMJO', 'ABCDEFGMLJK', 'ABCDEFMGNLJKF', 'ABCDEFNJK', 'ABCDEFNJKO', 'ABCDEFNJO', 'ABCDEFNJOK', 'ABCDEFNLMGJKF', 'ABCDEFNMLGJKF', 'ABCEDFLMGJKO', 'ABCEDFLMGJOK', 'ABCEDFMGLLMGJOK', 'ABCEDFMGNLJKF', 'ABCEDFMGNLNMLGNLMGMGNLJKF', 'ABCEDFMNLGMNGLJKF', 'ABCEDFNJKO', 'ABCEDFNJOK', 'ABCEDFNLMGJKF', 'ABCEDFNMLGNMGLJKF', 'ABDCEFGMLGMGMJK', 'ABDCEFLGMJK', 'ABDCEFLMGJOK', 'ABDCEFLMGLMGJKO', 'ABDCEFLMGLMGJOK', 'ABDCEFLMGLMGMGLJOK', 'ABDCEFLMGMLGJKO', 'ABDCEFMGLJKO', 'ABDCEFMGLJOK', 'ABDCEFMGNLJKF', 'ABDCEFMGNLMNLGMNGLJKF', 'ABDCEFMLGJOK', 'ABDCEFMLGMGLLMGJKO', 'ABDCEFMNGLJKF', 'ABDCEFMNGLNLMGMNLGMGNLJKF', 'ABDCEFMNLGJKF', 'ABDCEFMNLGNLMGMGNLJKF', 'ABDCEFNJO', 'ABDCEFNJOK', 'ABDCEFNLMGJKF', 'ABDCEFNMGLJKF', 'ABDCEFNMGLMGNLJKF', 'ABDCEFNMGLMGNLNLMGJKF', 'ABDCEFNMLGMGNLMGNLJKF', 'ABDECFGMLJK', 'ABDECFLGMGMJK', 'ABDECFNJO', 'ACBDEFLMGJOK', 'ACBDEFLMGMGLLMGLMGJKO', 'ACBDEFMNLGNLMGNMGLJKF', 'ACBDEFNJKO', 'ACBDEFNJOK', 'ACBDEFNLMGMGNLJKF', 'ACBDEFNMLGJKF', 'ACBEDFLMGJOK', 'ACBEDFLMGMLGMLGLMGJOK', 'ACBEDFMGLLMGMGLJOK', 'ACBEDFMGNLJKF', 'ACBEDFMNLGJKF', 'ACBEDFNJOK', 'ACBEDFNMGLMGNLMGNLNLMGMGNLJKF', 'ACEBDFLMGMLGJKO', 'ACEBDFMGLJKO', 'ACEBDFMGLMLGJKO', 'ACEBDFMGLMLGJOK', 'ACEBDFMGNLJKF', 'ACEBDFNJOK', 'ADBCEFGLMJO', 'ADBCEFLGMJO', 'ADBCEFNJO', 'ADBECFGLMGMJK', 'ADBECFLGMJK', 'ADBECFLGMJO', 'ADBECFNJO', 'ADEBCFGMLGMGMJK', 'ADEBCFGMLJO', 'ADEBCFLGMGMGMJO', 'ADEBCFLGMGMJK', 'ADEBCFLGMGMJO', 'PQRABCSGJKZV', 'PQRABGSCJK', 'PQRABGZVSC', 'PQRABSCGJK', 'PQRABSCGZVJK', 'PQRABSGJKC', 'PQRASBCGJK', 'PQRASBGCJK', 'PQRASBGCJZKV', 'PQRASCBGJK', 'PQRASCBGZV', 'PQRASGBCJZVK', 'PQRASGBCZJKV', 'PQRASGZJBVKC', 'PQRBAGZVSC', 'PQRBASCGJK', 'PQRBASGCJK', 'PQRBASGCZ', 'PQRBGJKASC', 'PQRBGZAVSC', 'PQRPQRABCSGJZKV', 'PQRPQRABCSGZJKV', 'PQRPQRABCSGZJVK', 'PQRPQRABSCGJZKV', 'PQRPQRABSGJKCZV', 'PQRPQRASBCGZVJK', 'PQRPQRPQRABCSGZJVK', 'PQRPQRPQRABCSGZVJK', 'PQRPQRPQRABSGZVJKC', 'PQRPQRPQRASGBCZVJK', 'PQRPQRPQRPQRABCSGZJKV', 'PQRPQRPQRPQRABSGJCZVK', 'PQRPQRPQRPQRASBCGZJKV', 'PQRPQRPQRPQRASBGZJKCV', 'PQRPQRPQRPQRPQRSABGCZJVK', 'PQRPQRPQRPQRSGJAZBVKC', 'PQRPQRPQRSGABZCVJK', 'PQRPQRPQRSGZVAJKBC', 'PQRPQRSAGJBKZVC', 'PQRQRABSCGZV', 'PQRQRASBCGZV', 'PQRQRASBGZC', 'PQRQRASCBGJK', 'PQRQRBAGZSVC', 'PQRQRBASCGZ', 'PQRQRBASCGZV', 'PQRQRBASGCZ', 'PQRQRBASGJCK', 'PQRQRBGAZSC', 'PQRQRQRABGZSC', 'PQRQRQRASCBGJK', 'PQRQRQRBGZAVSC', 'PQRQRQRQRBASCGZV', 'PQRQRQRQRBGAZSC', 'PQRQRQRQRQRASBGJCK', 'PQRQRQRQRQRASCBGZV', 'PQRQRQRQRQRQRQRASCBGJK', 'PQRSABCGJKZV', 'PQRSABCGZVJK', 'PQRSABGCJZKV', 'PQRSABGCZJVK', 'PQRSABGJCZKV', 'PQRSABGZCJVK', 'PQRSAGZVBCJK', 'PQRSGAJBZVCK', 'PQRSGAZVBJKC', 'PQRSGZVABJCK']

dend = [([35], [42], 1.0), ([100], [104], 1.0), ([77], [75], 1.0), ([128], [120], 1.0), ([108], [99], 1.0), ([123], [130], 1.0), ([126], [131], 1.0), ([128, 120], [134], 1.0), ([22], [25], 0.9167), ([69], [70], 0.9091), ([37], [36], 0.8944), ([4], [3], 0.8944), ([6], [5], 0.8944), ([123, 130], [122], 0.8944), ([124], [95], 0.8889), ([23], [21], 0.875), ([62], [60], 0.8712), ([24], [23, 21], 0.8667), ([76], [73], 0.8591), ([4, 3], [6, 5], 0.8556), ([26], [27], 0.8333), ([53], [46], 0.8333), ([9], [10], 0.8333), ([33], [29], 0.8333), ([86], [137], 0.8333), ([136], [82], 0.8333), ([135], [78], 0.8333), ([140], [138], 0.8333), ([101], [98], 0.8333), ([107], [103], 0.8333), ([108, 99], [100, 104], 0.8333), ([108, 99, 100, 104], [105], 0.8333), ([109], [102], 0.8333), ([111], [110], 0.8333), ([43], [1], 0.8182), ([69, 70], [76, 73], 0.8182), ([67], [66], 0.8182), ([58], [16], 0.8), ([58, 16], [65], 0.8), ([49], [50], 0.8), ([85], [87], 0.8), ([85, 87], [84], 0.9), ([81], [83], 0.8), ([80], [92], 0.8), ([80, 92], [97], 0.8), ([93], [94], 0.8), ([85, 87, 84], [93, 94], 0.8), ([85, 87, 84, 93, 94], [81, 83], 0.8333), ([117], [118], 0.8), ([133], [117, 118], 0.8), ([125], [132], 0.8), ([26, 27], [30], 0.7917), ([61], [62, 60], 0.7917), ([86, 137], [139], 0.7917), ([67, 66], [71], 0.7879), ([24, 23, 21], [22, 25], 0.7847), ([9, 10], [11], 0.7841), ([123, 130, 122], [133, 117, 118], 0.7833), ([26, 27, 30], [24, 23, 21, 22, 25], 0.7826), ([45], [72], 0.7778), ([119], [127], 0.7778), ([69, 70, 76, 73], [44], 0.7727), ([20], [19], 0.7682), ([43, 1], [20, 19], 0.7727), ([43, 1, 20, 19], [69, 70, 76, 73, 44], 0.8623), ([4, 3, 6, 5], [49, 50], 0.7667), ([108, 99, 100, 104, 105], [106], 0.7667), ([108, 99, 100, 104, 105, 106], [101, 98], 0.8472), ([108, 99, 100, 104, 105, 106, 101, 98], [109, 102], 0.8646), ([108, 99, 100, 104, 105, 106, 101, 98, 109, 102], [111, 110], 0.7916), ([108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110], [107, 103], 0.7847), ([26, 27, 30, 24, 23, 21, 22, 25], [31], 0.7656), ([26, 27, 30, 24, 23, 21, 22, 25, 31], [9, 10, 11], 0.7778), ([58, 16, 65], [4, 3, 6, 5, 49, 50], 0.75), ([58, 16, 65, 4, 3, 6, 5, 49, 50], [15], 0.8316), ([37, 36], [58, 16, 65, 4, 3, 6, 5, 49, 50, 15], 0.8048), ([41], [40], 0.75), ([128, 120, 134], [125, 132], 0.75), ([85, 87, 84, 93, 94, 81, 83], [136, 82], 0.7426), ([85, 87, 84, 93, 94, 81, 83, 136, 82], [135, 78], 0.7409), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78], [86, 137, 139], 0.7885), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139], [140, 138], 0.7604), ([80, 92, 97], [129], 0.7333), ([126, 131], [80, 92, 97, 129], 0.7833), ([119, 127], [126, 131, 80, 92, 97, 129], 0.7704), ([119, 127, 126, 131, 80, 92, 97, 129], [121], 0.825), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11], [47], 0.7257), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47], [54], 0.7692), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54], [55], 0.7589), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55], [53, 46], 0.75), ([123, 130, 122, 133, 117, 118], [88], 0.725), ([43, 1, 20, 19, 69, 70, 76, 73, 44], [0], 0.7197), ([43, 1, 20, 19, 69, 70, 76, 73, 44, 0], [67, 66, 71], 0.7483), ([43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71], [77, 75], 0.7701), ([43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71, 77, 75], [74], 0.7321), ([61, 62, 60], [63], 0.7121), ([45, 72], [68], 0.7037), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138], [108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103], 0.6832), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103], [112], 0.7708), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112], [114], 0.7096), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55, 53, 46], [61, 62, 60, 63], 0.6691), ([141], [143], 0.6667), ([141, 143], [144], 0.75), ([89], [90], 0.6667), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114], [89, 90], 0.6838), ([32], [34], 0.6666), ([32, 34], [28], 0.6666), ([12], [56], 0.6666), ([12, 56], [64], 0.6666), ([52], [8], 0.6666), ([7], [17], 0.6666), ([7, 17], [38], 0.6666), ([119, 127, 126, 131, 80, 92, 97, 129, 121], [123, 130, 122, 133, 117, 118, 88], 0.6604), ([119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88], [124, 95], 0.6951), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90], [128, 120, 134, 125, 132], 0.6556), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132], [119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95], 0.6833), ([37, 36, 58, 16, 65, 4, 3, 6, 5, 49, 50, 15], [45, 72, 68], 0.6494), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95], [116], 0.6479), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116], [113], 0.6455), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113], [115], 0.6458), ([35, 42], [33, 29], 0.625), ([32, 34, 28], [39], 0.625), ([32, 34, 28, 39], [7, 17, 38], 0.6508), ([32, 34, 28, 39, 7, 17, 38], [2], 0.6667), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115], [91], 0.6118), ([32, 34, 28, 39, 7, 17, 38, 2], [12, 56, 64], 0.6111), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115, 91], [141, 143, 144], 0.5983), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64], [52, 8], 0.5926), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115, 91, 141, 143, 144], [142], 0.5715), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55, 53, 46, 61, 62, 60, 63], [43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71, 77, 75, 74], 0.5694), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8], [57], 0.5606), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55, 53, 46, 61, 62, 60, 63, 43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71, 77, 75, 74], [37, 36, 58, 16, 65, 4, 3, 6, 5, 49, 50, 15, 45, 72, 68], 0.5194), ([48], [51], 0.5), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115, 91, 141, 143, 144, 142], [79], 0.4939), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57], [18], 0.4819), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18], [13], 0.8056), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13], [14], 0.52), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14], [35, 42, 33, 29], 0.4263), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14, 35, 42, 33, 29], [41, 40], 0.7333), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14, 35, 42, 33, 29, 41, 40], [48, 51], 0.6797), ([32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14, 35, 42, 33, 29, 41, 40, 48, 51], [59], 0.6029), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55, 53, 46, 61, 62, 60, 63, 43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71, 77, 75, 74, 37, 36, 58, 16, 65, 4, 3, 6, 5, 49, 50, 15, 45, 72, 68], [32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14, 35, 42, 33, 29, 41, 40, 48, 51, 59], 0.4229), ([85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115, 91, 141, 143, 144, 142, 79], [96], 0.3471), ([26, 27, 30, 24, 23, 21, 22, 25, 31, 9, 10, 11, 47, 54, 55, 53, 46, 61, 62, 60, 63, 43, 1, 20, 19, 69, 70, 76, 73, 44, 0, 67, 66, 71, 77, 75, 74, 37, 36, 58, 16, 65, 4, 3, 6, 5, 49, 50, 15, 45, 72, 68, 32, 34, 28, 39, 7, 17, 38, 2, 12, 56, 64, 52, 8, 57, 18, 13, 14, 35, 42, 33, 29, 41, 40, 48, 51, 59], [85, 87, 84, 93, 94, 81, 83, 136, 82, 135, 78, 86, 137, 139, 140, 138, 108, 99, 100, 104, 105, 106, 101, 98, 109, 102, 111, 110, 107, 103, 112, 114, 89, 90, 128, 120, 134, 125, 132, 119, 127, 126, 131, 80, 92, 97, 129, 121, 123, 130, 122, 133, 117, 118, 88, 124, 95, 116, 113, 115, 91, 141, 143, 144, 142, 79, 96], -0.2867)]

y_true = [12, 12, 13, 12, 11, 12, 11, 13, 13, 11, 11, 11, 13, 13, 13, 11, 11, 13, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 13, 13, 11, 11, 13, 13, 13, 13, 12, 11, 13, 13, 13, 13, 13, 12, 12, 12, 11, 11, 13, 11, 11, 13, 13, 11, 11, 11, 13, 13, 11, 13, 11, 11, 11, 11, 13, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 22, 21, 21, 21, 22, 21, 21, 21, 22, 21, 21, 22, 22, 22, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]

y_pred = [1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 1, 1, 2, 2, 4, 4, 3, 1, 1, 1, 1, 1, 5, 1, 1, 5, 2, 1, 1, 1, 2, 2, 1, 6, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

best_perc = 0.425

first_merge = get_first_merge_diff_clusters(21, 22, dend, y_true)

get_petri_nets(first_merge[0], first_merge[1])


utils = Utils()
agglomClust = CustomAgglomClust()
Z = agglomClust.gen_Z(dend)

print(get_df_distrib(y_true, y_pred))
df_gd = get_df_gd(y_true, y_pred, traces)

hierarchy.dendrogram(Z,color_threshold=max(Z[:,2])*best_perc)
# plt.show(block=True)
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
plt.savefig('temp/dendro_crossf.png', dpi=400)
plt.close()

# config 2, considering only the two major clusters
y_true2 = [0 if x < 20 else 1 for x in y_true]

# get best number of clusters
t = max(Z[:,2])

min_perc = 0.4
max_perc = 0.9
step_perc = 0.025
perc = min_perc

best_ARI = -1
best_Vm = -1
best_perc = -1
best_y_pred = []


while perc <= max_perc:
    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

    # get performance by adjusted rand-score metric
    
    # y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)
    y_pred = labels

    ARI = adjusted_rand_score(y_true2, y_pred)
    Vm = v_measure_score(y_true2, y_pred)

    if ARI > best_ARI:
        best_ARI = ARI
        best_Vm = Vm
        best_perc = perc
        best_y_pred = y_pred.copy()

    perc += step_perc


best_ARI = round(best_ARI, 4)
best_Vm = round(best_Vm, 4)
best_perc = round(best_perc, 4)

print('best ARI: ' + str(best_ARI))
print('best Vm: ' + str(best_Vm))
print('best perc: ' + str(best_perc))

agglomClust = CustomAgglomClust()
Z = agglomClust.gen_Z(dend)

hierarchy.dendrogram(Z,color_threshold=max(Z[:,2])*best_perc)
plt.show(block=True)
plt.close()
