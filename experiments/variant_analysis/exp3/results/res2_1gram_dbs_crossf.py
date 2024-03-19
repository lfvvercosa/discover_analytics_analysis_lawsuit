from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust

import pandas as pd


def get_dict_var(dict_gd):
    dict_var = {}

    for k in dict_gd:
        dict_var[k] = []

        for i in dict_gd[k]:
            dict_var[k] += dict_gd[k][i]
    

    return dict_var

if __name__ == '__main__':
    traces = ['ABCDEFBGLLBGBLGLBGLBGBLGLBGJKL', 'ABCDEFBGNLJKF', 'ABCDEFBLGLBGJKL', 'ABCDEFBNGLBNLGJKF', 'ABCDEFBNLGJKF', 'ABCDEFGBLJL', 'ABCDEFGLBJK', 'ABCDEFGLBJL', 'ABCDEFLBGJLK', 'ABCDEFLBGLBGJKL', 'ABCDEFLGBJK', 'ABCDEFLGBJL', 'ABCDEFNBGLNBGLBNLGJKF', 'ABCDEFNBLGBNGLBGNLJKF', 'ABCDEFNBLGBNGLJKF', 'ABCDEFNBLGJKF', 'ABCDEFNBLGNLBGJKF', 'ABCDEFNJK', 'ABCDEFNJKL', 'ABCDEFNJLK', 'ABCDEFNLBGJKF', 'ABCDEFNLBGNLBGBNLGJKF', 'ABCEDFBGNLJKF', 'ABCEDFBGNLNBLGBNLGNBLGNBLGBNLGJKF', 'ABCEDFBNLGBNLGNBGLBNGLNBLGNBLGJKF', 'ABCEDFNBLGJKF', 'ABCEDFNBLGNLBGBGNLBNGLJKF', 'ABCEDFNJKL', 'ABCEDFNJLK', 'ABCEDFNLBGBNLGNBGLJKF', 'ABCEDFNLBGJKF', 'ABDCEFBGNLJKF', 'ABDCEFBLGBGLBLGJKL', 'ABDCEFBLGJLK', 'ABDCEFBLGLBGJKL', 'ABDCEFBNGLJKF', 'ABDCEFBNLGBNGLNLBGNBGLBNLGNLBGNBGLJKF', 'ABDCEFBNLGBNLGJKF', 'ABDCEFBNLGJKF', 'ABDCEFBNLGNLBGJKF', 'ABDCEFGBGBLGBJL', 'ABDCEFGBLGBGBJL', 'ABDCEFGBLGBJL', 'ABDCEFGLBJL', 'ABDCEFLBGBLGJLK', 'ABDCEFLBGJKL', 'ABDCEFLGBGBGBGBGBJK', 'ABDCEFLGBJL', 'ABDCEFNJK', 'ABDCEFNJKL', 'ABDCEFNJL', 'ABDCEFNJLK', 'ABDCEFNLBGNBLGBGNLJKF', 'ABDCEFNLBGNLBGJKF', 'ABDECFGLBGBJL', 'ABDECFGLBJL', 'ABDECFLGBJK', 'ABDECFLGBJL', 'ABDECFNJK', 'ABDECFNJL', 'ACBDEFBGLLBGBGLLBGBLGJKL', 'ACBDEFBGLLBGJKL', 'ACBDEFBGNLJKF', 'ACBDEFBGNLNBGLNLBGJKF', 'ACBDEFBNGLNLBGJKF', 'ACBDEFNBGLJKF', 'ACBDEFNBLGBGNLJKF', 'ACBDEFNJKL', 'ACBDEFNJLK', 'ACBEDFBGLLBGBGLLBGLBGJLK', 'ACBEDFBGNLJKF', 'ACBEDFBLGJKL', 'ACBEDFBNGLBNLGBGNLNBGLNBLGJKF', 'ACBEDFLBGJLK', 'ACBEDFNBGLNLBGJKF', 'ACBEDFNBLGJKF', 'ACBEDFNJKL', 'ACBEDFNLBGJKF', 'ACEBDFBGLBLGLBGBLGJLK', 'ACEBDFBGLJLK', 'ACEBDFBGLLBGJKL', 'ACEBDFBGLLBGLBGBGLLBGJKL', 'ACEBDFBGLLBGLBGJKL', 'ACEBDFBGNLBGNLJKF', 'ACEBDFBGNLBNGLJKF', 'ACEBDFBGNLJKF', 'ACEBDFBGNLNBLGBGNLNLBGJKF', 'ACEBDFLBGJLK', 'ACEBDFNBGLJKF', 'ACEBDFNBLGJKF', 'ACEBDFNBLGNLBGJKF', 'ACEBDFNJKL', 'ACEBDFNJLK', 'ACEBDFNLBGBGNLJKF', 'ACEBDFNLBGNBGLJKF', 'ADBCEFGBLJK', 'ADBCEFGLBGBGBJK', 'ADBCEFLGBGBJK', 'ADBCEFLGBGBJL', 'ADBCEFLGBJL', 'ADBCEFNJK', 'ADBECFGBLGBGBGBGBJL', 'ADBECFGLBJK', 'ADBECFLGBGBJK', 'ADBECFLGBJK', 'ADBECFNJK', 'ADEBCFLGBGBGBGBGBGBJL', 'ADEBCFLGBJK', 'ADEBCFNJK', 'ADEBCFNJL', 'PQRABBCGJZKV', 'PQRABBCGZ', 'PQRABBCGZJKV', 'PQRABBCGZV', 'PQRABBGCJZKV', 'PQRABBGCZVJK', 'PQRABBGZCV', 'PQRABCBGJK', 'PQRABCBGJZVK', 'PQRABCBGZ', 'PQRABCBGZJKV', 'PQRABCBGZVJK', 'PQRABGJKBC', 'PQRABGJKBZVC', 'PQRABGJZKBVC', 'PQRABGZBC', 'PQRABGZVBJKC', 'PQRABGZVJBCK', 'PQRBABCGJK', 'PQRBABCGJZKV', 'PQRBABCGZ', 'PQRBABCGZJKV', 'PQRBABCGZV', 'PQRBABGJKCZV', 'PQRBABGJKZVC', 'PQRBAGBJKC', 'PQRBAGJBCK', 'PQRBAGZBJVKC', 'PQRBGABCJZKV', 'PQRBGABJCZKV', 'PQRBGAJKBC', 'PQRBGAJZBCVK', 'PQRBGAJZBVCK', 'PQRBGAZBJCKV', 'PQRBGJABKZVC', 'PQRBGJKABC', 'PQRBGJZKVABC', 'PQRBGJZVKABC', 'PQRBGZVABC', 'PQRPQRABBCGJKZV', 'PQRPQRABBCGZJKV', 'PQRPQRABBGZCJVK', 'PQRPQRABBGZJVCK', 'PQRPQRABCBGZJVK', 'PQRPQRABGBJZVCK', 'PQRPQRBABCGJZVK', 'PQRPQRBABCGZVJK', 'PQRPQRBABGZVJCK', 'PQRPQRBAGBZVCJK', 'PQRPQRBAGJZKBVC', 'PQRPQRBGJKZABCV', 'PQRPQRBGJKZVABC', 'PQRPQRPQRABBGZJVKC', 'PQRPQRPQRABCBGZVJK', 'PQRPQRPQRBABCGZJVK', 'PQRPQRPQRPQRABBGCZJKV', 'PQRPQRPQRPQRABGJKZVBC', 'PQRPQRPQRPQRPQRBABCGJKZV', 'PQRPQRPQRPQRPQRBAGJKZVBC', 'PQRQRABBGCZ', 'PQRQRABGBCZV', 'PQRQRBABCGJK', 'PQRQRBABCGZ', 'PQRQRBABGCJK', 'PQRQRBABGZC', 'PQRQRBABGZCV', 'PQRQRBAGJBCK', 'PQRQRBAGJBKC', 'PQRQRBAGJKBC', 'PQRQRBAGZBC', 'PQRQRBAGZBVC', 'PQRQRBGAZVBC', 'PQRQRBGJAKBC', 'PQRQRBGZABC', 'PQRQRQRABCBGJK', 'PQRQRQRABCBGZV', 'PQRQRQRABGZBC', 'PQRQRQRBABGCJK', 'PQRQRQRBABGZC', 'PQRQRQRBAGJBCK', 'PQRQRQRBGAJBCK', 'PQRQRQRBGJABCK', 'PQRQRQRBGZVABC', 'PQRQRQRQRQRABCBGZV', 'PQRQRQRQRQRBAGJBKC', 'PQRQRQRQRQRQRBAGZBC', 'PQRQRQRQRQRQRQRBGJKABC', 'PQRQRQRQRQRQRQRQRQRBAGZBC']
    y_true = [11, 13, 11, 13, 13, 12, 12, 12, 11, 11, 12, 12, 13, 13, 13, 13, 13, 12, 11, 11, 13, 13, 13, 13, 13, 13, 13, 11, 11, 13, 13, 13, 11, 11, 11, 13, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 12, 12, 12, 11, 12, 11, 13, 13, 12, 12, 12, 12, 12, 12, 11, 11, 13, 13, 13, 13, 13, 11, 11, 11, 13, 11, 13, 11, 13, 13, 11, 13, 11, 11, 11, 11, 11, 13, 13, 13, 13, 11, 13, 13, 13, 11, 11, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 22, 21, 22, 21, 22, 22, 21, 21, 22, 21, 22, 22, 21, 22, 22, 21, 22, 22, 21, 22, 21, 22, 21, 22, 22, 21, 21, 22, 22, 22, 21, 22, 22, 22, 22, 21, 22, 22, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
    dend = [([11], [9], 1.0), ([4], [5], 0.8819), ([0], [2], 0.7821), ([4, 5], [7], 0.7597), ([8], [6], 0.7273), ([8, 6], [3], 0.7136), ([0, 2], [1], 0.7101), ([8, 6, 3], [4, 5, 7], 0.6581), ([8, 6, 3, 4, 5, 7], [10], 0.3276), ([8, 6, 3, 4, 5, 7, 10], [11, 9], 0.2804), ([0, 2, 1], [8, 6, 3, 4, 5, 7, 10, 11, 9], -0.3752)]
    dict_gd = {0: {'G11': ['ABCDEFBGLLBGBLGLBGLBGBLGLBGJKL', 'ABCDEFBLGLBGJKL', 'ABCDEFLBGJLK', 'ABCDEFLBGLBGJKL', 'ABCDEFNJKL', 'ABCDEFNJLK', 'ABCEDFNJKL', 'ABCEDFNJLK', 'ABDCEFBLGBGLBLGJKL', 'ABDCEFBLGJLK', 'ABDCEFBLGLBGJKL', 'ABDCEFLBGBLGJLK', 'ABDCEFLBGJKL', 'ABDCEFNJKL', 'ABDCEFNJLK', 'ACBDEFBGLLBGBGLLBGBLGJKL', 'ACBDEFBGLLBGJKL', 'ACBDEFNJKL', 'ACBDEFNJLK', 'ACBEDFBGLLBGBGLLBGLBGJLK', 'ACBEDFBLGJKL', 'ACBEDFLBGJLK', 'ACBEDFNJKL', 'ACEBDFBGLBLGLBGBLGJLK', 'ACEBDFBGLJLK', 'ACEBDFBGLLBGJKL', 'ACEBDFBGLLBGLBGBGLLBGJKL', 'ACEBDFBGLLBGLBGJKL', 'ACEBDFLBGJLK', 'ACEBDFNJKL', 'ACEBDFNJLK'], 'G12': ['ABCDEFGLBJK', 'ABCDEFLGBJK', 'ABCDEFNJK', 'ABDCEFLGBGBGBGBGBJK', 'ABDCEFNJK', 'ABDECFLGBJK', 'ABDECFNJK', 'ADBCEFGBLJK', 'ADBCEFGLBGBGBJK', 'ADBCEFLGBGBJK', 'ADBCEFNJK', 'ADBECFGLBJK', 'ADBECFLGBGBJK', 'ADBECFLGBJK', 'ADBECFNJK', 'ADEBCFLGBJK', 'ADEBCFNJK']}, 1: {'G13': ['ABCDEFBGNLJKF', 'ABCDEFBNGLBNLGJKF', 'ABCDEFBNLGJKF', 'ABCDEFNBGLNBGLBNLGJKF', 'ABCDEFNBLGBNGLBGNLJKF', 'ABCDEFNBLGBNGLJKF', 'ABCDEFNBLGJKF', 'ABCDEFNBLGNLBGJKF', 'ABCDEFNLBGJKF', 'ABCDEFNLBGNLBGBNLGJKF', 'ABCEDFBGNLJKF', 'ABCEDFBGNLNBLGBNLGNBLGNBLGBNLGJKF', 'ABCEDFBNLGBNLGNBGLBNGLNBLGNBLGJKF', 'ABCEDFNBLGJKF', 'ABCEDFNBLGNLBGBGNLBNGLJKF', 'ABCEDFNLBGBNLGNBGLJKF', 'ABCEDFNLBGJKF', 'ABDCEFBGNLJKF', 'ABDCEFBNGLJKF', 'ABDCEFBNLGBNGLNLBGNBGLBNLGNLBGNBGLJKF', 'ABDCEFBNLGBNLGJKF', 'ABDCEFBNLGJKF', 'ABDCEFBNLGNLBGJKF', 'ABDCEFNLBGNBLGBGNLJKF', 'ABDCEFNLBGNLBGJKF', 'ACBDEFBGNLJKF', 'ACBDEFBGNLNBGLNLBGJKF', 'ACBDEFBNGLNLBGJKF', 'ACBDEFNBGLJKF', 'ACBDEFNBLGBGNLJKF', 'ACBEDFBGNLJKF', 'ACBEDFBNGLBNLGBGNLNBGLNBLGJKF', 'ACBEDFNBGLNLBGJKF', 'ACBEDFNBLGJKF', 'ACBEDFNLBGJKF', 'ACEBDFBGNLBGNLJKF', 'ACEBDFBGNLBNGLJKF', 'ACEBDFBGNLJKF', 'ACEBDFBGNLNBLGBGNLNLBGJKF', 'ACEBDFNBGLJKF', 'ACEBDFNBLGJKF', 'ACEBDFNBLGNLBGJKF', 'ACEBDFNLBGBGNLJKF', 'ACEBDFNLBGNBGLJKF']}, 2: {'G12': ['ABCDEFGBLJL', 'ABCDEFGLBJL', 'ABCDEFLGBJL', 'ABDCEFGBGBLGBJL', 'ABDCEFGBLGBGBJL', 'ABDCEFGBLGBJL', 'ABDCEFGLBJL', 'ABDCEFLGBJL', 'ABDCEFNJL', 'ABDECFGLBGBJL', 'ABDECFGLBJL', 'ABDECFLGBJL', 'ABDECFNJL', 'ADBCEFLGBGBJL', 'ADBCEFLGBJL', 'ADBECFGBLGBGBGBGBJL', 'ADEBCFLGBGBGBGBGBGBJL', 'ADEBCFNJL']}, 3: {'G22': ['PQRABBCGJZKV', 'PQRABBCGZJKV', 'PQRABBGCJZKV', 'PQRABBGCZVJK', 'PQRABCBGJZVK', 'PQRABCBGZJKV', 'PQRABCBGZVJK', 'PQRABGJKBZVC', 'PQRABGJZKBVC', 'PQRABGZVBJKC', 'PQRABGZVJBCK', 'PQRBABCGJZKV', 'PQRBABCGZJKV', 'PQRBABGJKCZV', 'PQRBABGJKZVC', 'PQRBAGZBJVKC', 'PQRBGABCJZKV', 'PQRBGABJCZKV', 'PQRBGAJZBCVK', 'PQRBGAJZBVCK', 'PQRBGAZBJCKV', 'PQRBGJABKZVC', 'PQRBGJZKVABC', 'PQRBGJZVKABC', 'PQRPQRABBCGJKZV', 'PQRPQRABBCGZJKV', 'PQRPQRABBGZCJVK', 'PQRPQRABBGZJVCK', 'PQRPQRABCBGZJVK', 'PQRPQRABGBJZVCK', 'PQRPQRBABCGJZVK', 'PQRPQRBABCGZVJK', 'PQRPQRBABGZVJCK', 'PQRPQRBAGBZVCJK', 'PQRPQRBAGJZKBVC', 'PQRPQRBGJKZABCV', 'PQRPQRBGJKZVABC', 'PQRPQRPQRABBGZJVKC', 'PQRPQRPQRABCBGZVJK', 'PQRPQRPQRBABCGZJVK', 'PQRPQRPQRPQRABBGCZJKV', 'PQRPQRPQRPQRABGJKZVBC', 'PQRPQRPQRPQRPQRBABCGJKZV', 'PQRPQRPQRPQRPQRBAGJKZVBC']}, 4: {'G21': ['PQRABBCGZ', 'PQRABCBGZ', 'PQRABGZBC', 'PQRBABCGZ', 'PQRQRABBGCZ', 'PQRQRBABCGZ', 'PQRQRBABGZC', 'PQRQRBAGZBC', 'PQRQRBGZABC', 'PQRQRQRABGZBC', 'PQRQRQRBABGZC']}, 5: {'G21': ['PQRABBCGZV', 'PQRABBGZCV', 'PQRBABCGZV', 'PQRBGZVABC', 'PQRQRABGBCZV', 'PQRQRBABGZCV', 'PQRQRBAGZBVC', 'PQRQRBGAZVBC', 'PQRQRQRABCBGZV', 'PQRQRQRBGZVABC']}, 6: {'G21': ['PQRABCBGJK', 'PQRABGJKBC', 'PQRBABCGJK', 'PQRBAGBJKC', 'PQRBAGJBCK', 'PQRBGAJKBC', 'PQRBGJKABC', 'PQRQRBABCGJK', 'PQRQRBABGCJK', 'PQRQRBAGJBCK', 'PQRQRBAGJBKC', 'PQRQRBAGJKBC', 'PQRQRBGJAKBC', 'PQRQRQRABCBGJK', 'PQRQRQRBABGCJK', 'PQRQRQRBAGJBCK', 'PQRQRQRBGAJBCK', 'PQRQRQRBGJABCK']}, 7: {'G21': ['PQRQRQRQRQRABCBGZV']}, 8: {'G21': ['PQRQRQRQRQRBAGJBKC']}, 9: {'G21': ['PQRQRQRQRQRQRBAGZBC']}, 10: {'G21': ['PQRQRQRQRQRQRQRBGJKABC']}, 11: {'G21': ['PQRQRQRQRQRQRQRQRQRBAGZBC']}}
    perc = 0.23
    dict_var = get_dict_var(dict_gd)

    utils = Utils()
    agglomClust = CustomAgglomClust()
    Z = agglomClust.gen_Z(dend)
    t = max(Z[:,2])

    # hierarchy.dendrogram(Z, color_threshold=t*perc)
    # plt.show(block=True)
    # plt.close()

    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
    y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

    ARI = adjusted_rand_score(y_true, y_pred)
    Vm = v_measure_score(y_true, y_pred)

    # config 2, considering only the two major clusters
    y_true2 = [0 if x < 20 else 1 for x in y_true]
    dict_var = get_dict_var(dict_gd)

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
        
        y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

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

