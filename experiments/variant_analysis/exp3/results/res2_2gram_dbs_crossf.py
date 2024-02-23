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
    dend = [([2], [1], 0.9357), ([9], [11], 0.9167), ([17], [23], 0.9107), ([9, 11], [3], 0.9087), ([9, 11, 3], [17, 23], 0.875), ([24], [2, 1], 0.8562), ([24, 2, 1], [10], 0.8571), ([29], [25], 0.8558), ([29, 25], [21], 0.8864), ([20], [27], 0.85), ([9, 11, 3, 17, 23], [6], 0.8478), ([9, 11, 3, 17, 23, 6], [29, 25, 21], 0.8525), ([24, 2, 1, 10], [15], 0.8291), ([0], [9, 11, 3, 17, 23, 6, 29, 25, 21], 0.8044), ([31], [8], 0.8), ([19], [18], 0.8), ([30], [7], 0.8), ([30, 7], [26], 0.9), ([31, 8], [30, 7, 26], 0.8), ([0, 9, 11, 3, 17, 23, 6, 29, 25, 21], [5], 0.7986), ([20, 27], [28], 0.7875), ([14], [19, 18], 0.7833), ([14, 19, 18], [4], 0.8648), ([31, 8, 30, 7, 26], [14, 19, 18, 4], 0.7864), ([39], [16], 0.7778), ([39, 16], [36], 0.8272), ([0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5], [20, 27, 28], 0.7767), ([24, 2, 1, 10, 15], [0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28], 0.7589), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28], [12], 0.7529), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12], [32], 0.7225), ([39, 16, 36], [33], 0.7094), ([31, 8, 30, 7, 26, 14, 19, 18, 4], [39, 16, 36, 33], 0.6813), ([40], [41], 0.6602), ([40, 41], [42], 0.6319), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32], [34], 0.5972), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34], [31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33], 0.5866), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34, 31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33], [37], 0.6075), ([35], [38], 0.5222), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34, 31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33, 37], [35, 38], 0.6801), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34, 31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33, 37, 35, 38], [13], 0.0549), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34, 31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33, 37, 35, 38, 13], [22], -0.0957), ([24, 2, 1, 10, 15, 0, 9, 11, 3, 17, 23, 6, 29, 25, 21, 5, 20, 27, 28, 12, 32, 34, 31, 8, 30, 7, 26, 14, 19, 18, 4, 39, 16, 36, 33, 37, 35, 38, 13, 22], [40, 41, 42], -0.31)]
    dict_gd = {0: {'G11': ['ABCDEFBGLLBGBLGLBGLBGBLGLBGJKL', 'ABCDEFBLGLBGJKL', 'ABCDEFLBGLBGJKL'], 'G13': ['ABCDEFBGNLJKF', 'ABCDEFBNGLBNLGJKF', 'ABCDEFBNLGJKF', 'ABCDEFNBGLNBGLBNLGJKF', 'ABCDEFNBLGJKF', 'ABCDEFNBLGNLBGJKF', 'ABCDEFNLBGJKF', 'ABCDEFNLBGNLBGBNLGJKF']}, 1: {'G12': ['ABCDEFGBLJL', 'ABCDEFGLBJK', 'ABCDEFGLBJL', 'ABCDEFLGBJK', 'ABCDEFLGBJL']}, 2: {'G11': ['ABCDEFLBGJLK']}, 3: {'G13': ['ABCDEFNBLGBNGLBGNLJKF', 'ABCDEFNBLGBNGLJKF']}, 4: {'G12': ['ABCDEFNJK'], 'G11': ['ABCDEFNJKL', 'ABCDEFNJLK']}, 5: {'G13': ['ABCEDFBGNLJKF', 'ABCEDFNBLGJKF', 'ABCEDFNBLGNLBGBGNLBNGLJKF', 'ABCEDFNLBGBNLGNBGLJKF', 'ABCEDFNLBGJKF']}, 6: {'G13': ['ABCEDFBGNLNBLGBNLGNBLGNBLGBNLGJKF', 'ABCEDFBNLGBNLGNBGLBNGLNBLGNBLGJKF']}, 7: {'G11': ['ABCEDFNJKL']}, 8: {'G11': ['ABCEDFNJLK']}, 9: {'G13': ['ABDCEFBGNLJKF', 'ABDCEFBNGLJKF', 'ABDCEFBNLGBNLGJKF', 'ABDCEFBNLGJKF', 'ABDCEFBNLGNLBGJKF', 'ABDCEFNLBGNBLGBGNLJKF', 'ABDCEFNLBGNLBGJKF'], 'G11': ['ABDCEFBLGBGLBLGJKL', 'ABDCEFBLGLBGJKL', 'ABDCEFLBGJKL']}, 10: {'G11': ['ABDCEFBLGJLK', 'ABDCEFLBGBLGJLK']}, 11: {'G13': ['ABDCEFBNLGBNGLNLBGNBGLBNLGNLBGNBGLJKF']}, 12: {'G12': ['ABDCEFGBGBLGBJL', 'ABDCEFGBLGBGBJL', 'ABDCEFGBLGBJL', 'ABDCEFGLBJL', 'ABDCEFLGBJL']}, 13: {'G12': ['ABDCEFLGBGBGBGBGBJK']}, 14: {'G12': ['ABDCEFNJK', 'ABDCEFNJL'], 'G11': ['ABDCEFNJKL', 'ABDCEFNJLK']}, 15: {'G12': ['ABDECFGLBGBJL', 'ABDECFGLBJL', 'ABDECFLGBJK', 'ABDECFLGBJL']}, 16: {'G12': ['ABDECFNJK', 'ABDECFNJL']}, 17: {'G11': ['ACBDEFBGLLBGBGLLBGBLGJKL', 'ACBDEFBGLLBGJKL'], 'G13': ['ACBDEFBGNLJKF', 'ACBDEFBGNLNBGLNLBGJKF', 'ACBDEFBNGLNLBGJKF', 'ACBDEFNBGLJKF', 'ACBDEFNBLGBGNLJKF']}, 18: {'G11': ['ACBDEFNJKL']}, 19: {'G11': ['ACBDEFNJLK']}, 20: {'G11': ['ACBEDFBGLLBGBGLLBGLBGJLK']}, 21: {'G13': ['ACBEDFBGNLJKF']}, 22: {'G11': ['ACBEDFBLGJKL']}, 23: {'G13': ['ACBEDFBNGLBNLGBGNLNBGLNBLGJKF']}, 24: {'G11': ['ACBEDFLBGJLK']}, 25: {'G13': ['ACBEDFNBGLNLBGJKF', 'ACBEDFNBLGJKF', 'ACBEDFNLBGJKF']}, 26: {'G11': ['ACBEDFNJKL']}, 27: {'G11': ['ACEBDFBGLBLGLBGBLGJLK', 'ACEBDFBGLJLK', 'ACEBDFLBGJLK']}, 28: {'G11': ['ACEBDFBGLLBGJKL', 'ACEBDFBGLLBGLBGBGLLBGJKL', 'ACEBDFBGLLBGLBGJKL']}, 29: {'G13': ['ACEBDFBGNLBGNLJKF', 'ACEBDFBGNLBNGLJKF', 'ACEBDFBGNLJKF', 'ACEBDFBGNLNBLGBGNLNLBGJKF', 'ACEBDFNBGLJKF', 'ACEBDFNBLGJKF', 'ACEBDFNBLGNLBGJKF', 'ACEBDFNLBGBGNLJKF', 'ACEBDFNLBGNBGLJKF']}, 30: {'G11': ['ACEBDFNJKL']}, 31: {'G11': ['ACEBDFNJLK']}, 32: {'G12': ['ADBCEFGBLJK', 'ADBCEFGLBGBGBJK', 'ADBCEFLGBGBJK', 'ADBCEFLGBGBJL', 'ADBCEFLGBJL']}, 33: {'G12': ['ADBCEFNJK']}, 34: {'G12': ['ADBECFGBLGBGBGBGBJL']}, 35: {'G12': ['ADBECFGLBJK', 'ADBECFLGBGBJK', 'ADBECFLGBJK']}, 36: {'G12': ['ADBECFNJK']}, 37: {'G12': ['ADEBCFLGBGBGBGBGBGBJL']}, 38: {'G12': ['ADEBCFLGBJK']}, 39: {'G12': ['ADEBCFNJK', 'ADEBCFNJL']}, 40: {'G22': ['PQRABBCGJZKV', 'PQRABBCGZJKV', 'PQRABBGCJZKV', 'PQRABBGCZVJK', 'PQRABCBGJZVK', 'PQRABCBGZJKV', 'PQRABCBGZVJK', 'PQRABGJKBZVC', 'PQRABGJZKBVC', 'PQRABGZVBJKC', 'PQRABGZVJBCK', 'PQRBABCGJZKV', 'PQRBABCGZJKV', 'PQRBABGJKCZV', 'PQRBABGJKZVC', 'PQRBAGZBJVKC', 'PQRBGABCJZKV', 'PQRBGABJCZKV', 'PQRBGAJZBCVK', 'PQRBGAJZBVCK', 'PQRBGAZBJCKV', 'PQRBGJABKZVC', 'PQRBGJZKVABC', 'PQRBGJZVKABC', 'PQRPQRABBCGJKZV', 'PQRPQRABBCGZJKV', 'PQRPQRABBGZCJVK', 'PQRPQRABBGZJVCK', 'PQRPQRABCBGZJVK', 'PQRPQRBABCGJZVK', 'PQRPQRBABCGZVJK', 'PQRPQRBABGZVJCK', 'PQRPQRBAGJZKBVC', 'PQRPQRBGJKZABCV', 'PQRPQRBGJKZVABC', 'PQRPQRPQRABBGZJVKC', 'PQRPQRPQRABCBGZVJK', 'PQRPQRPQRBABCGZJVK', 'PQRPQRPQRPQRABBGCZJKV', 'PQRPQRPQRPQRABGJKZVBC', 'PQRPQRPQRPQRPQRBABCGJKZV', 'PQRPQRPQRPQRPQRBAGJKZVBC'], 'G21': ['PQRABBCGZ', 'PQRABBCGZV', 'PQRABBGZCV', 'PQRABCBGJK', 'PQRABCBGZ', 'PQRABGJKBC', 'PQRABGZBC', 'PQRBABCGJK', 'PQRBABCGZ', 'PQRBABCGZV', 'PQRBAGBJKC', 'PQRBAGJBCK', 'PQRBGAJKBC', 'PQRBGJKABC', 'PQRBGZVABC', 'PQRQRABBGCZ', 'PQRQRABGBCZV', 'PQRQRBABCGJK', 'PQRQRBABCGZ', 'PQRQRBABGCJK', 'PQRQRBABGZC', 'PQRQRBABGZCV', 'PQRQRBAGJBCK', 'PQRQRBAGJBKC', 'PQRQRBAGJKBC', 'PQRQRBAGZBC', 'PQRQRBAGZBVC', 'PQRQRBGAZVBC', 'PQRQRBGJAKBC', 'PQRQRBGZABC', 'PQRQRQRABCBGJK', 'PQRQRQRABCBGZV', 'PQRQRQRABGZBC', 'PQRQRQRBABGCJK', 'PQRQRQRBABGZC', 'PQRQRQRBAGJBCK', 'PQRQRQRBGAJBCK', 'PQRQRQRBGJABCK', 'PQRQRQRBGZVABC', 'PQRQRQRQRQRABCBGZV', 'PQRQRQRQRQRBAGJBKC', 'PQRQRQRQRQRQRBAGZBC', 'PQRQRQRQRQRQRQRBGJKABC', 'PQRQRQRQRQRQRQRQRQRBAGZBC']}, 41: {'G22': ['PQRPQRABGBJZVCK']}, 42: {'G22': ['PQRPQRBAGBZVCJK']}}
    perc = 0.4

    utils = Utils()
    agglomClust = CustomAgglomClust()
    Z = agglomClust.gen_Z(dend)

    # hierarchy.dendrogram(Z)
    # plt.show(block=True)
    # plt.close()


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


