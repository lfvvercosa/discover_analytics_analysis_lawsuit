from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils


def get_dict_var(dict_gd):
    dict_var = {}

    for k in dict_gd:
        dict_var[k] = []

        for i in dict_gd[k]:
            dict_var[k] += dict_gd[k][i]
    

    return dict_var


dend = [([15], [18], 1.0), ([49], [53], 1.0), ([36], [39], 0.875), ([41], [36, 39], 0.873), ([27], [19], 0.8636), ([14], [12], 0.8333), ([41, 36, 39], [38], 0.8333), ([42], [37], 0.8333), ([27, 19], [0], 0.8099), ([55], [52], 0.8), ([56], [54], 0.7999), ([41, 36, 39, 38], [42, 37], 0.7992), ([11], [10], 0.7833), ([49, 53], [50], 0.7778), ([49, 53, 50], [48], 0.7944), ([11, 10], [13], 0.7737), ([30], [56, 54], 0.7722), ([9], [8], 0.7682), ([27, 19, 0], [9, 8], 0.803), ([11, 10, 13], [20], 0.7667), ([17], [16], 0.75), ([30, 56, 54], [55, 52], 0.7443), ([1], [27, 19, 0, 9, 8], 0.7403), ([31], [51], 0.7333), ([49, 53, 50, 48], [31, 51], 0.7875), ([1, 27, 19, 0, 9, 8], [28], 0.7321), ([11, 10, 13, 20], [26], 0.7222), ([11, 10, 13, 20, 26], [23], 0.7546), ([11, 10, 13, 20, 26, 23], [4], 0.75), ([11, 10, 13, 20, 26, 23, 4], [24], 0.75), ([30, 56, 54, 55, 52], [29], 0.7151), ([2], [11, 10, 13, 20, 26, 23, 4, 24], 0.7122), ([41, 36, 39, 38, 42, 37], [40], 0.6923), ([30, 56, 54, 55, 52, 29], [41, 36, 39, 38, 42, 37, 40], 0.6744), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40], [43], 0.7225), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43], [45], 0.6675), ([58], [60], 0.6667), ([58, 60], [61], 0.75), ([32], [33], 0.6667), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45], [32, 33], 0.6582), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33], [57], 0.6324), ([15, 18], [14, 12], 0.625), ([58, 60, 61], [59], 0.5833), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57], [49, 53, 50, 48, 31, 51], 0.5746), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51], [47], 0.5649), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47], [44], 0.6445), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47, 44], [46], 0.6448), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47, 44, 46], [58, 60, 61, 59], 0.6211), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47, 44, 46, 58, 60, 61, 59], [34], 0.6183), ([2, 11, 10, 13, 20, 26, 23, 4, 24], [3], 0.5579), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3], [1, 27, 19, 0, 9, 8, 28], 0.5996), ([21], [22], 0.5), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28], [17, 16], 0.4568), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16], [15, 18, 14, 12], 0.5309), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12], [5], 0.5354), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12, 5], [21, 22], 0.4979), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12, 5, 21, 22], [7], 0.4439), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12, 5, 21, 22, 7], [6], 0.4408), ([30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47, 44, 46, 58, 60, 61, 59, 34], [35], 0.4304), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12, 5, 21, 22, 7, 6], [25], 0.4145), ([2, 11, 10, 13, 20, 26, 23, 4, 24, 3, 1, 27, 19, 0, 9, 8, 28, 17, 16, 15, 18, 14, 12, 5, 21, 22, 7, 6, 25], [30, 56, 54, 55, 52, 29, 41, 36, 39, 38, 42, 37, 40, 43, 45, 32, 33, 57, 49, 53, 50, 48, 31, 51, 47, 44, 46, 58, 60, 61, 59, 34, 35], -0.2867)]
y_true = [12, 12, 13, 12, 11, 12, 11, 13, 13, 11, 11, 11, 13, 13, 13, 11, 11, 13, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 13, 13, 11, 11, 13, 13, 13, 13, 12, 11, 13, 13, 13, 13, 13, 12, 12, 12, 11, 11, 13, 11, 11, 13, 13, 11, 11, 11, 13, 13, 11, 13, 11, 11, 11, 11, 13, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 22, 21, 21, 21, 22, 21, 21, 21, 22, 21, 21, 22, 22, 22, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
traces = ['ABCDEFGLMGMJO', 'ABCDEFGMLJK', 'ABCDEFMGNLJKF', 'ABCDEFNJK', 'ABCDEFNJKO', 'ABCDEFNJO', 'ABCDEFNJOK', 'ABCDEFNLMGJKF', 'ABCDEFNMLGJKF', 'ABCEDFLMGJKO', 'ABCEDFLMGJOK', 'ABCEDFMGLLMGJOK', 'ABCEDFMGNLJKF', 'ABCEDFMGNLNMLGNLMGMGNLJKF', 'ABCEDFMNLGMNGLJKF', 'ABCEDFNJKO', 'ABCEDFNJOK', 'ABCEDFNLMGJKF', 'ABCEDFNMLGNMGLJKF', 'ABDCEFGMLGMGMJK', 'ABDCEFLGMJK', 'ABDCEFLMGJOK', 'ABDCEFLMGLMGJKO', 'ABDCEFLMGLMGJOK', 'ABDCEFLMGLMGMGLJOK', 'ABDCEFLMGMLGJKO', 'ABDCEFMGLJKO', 'ABDCEFMGLJOK', 'ABDCEFMGNLJKF', 'ABDCEFMGNLMNLGMNGLJKF', 'ABDCEFMLGJOK', 'ABDCEFMLGMGLLMGJKO', 'ABDCEFMNGLJKF', 'ABDCEFMNGLNLMGMNLGMGNLJKF', 'ABDCEFMNLGJKF', 'ABDCEFMNLGNLMGMGNLJKF', 'ABDCEFNJO', 'ABDCEFNJOK', 'ABDCEFNLMGJKF', 'ABDCEFNMGLJKF', 'ABDCEFNMGLMGNLJKF', 'ABDCEFNMGLMGNLNLMGJKF', 'ABDCEFNMLGMGNLMGNLJKF', 'ABDECFGMLJK', 'ABDECFLGMGMJK', 'ABDECFNJO', 'ACBDEFLMGJOK', 'ACBDEFLMGMGLLMGLMGJKO', 'ACBDEFMNLGNLMGNMGLJKF', 'ACBDEFNJKO', 'ACBDEFNJOK', 'ACBDEFNLMGMGNLJKF', 'ACBDEFNMLGJKF', 'ACBEDFLMGJOK', 'ACBEDFLMGMLGMLGLMGJOK', 'ACBEDFMGLLMGMGLJOK', 'ACBEDFMGNLJKF', 'ACBEDFMNLGJKF', 'ACBEDFNJOK', 'ACBEDFNMGLMGNLMGNLNLMGMGNLJKF', 'ACEBDFLMGMLGJKO', 'ACEBDFMGLJKO', 'ACEBDFMGLMLGJKO', 'ACEBDFMGLMLGJOK', 'ACEBDFMGNLJKF', 'ACEBDFNJOK', 'ADBCEFGLMJO', 'ADBCEFLGMJO', 'ADBCEFNJO', 'ADBECFGLMGMJK', 'ADBECFLGMJK', 'ADBECFLGMJO', 'ADBECFNJO', 'ADEBCFGMLGMGMJK', 'ADEBCFGMLJO', 'ADEBCFLGMGMGMJO', 'ADEBCFLGMGMJK', 'ADEBCFLGMGMJO', 'PQRABCSGJKZV', 'PQRABGSCJK', 'PQRABGZVSC', 'PQRABSCGJK', 'PQRABSCGZVJK', 'PQRABSGJKC', 'PQRASBCGJK', 'PQRASBGCJK', 'PQRASBGCJZKV', 'PQRASCBGJK', 'PQRASCBGZV', 'PQRASGBCJZVK', 'PQRASGBCZJKV', 'PQRASGZJBVKC', 'PQRBAGZVSC', 'PQRBASCGJK', 'PQRBASGCJK', 'PQRBASGCZ', 'PQRBGJKASC', 'PQRBGZAVSC', 'PQRPQRABCSGJZKV', 'PQRPQRABCSGZJKV', 'PQRPQRABCSGZJVK', 'PQRPQRABSCGJZKV', 'PQRPQRABSGJKCZV', 'PQRPQRASBCGZVJK', 'PQRPQRPQRABCSGZJVK', 'PQRPQRPQRABCSGZVJK', 'PQRPQRPQRABSGZVJKC', 'PQRPQRPQRASGBCZVJK', 'PQRPQRPQRPQRABCSGZJKV', 'PQRPQRPQRPQRABSGJCZVK', 'PQRPQRPQRPQRASBCGZJKV', 'PQRPQRPQRPQRASBGZJKCV', 'PQRPQRPQRPQRPQRSABGCZJVK', 'PQRPQRPQRPQRSGJAZBVKC', 'PQRPQRPQRSGABZCVJK', 'PQRPQRPQRSGZVAJKBC', 'PQRPQRSAGJBKZVC', 'PQRQRABSCGZV', 'PQRQRASBCGZV', 'PQRQRASBGZC', 'PQRQRASCBGJK', 'PQRQRBAGZSVC', 'PQRQRBASCGZ', 'PQRQRBASCGZV', 'PQRQRBASGCZ', 'PQRQRBASGJCK', 'PQRQRBGAZSC', 'PQRQRQRABGZSC', 'PQRQRQRASCBGJK', 'PQRQRQRBGZAVSC', 'PQRQRQRQRBASCGZV', 'PQRQRQRQRBGAZSC', 'PQRQRQRQRQRASBGJCK', 'PQRQRQRQRQRASCBGZV', 'PQRQRQRQRQRQRQRASCBGJK', 'PQRSABCGJKZV', 'PQRSABCGZVJK', 'PQRSABGCJZKV', 'PQRSABGCZJVK', 'PQRSABGJCZKV', 'PQRSABGZCJVK', 'PQRSAGZVBCJK', 'PQRSGAJBZVCK', 'PQRSGAZVBJKC', 'PQRSGZVABJCK']
dict_gd = {0: {'G12': ['ABCDEFGLMGMJO']}, 1: {'G12': ['ABCDEFGMLJK', 'ABDECFGMLJK']}, 2: {'G13': ['ABCDEFMGNLJKF', 'ABCDEFNLMGJKF', 'ABCDEFNMLGJKF', 'ABCEDFMGNLJKF', 'ABCEDFNLMGJKF', 'ABDCEFMGNLJKF', 'ABDCEFMNGLJKF', 'ABDCEFMNLGJKF', 'ABDCEFNLMGJKF', 'ABDCEFNMGLJKF', 'ACBDEFNMLGJKF', 'ACBEDFMGNLJKF', 'ACBEDFMNLGJKF', 'ACEBDFMGNLJKF'], 'G11': ['ABCEDFLMGJKO', 'ABCEDFLMGJOK', 'ABDCEFLMGJOK', 'ABDCEFMGLJKO', 'ABDCEFMGLJOK', 'ABDCEFMLGJOK', 'ACBDEFLMGJOK', 'ACBEDFLMGJOK', 'ACEBDFMGLJKO']}, 3: {'G12': ['ABCDEFNJK', 'ABCDEFNJO', 'ABDCEFNJO', 'ABDECFNJO', 'ADBCEFNJO', 'ADBECFNJO'], 'G11': ['ABCDEFNJKO', 'ABCDEFNJOK', 'ABCEDFNJKO', 'ABCEDFNJOK', 'ABDCEFNJOK', 'ACBDEFNJKO', 'ACBDEFNJOK', 'ACBEDFNJOK', 'ACEBDFNJOK']}, 4: {'G11': ['ABCEDFMGLLMGJOK']}, 5: {'G13': ['ABCEDFMGNLNMLGNLMGMGNLJKF']}, 6: {'G13': ['ABCEDFMNLGMNGLJKF']}, 7: {'G13': ['ABCEDFNMLGNMGLJKF']}, 8: {'G12': ['ABDCEFGMLGMGMJK']}, 9: {'G12': ['ABDCEFLGMJK']}, 10: {'G11': ['ABDCEFLMGLMGJKO', 'ABDCEFLMGLMGJOK', 'ABDCEFLMGMLGJKO']}, 11: {'G11': ['ABDCEFLMGLMGMGLJOK']}, 12: {'G13': ['ABDCEFMGNLMNLGMNGLJKF']}, 13: {'G11': ['ABDCEFMLGMGLLMGJKO']}, 14: {'G13': ['ABDCEFMNGLNLMGMNLGMGNLJKF']}, 15: {'G13': ['ABDCEFMNLGNLMGMGNLJKF']}, 16: {'G13': ['ABDCEFNMGLMGNLJKF']}, 17: {'G13': ['ABDCEFNMGLMGNLNLMGJKF']}, 18: {'G13': ['ABDCEFNMLGMGNLMGNLJKF']}, 19: {'G12': ['ABDECFLGMGMJK', 'ADEBCFGMLGMGMJK', 'ADEBCFLGMGMGMJO', 'ADEBCFLGMGMJK', 'ADEBCFLGMGMJO']}, 20: {'G11': ['ACBDEFLMGMGLLMGLMGJKO']}, 21: {'G13': ['ACBDEFMNLGNLMGNMGLJKF']}, 22: {'G13': ['ACBDEFNLMGMGNLJKF']}, 23: {'G11': ['ACBEDFLMGMLGMLGLMGJOK']}, 24: {'G11': ['ACBEDFMGLLMGMGLJOK']}, 25: {'G13': ['ACBEDFNMGLMGNLMGNLNLMGMGNLJKF']}, 26: {'G11': ['ACEBDFLMGMLGJKO', 'ACEBDFMGLMLGJKO', 'ACEBDFMGLMLGJOK']}, 27: {'G12': ['ADBCEFGLMJO', 'ADBCEFLGMJO', 'ADBECFGLMGMJK', 'ADBECFLGMJK', 'ADBECFLGMJO']}, 28: {'G12': ['ADEBCFGMLJO']}, 29: {'G22': ['PQRABCSGJKZV', 'PQRSABCGJKZV']}, 30: {'G21': ['PQRABGSCJK', 'PQRABSCGJK', 'PQRABSGJKC', 'PQRASBCGJK', 'PQRASBGCJK', 'PQRASCBGJK', 'PQRASCBGZV', 'PQRBASCGJK', 'PQRBASGCJK', 'PQRBASGCZ', 'PQRQRABSCGZV', 'PQRQRASBCGZV', 'PQRQRASBGZC', 'PQRQRASCBGJK', 'PQRQRBASCGZ', 'PQRQRBASCGZV', 'PQRQRBASGCZ', 'PQRQRBASGJCK', 'PQRQRQRASCBGJK'], 'G22': ['PQRABSCGZVJK', 'PQRASBGCJZKV', 'PQRSABCGZVJK', 'PQRSABGCJZKV', 'PQRSABGJCZKV']}, 31: {'G21': ['PQRABGZVSC', 'PQRBAGZVSC', 'PQRBGZAVSC']}, 32: {'G22': ['PQRASGBCJZVK']}, 33: {'G22': ['PQRASGBCZJKV']}, 34: {'G22': ['PQRASGZJBVKC']}, 35: {'G21': ['PQRBGJKASC']}, 36: {'G22': ['PQRPQRABCSGJZKV', 'PQRPQRABCSGZJKV', 'PQRPQRABCSGZJVK', 'PQRPQRABSCGJZKV']}, 37: {'G22': ['PQRPQRABSGJKCZV']}, 38: {'G22': ['PQRPQRASBCGZVJK']}, 39: {'G22': ['PQRPQRPQRABCSGZJVK', 'PQRPQRPQRABCSGZVJK', 'PQRPQRPQRABSGZVJKC']}, 40: {'G22': ['PQRPQRPQRASGBCZVJK']}, 41: {'G22': ['PQRPQRPQRPQRABCSGZJKV', 'PQRPQRPQRPQRASBCGZJKV', 'PQRPQRPQRPQRASBGZJKCV']}, 42: {'G22': ['PQRPQRPQRPQRABSGJCZVK']}, 43: {'G22': ['PQRPQRPQRPQRPQRSABGCZJVK']}, 44: {'G22': ['PQRPQRPQRPQRSGJAZBVKC']}, 45: {'G22': ['PQRPQRPQRSGABZCVJK']}, 46: {'G22': ['PQRPQRPQRSGZVAJKBC']}, 47: {'G22': ['PQRPQRSAGJBKZVC']}, 48: {'G21': ['PQRQRBAGZSVC']}, 49: {'G21': ['PQRQRBGAZSC']}, 50: {'G21': ['PQRQRQRABGZSC']}, 51: {'G21': ['PQRQRQRBGZAVSC']}, 52: {'G21': ['PQRQRQRQRBASCGZV']}, 53: {'G21': ['PQRQRQRQRBGAZSC']}, 54: {'G21': ['PQRQRQRQRQRASBGJCK']}, 55: {'G21': ['PQRQRQRQRQRASCBGZV']}, 56: {'G21': ['PQRQRQRQRQRQRQRASCBGJK']}, 57: {'G22': ['PQRSABGCZJVK', 'PQRSABGZCJVK']}, 58: {'G22': ['PQRSAGZVBCJK']}, 59: {'G22': ['PQRSGAJBZVCK']}, 60: {'G22': ['PQRSGAZVBJKC']}, 61: {'G22': ['PQRSGZVABJCK']}}

utils = Utils()
agglomClust = CustomAgglomClust()
Z = agglomClust.gen_Z(dend)

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

    ARI = adjusted_rand_score(y_true, y_pred)
    Vm = v_measure_score(y_true, y_pred)

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