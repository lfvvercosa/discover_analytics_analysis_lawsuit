

x = ['TOP7_RF', 
     'TOP5_RF', 
     'TOP3_RF',
     'TRSH15_RF',
     'TOP7_XGB', 
     'TOP5_XGB',
     'TOP3_XGB', 
     'TRSH15_XGB',
     'Top 10 RF']


new_order = [
             'TOP3_XGB',
             'TOP3_RF',
             'TOP5_XGB',
             'TOP5_RF',
             'TRSH15_RF',
             'TRSH15_XGB',
             'TOP7_XGB',
             'TOP7_RF',
             'Top 10 RF',
            ]


y_svr = [85, 85, 85, 83, 87, 85, 85, 87, 86]
y_lr = [81, 82, 83, 83, 83, 82, 83, 84, 83]
y_MLP = [80, 79, 74, 71, 86, 66, 78, 83, 85]
y_RFR = [88, 89, 84, 79, 89, 89, 84, 89, 86]
y_XGB = [89, 90, 85, 83, 90, 90, 85, 90, 88]


def get_new_order_idx(old, new):
    mapping = {}

    for i in range(len(old)):
        mapping[new.index(old[i])] =  i

    return mapping


def get_new_order(y, mapping):
    l = [None] * len(y)

    for e in range(len(y)):
        l[e] = y[mapping[e]]
    
    return l

mapping = get_new_order_idx(x, new_order)

print('y_svr = ' + str(get_new_order(y_svr, mapping)))
print('y_lr = ' + str(get_new_order(y_lr, mapping)))
print('y_MLP = ' + str(get_new_order(y_MLP, mapping)))
print('y_RFR = ' + str(get_new_order(y_RFR, mapping)))
print('y_XGB = ' + str(get_new_order(y_XGB, mapping)))

# print(mapping)
# print(l)










