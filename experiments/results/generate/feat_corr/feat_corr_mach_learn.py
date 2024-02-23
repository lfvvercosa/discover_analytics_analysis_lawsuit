from statistics import mean
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from experiments.results.generate.feat_corr.feat_utils import show_hist
from experiments.results.generate.feat_corr.feat_utils import show_hist_Threshold
from experiments.results.pre_process_dataset import pre_process_dataset
from experiments.features_creation.feat_manager import translate_features_name
from sklearn.model_selection import KFold


def load_df(k):
    dataset_path = 'experiments/results/markov/k_' + str(k) + \
                   '/df_markov_k_' +  str(k) + '.csv'

    df = pd.read_csv(dataset_path, sep='\t')

    return df


def get_feat(df, gt, not_a_feature):
    feats = list(df.columns)
    feats.remove(gt)
    feats = [f for f in feats if f not in not_a_feature]

    return feats


def get_feat_2(df, is_a_feature):
    feats = list(df.columns)
    feats = [f for f in feats if f in is_a_feature]

    return feats



def mach_learn_feat_impor(df, feats, gt, alg):
    X = df[feats].to_numpy()
    y = df[gt].to_numpy()
    feat_impor = []

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    if alg == 'RF':
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            rf = RandomForestRegressor()
            rf.fit(X_train,y_train)
            feat_impor.append(rf.feature_importances_)

        feat_impor = np.array(feat_impor).mean(axis=0)
    
    if alg == 'XGB':
        d = {}

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            xgb = XGBRegressor(n_estimators=100)
            xgb.fit(X_train, y_train)
            fi = xgb.get_booster().get_score(importance_type='gain')

            for f in fi.keys():
                if f not in d:
                    d[f] = []

                d[f].append(fi[f])

        for f in d:
            d[f] = mean(d[f])
        
        count = 0

        for i in range(len(d)):
            key = 'f' + str(i)

            if key in d:
                feat_impor.append(d[key])
            else:
                feat_impor.append(0)

    return feat_impor


def rf_feat_selec(df, feats, gt):
    X = df[feats].to_numpy()
    y = df[gt].to_numpy()
    feat_impor = []

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
        sel.fit(X_train, y_train)

        chosen_feat = sel.get_support()
        sel_chosen = [a[0] for a in zip(feats, chosen_feat) if a[1]]

        feat_impor.append(sel_chosen)


    return feat_impor


# def avg_feat_import(gt, not_a_feature):
#     k_list = [1,2,3]

#     for k in k_list:
#         df = load_df(k)
#         df = pre_process_dataset(df, gt, not_a_feature)

#         feats = get_feat(df, gt)
#         feat_import = mach_learn_feat_impor(df, feats, gt)


#     feats = translate_features_name(feats)

if __name__ == '__main__':
    k = 1
    
    #ground_truth = 'PRECISION'
    ground_truth = 'FITNESS'
    '''
    not_a_feature = ['EVENT_LOG', 
                     'DISCOVERY_ALG', 
                     'PRECISION', 
                     'FITNESS', 
                     'TIME_MARKOV',
                     'PRECISION_50_RAND',
                     'PRECISION_50_FREQ',
                     'PRECISION_25_RAND',
                     'PRECISION_25_FREQ',
                     ]
    '''
    not_a_feature = ['EVENT_LOG', 
                     'DISCOVERY_ALG', 
                     'PRECISION', 
                     'FITNESS', 
                     'TIME_MARKOV',
                     'FITNESS_50_RAND',
                     'FITNESS_50_FREQ',
                     'FITNESS_25_RAND',
                     'FITNESS_25_FREQ',
                     ]
    
    aux_path = 'experiments/results/markov/k_3/df_markov_k_3.csv'

    df_aux = pd.read_csv(
                      aux_path,
                      sep='\t',
             )
    df_aux = df_aux[['EVENT_LOG', 'DISCOVERY_ALG']]
    
    df = load_df(k)
    df = df.merge(df_aux,
                  on=['EVENT_LOG', 'DISCOVERY_ALG'],
                  how='inner')

    df = pre_process_dataset(df, ground_truth, not_a_feature)

    feats = get_feat(df, ground_truth, not_a_feature)
    print("### total features: " + str(len(feats)))

    # print('### selected features RF:')
    # print(rf_feat_selec(df, feats, ground_truth))

    feat_import = mach_learn_feat_impor(df, feats, ground_truth, 'RF')
    #feat_import = mach_learn_feat_impor(df, feats, ground_truth, 'XGB')

    # feats = translate_features_name(feats)

    show_hist(feats, feat_import, 10)
    #show_hist(feats, feat_import, 7)
    #show_hist(feats, feat_import, 5)
    #show_hist(feats, feat_import, 3)

    #print(feat_import)
    #print(feats)

    #show_hist_Threshold(feats, feat_import, 0.15)

    

    
