import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from experiments.results.generate.feat_corr.feat_utils import show_hist
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


def random_forest_feat_impor(df, feats, gt):
    X = df[feats].to_numpy()
    y = df[gt].to_numpy()
    feat_impor = []

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        rf = XGBRegressor()
        rf.fit(X_train,y_train)
        feat_impor.append(rf.feature_importances_)

    feat_impor = np.array(feat_impor).mean(axis=0)
    return feat_impor


def avg_feat_import(gt, not_a_feature):
    k_list = [1,2,3]

    for k in k_list:
        df = load_df(k)
        df = pre_process_dataset(df, gt, not_a_feature)

        feats = get_feat(df, gt)
        feat_import = random_forest_feat_impor(df, feats, gt)


    feats = translate_features_name(feats)

if __name__ == '__main__':
    k = 3
    dataset_path = 'experiments/results/markov/k_' + str(k) + \
                   '/df_markov_k_' +  str(k) + '.csv'
    ground_truth = 'PRECISION'
    not_a_feature = ['EVENT_LOG', 'DISCOVERY_ALG', 'PRECISION', 'FITNESS']

    df = pd.read_csv(dataset_path, sep='\t')
    df = pre_process_dataset(df, ground_truth, not_a_feature)


    feats = list(df.columns)
    feats.remove(ground_truth)
    feats = [f for f in feats if f not in not_a_feature]

    X = df[feats]
    y = df[[ground_truth]]

    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)

    feat_import = xgb.get_booster().get_score(importance_type='gain')
    features = list(feat_import.keys())
    features = translate_features_name(features)
    values = list(feat_import.values())
    show_hist(features, values, 12)