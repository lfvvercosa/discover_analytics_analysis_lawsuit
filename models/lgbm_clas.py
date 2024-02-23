import pandas as pd
import warnings
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from models.DatasetSplit import DatasetSplit
from lightgbm import LGBMClassifier


DEBUG = True


def get_best_params(df, 
                    gt, 
                    not_a_feature, 
                    params, 
                    random_seed, 
                    splits, 
                    number_cores,
                    test_size):
    
    if params is None:
        boost_type = ['rf', 'gbdt', 'dart']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6]
        n_estim = [100, 200, 400, 600, 800, 1000]
        objective = ['multiclassova', 'multiclass']
    else:
        boost_type = [params['boosting_type']]
        learn_rate = [params['learning_rate']]
        n_estim = [params['n_estimators']]

    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)

    df = df[cols]

    min_log_loss = float('inf')

    feat = [f for f in df.columns if f not in not_a_feature]
    feature_names = df[feat].columns
    df_import = pd.DataFrame({'Feature': feature_names})

    params_and_result = {'training_perf':{}, 'test_perf':{}, 'params':{}}
    feat = [f for f in df.columns if f not in not_a_feature]

    dataset_split = DatasetSplit()

    # Make target to last column
    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)
    df = df[cols]

    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        strat_train_test_split(df, None, None, gt, test_size, random_seed)
    
    for boost in boost_type:
        print('boost type: ' + str(boost))
        for learn in learn_rate:
            for n in n_estim:
                for obj in objective:

                    print('parameters: ')
                    print(boost)
                    print(learn)
                    print(n)
                    print(obj)

                    skf = dataset_split.strat_kfold(X_train_,
                                                    y_train_,
                                                    splits, 
                                                    random_seed)
                    
                    logloss = []

                    for train_index, test_index in skf:
                        X_trainkf, X_validkf = X_train_[train_index,:-1], X_train_[test_index,:-1]
                        y_trainkf, y_validkf = y_train_[train_index].ravel(), \
                                               y_train_[test_index].ravel()

                        # y_trainkf, y_validkf = X_train_[train_index,-1], X_train_[test_index,-1]

                        model = LGBMClassifier(boosting_type=boost,
                                               learning_rate=learn,
                                               n_estimators=n,
                                               n_jobs=number_cores,
                                               verbose=0,
                                               objective=obj,
                                               bagging_freq=3,
                                               bagging_fraction=0.7)
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            model.fit(X_trainkf, y_trainkf)
                            y_pred_prob_kf = model.predict_proba(X_validkf)

                        logloss.append(log_loss(y_validkf, y_pred_prob_kf))

                    logloss_mean = sum(logloss) / len(logloss)
                    logloss_var = sum([(x - logloss_mean) ** 2 for x in logloss]) \
                                / len(logloss)
                    logloss_std = logloss_var**0.5


                    if logloss_mean < min_log_loss:
                        if DEBUG:
                            print('#### MIN LOG LOSS LGBM: ' + str(logloss_mean))
                        
                        min_log_loss = logloss_mean

                        params_and_result['params']['boosting_type'] = boost
                        params_and_result['params']['learning_rate'] = learn
                        params_and_result['params']['n_estimators'] = n
                        params_and_result['params']['objective'] = obj

                        params_and_result['training_perf']['LOSS'] = logloss
                        params_and_result['training_perf']['LOSS_avg'] = min_log_loss
                        params_and_result['training_perf']['LOSS_std'] = logloss_std

    X_train, X_test = X_train_[:,:-1], X_test_[:,:-1]
    # y_train, y_test = X_test_[:,:-1], X_test_[:,-1]

    model = LGBMClassifier(boosting_type = params_and_result['params']['boosting_type'],
                           learning_rate = params_and_result['params']['learning_rate'],
                           n_estimators = params_and_result['params']['n_estimators'],
                           objective = params_and_result['params']['objective'],
                           n_jobs=number_cores,
                           verbose=-1)
    model.fit(X_train, y_train_)

    y_pred_prob = model.predict_proba(X_test)

    params_and_result['test_perf']['LOSS'] = log_loss(y_test_, y_pred_prob)

    importance_type = model.get_params()['importance_type']
    import_col = 'import_' + importance_type
    df_import[import_col] = model.feature_importances_
    df_import = df_import.sort_values(import_col, ascending=False)


    return params_and_result, df_import


def run_model(
              df, 
              gt, 
              bins,
              labels, 
              params, 
              random_seed, 
              number_cores,
              test_size
             ):
    
    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)

    df = df[cols]

    dataset_split = DatasetSplit()

    if bins is not None and labels is not None:
        X_train_, X_test_, y_train_, y_test_ = dataset_split.\
            strat_train_test_split(df, bins, labels, gt, test_size, random_seed)
    else:
        X_train_, X_test_, y_train_, y_test_ = dataset_split.\
            strat_train_test_split(df, None, None, gt, test_size, random_seed)


    X_train_ = X_train_[:,:-1]
    X_test_ = X_test_[:,:-1]
    y_train_ = y_train_.ravel()
    y_test_ = y_test_.ravel()

    model = LGBMClassifier(
                           boosting_type=params['boosting_type'],
                           n_estimators=params['n_estimators'],
                           learning_rate=params['learning_rate'],
                           objective=params['objective'],
                           n_jobs=number_cores,
                           class_weight='balanced'
                          )

    model.fit(X_train_, y_train_)
    
    
    return model, y_test_, model.predict(X_test_) 