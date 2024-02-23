from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from models.DatasetSplit import DatasetSplit
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DEBUG = True


def train_and_run_lgbm(df, 
                       gt, 
                       not_a_feature, 
                       params, 
                       random_seed, 
                       splits, 
                       number_cores,
                       test_size
            ):
    
    if params is None:
        boost_type = ['gbdt','dart']
        learn_rate = [0.05, 0.1, 0.2]
        n_estim = [100, 200, 400, 800, 1600, 3200]
        # boost_type = ['gbdt']
        # learn_rate = [0.05]
        # n_estim = [100]

        importance_type = 'gain'
    else:
        boost_type = [params['params']['boosting_type']]
        learn_rate = [params['params']['learning_rate']]
        n_estim = [params['params']['n_estimators']]
        importance_type = params['params']['importance_type']

    min_mse = float('inf')
    min_mae = float('inf')
    min_r2 = float('inf')

    feat = [f for f in df.columns if f not in not_a_feature]
    feature_names = df[feat].columns
    df_import = pd.DataFrame({'Feature': feature_names})

    params_and_result = {'training_perf':{}, 
                         'test_perf':{}, 
                         'params':{}}
    feat = [f for f in df.columns if f not in not_a_feature]

    dataset_split = DatasetSplit()

    # Make target to last column
    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)
    df = df[cols]

    # X_train_, X_test_, y_train_, y_test_ = dataset_split.\
    #     strat_train_test_split(df, None, None, gt, test_size, random_seed)

    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        train_test_split(df, gt, test_size, random_seed)
    
    for boost in boost_type:
        print('boost type: ' + str(boost))
        for learn in learn_rate:
            print('learning_rate: ' + str(learn))
            for n in n_estim:
                print('estimators: ' + str(n))

                # skf = dataset_split.strat_kfold(X_train_,
                #                                 y_train_,
                #                                 splits, 
                #                                 random_seed)
                
                skf = dataset_split.kfold(X_train_,
                                          y_train_,
                                          splits, 
                                          random_seed)

                mse = []
                r2 = []
                mae = []

                for train_index, test_index in skf:
                    # X_trainkf, X_validkf = X_train_[train_index,:-1], X_train_[test_index,:-1]
                    # y_trainkf, y_validkf = X_train_[train_index,-1], X_train_[test_index,-1]

                    X_trainkf, X_validkf = X_train_[train_index], X_train_[test_index]
                    y_trainkf, y_validkf = y_train_[train_index], y_train_[test_index]

                    reg = LGBMRegressor(boosting_type=boost,
                                        learning_rate=learn,
                                        n_estimators=n,
                                        n_jobs=number_cores,
                                        verbose=-1,
                                        importance_type=importance_type)
                    reg.fit(X_trainkf, y_trainkf)
                    y_predkf = reg.predict(X_validkf)

                    mse.append(mean_squared_error(y_validkf, y_predkf))
                    r2.append(r2_score(y_validkf, y_predkf))
                    mae.append(mean_absolute_error(y_validkf, y_predkf))

                mse_mean = sum(mse) / len(mse)
                r2_mean = sum(r2) / len(r2)
                mae_mean = sum(mae) / len(mae)

                mse_var = sum([(x - mse_mean) ** 2 for x in mse]) / len(mse)
                r2_var = sum([(x - r2_mean) ** 2 for x in r2]) / len(mse)
                mae_var = sum([(x - mae_mean) ** 2 for x in mae]) / len(mse)

                mse_std = mse_var**0.5
                r2_std = r2_var**0.5
                mae_std = mae_var**0.5


                if mse_mean < min_mse:
                    if DEBUG:
                        print('#### MIN MSE LGBM: ' + str(mse_mean))
                    
                    min_mse = mse_mean
                    min_mae = mae_mean
                    min_r2 = r2_mean

                    min_mse_std = mse_std
                    min_mae_std = mae_std
                    min_r2_std = r2_std

                    params_and_result['params']['boosting_type'] = boost
                    params_and_result['params']['learning_rate'] = learn
                    params_and_result['params']['n_estimators'] = n
                    params_and_result['params']['importance_type'] = importance_type

                    params_and_result['training_perf']['MSE'] = mse
                    params_and_result['training_perf']['MAE'] = mae
                    params_and_result['training_perf']['R2'] = r2


                    params_and_result['training_perf']['MSE_avg'] = min_mse
                    params_and_result['training_perf']['MAE_avg'] = min_mae
                    params_and_result['training_perf']['R2_avg'] = min_r2

                    params_and_result['training_perf']['MSE_std'] = min_mse_std
                    params_and_result['training_perf']['MAE_std'] = min_mae_std
                    params_and_result['training_perf']['R2_std'] = min_r2_std

    # X_train, y_train = X_train_[:,:-1], X_train_[:,-1]
    # X_test, y_test = X_test_[:,:-1], X_test_[:,-1]

    X_train, X_test = X_train_, X_test_
    y_train, y_test = y_train_, y_test_

    reg = LGBMRegressor(boosting_type = params_and_result['params']['boosting_type'],
                        learning_rate = params_and_result['params']['learning_rate'],
                        n_estimators = params_and_result['params']['n_estimators'],
                        n_jobs=number_cores,
                        verbose=-1,
                        importance_type=params_and_result['params']['importance_type']
                       )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    params_and_result['test_perf']['MSE'] = mean_squared_error(y_test, y_pred)
    params_and_result['test_perf']['MAE'] = mean_absolute_error(y_test, y_pred)
    params_and_result['test_perf']['R2'] = r2_score(y_test, y_pred) 

    importance_type = reg.get_params()['importance_type']
    import_col = 'import_' + importance_type
    df_import[import_col] = reg.feature_importances_
    df_import = df_import.sort_values(import_col, ascending=False)

    if params is not None and 'discrete' in params:
            y_pred[y_pred < 0] = 1
            df_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

            bins = params['discrete']['bins']
            labels = params['discrete']['labels']

            df_results = dataset_split.gen_categories(df_results, 
                                                      bins, 
                                                      labels, 
                                                      'y_test', 
                                                      'cat_y_test')
            
            df_results = dataset_split.gen_categories(df_results, 
                                                      bins, 
                                                      labels, 
                                                      'y_pred', 
                                                      'cat_y_pred')

            return df_results
            

    return params_and_result, df_import


def run_lgbm(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = LGBMRegressor(verbose=-1)
    else:
        
        boost = best_params['boosting_type'] 
        learn = best_params['learning_rate'] 
        n = best_params['n_estimators'] 

        reg = LGBMRegressor(boosting_type=boost,
                            learning_rate=learn,
                            n_estimators=n,
                            verbose=-1)

        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        y_pred = list(y_pred)
        # y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred


def get_feat_import_lgbm(df, 
                         gt, 
                         not_a_feature,
                         random_seed, 
                         importance_type, 
                         params):
    
    feat = [f for f in df.columns if f not in not_a_feature]
    X = df[feat].to_numpy()
    y = df[gt].to_numpy()
    splits = 10
    count = 0
    feature_names = df[feat].columns
    df_import = pd.DataFrame({'Feature': feature_names})

    skf = KFold(n_splits=splits, 
                shuffle=True, 
                random_state=random_seed)
    
    if DEBUG:
        print('Running LGBM...')

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = LGBMRegressor(boosting_type=params['boosting_type'],
                            learning_rate=params['learning_rate'],
                            n_estimators=params['n_estimators'],
                            importance_type=importance_type)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_pred = [y if y > 0 else 0 for y in y_pred]

        

        df_import[str(count)] = reg.feature_importances_

        count += 1

    df_import = df_import.set_index('Feature')
    se_import = df_import.mean(axis=1)
    se_import = se_import.sort_values(ascending=False)


    return se_import