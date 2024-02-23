from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble._weight_boosting import AdaBoostRegressor
from sklearn.model_selection import KFold

from models.DatasetSplit import DatasetSplit
import pandas as pd

DEBUG = True


def train_and_run_ada(df, 
                       gt, 
                       not_a_feature, 
                       params, 
                       random_seed, 
                       splits, 
                       test_size):
    
    if params is None:
        loss_type = ['linear','square','exponential']
        learn_rate = [0.5, 1, 2]
        n_estim = [50, 200, 400, 800]
    else:
        loss_type = [params['loss_type']]
        learn_rate = [params['learning_rate']]
        n_estim = [params['n_estimators']]

    min_mse = float('inf')
    min_mae = float('inf')
    min_r2 = float('inf')

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
    
    for loss in loss_type:
        print('boost type: ' + str(loss))
        for learn in learn_rate:
            print('learning_rate: ' + str(learn))
            for n in n_estim:
                print('estimators: ' + str(n))

                skf = dataset_split.strat_kfold(X_train_,
                                                y_train_,
                                                splits, 
                                                random_seed)
                
                mse = []
                r2 = []
                mae = []

                for train_index, test_index in skf:
                    X_trainkf, X_validkf = X_train_[train_index,:-1], X_train_[test_index,:-1]
                    y_trainkf, y_validkf = X_train_[train_index,-1], X_train_[test_index,-1]

                    reg = AdaBoostRegressor(
                                            loss=loss,
                                            learning_rate=learn,
                                            n_estimators=n,
                                           )
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
                        print('#### MIN MSE ADA: ' + str(mse_mean))
                    
                    min_mse = mse_mean
                    min_mae = mae_mean
                    min_r2 = r2_mean

                    min_mse_std = mse_std
                    min_mae_std = mae_std
                    min_r2_std = r2_std

                    params_and_result['params']['loss_type'] = loss
                    params_and_result['params']['learning_rate'] = learn
                    params_and_result['params']['n_estimators'] = n

                    params_and_result['training_perf']['MSE'] = mse
                    params_and_result['training_perf']['MAE'] = mae
                    params_and_result['training_perf']['R2'] = r2


                    params_and_result['training_perf']['MSE_avg'] = min_mse
                    params_and_result['training_perf']['MAE_avg'] = min_mae
                    params_and_result['training_perf']['R2_avg'] = min_r2

                    params_and_result['training_perf']['MSE_std'] = min_mse_std
                    params_and_result['training_perf']['MAE_std'] = min_mae_std
                    params_and_result['training_perf']['R2_std'] = min_r2_std

    X_train, y_train = X_train_[:,:-1], X_train_[:,-1]
    X_test, y_test = X_test_[:,:-1], X_test_[:,-1]

    reg = AdaBoostRegressor(
                            loss = params_and_result['params']['loss_type'],
                            learning_rate = params_and_result['params']['learning_rate'],
                            n_estimators = params_and_result['params']['n_estimators'],
                           )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    params_and_result['test_perf']['MSE'] = mean_squared_error(y_test, y_pred)
    params_and_result['test_perf']['MAE'] = mean_absolute_error(y_test, y_pred)
    params_and_result['test_perf']['R2'] = r2_score(y_test, y_pred) 

    # importance_type = reg.get_params()['importance_type']
    # import_col = 'import_' + importance_type
    # df_import[import_col] = reg.feature_importances_
    # df_import = df_import.sort_values(import_col, ascending=False)


    return params_and_result
