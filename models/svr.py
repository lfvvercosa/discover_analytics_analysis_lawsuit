import pandas as pd
from models.DatasetSplit import DatasetSplit
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

def run_svr(df, 
            gt, 
            params, 
            random_seed, 
            splits, 
            test_size):

    DEBUG = True

    if params is None:
        # C = [1024, 2048, 4096]
        C = [4096]
        kernel = ['poly', 'rbf', 'sigmoid']
        epsilon = [0.1, 1, 10]
        # C = [10]
        # kernel = ['linear']
        # epsilon = [0.01]

        
    else:
        C = [params['C']]
        kernel = [params['kernel']]
        epsilon = [params['epsilon']]

    min_mse = float('inf')
    min_mae = float('inf')
    min_r2 = float('inf')

    params_and_result = {'training_perf':{}, 'test_perf':{}, 'params':{}}
    dataset_split = DatasetSplit()

    feat = list(df.columns)
    feat.remove(gt)
    feature_names = feat
    df_import = pd.DataFrame({'Feature': feature_names})

    # Make target to last column
    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)
    df = df[cols]

    # X_train_, X_test_, y_train_, y_test_ = dataset_split.\
    #     strat_train_test_split(df, None, None, gt, test_size, random_seed)

    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        train_test_split(df, gt, test_size, random_seed)
    
    for c in C:
        print('c: ' + str(c))
        for ker in kernel:
            print('ker: ' + str(ker))
            for eps in epsilon:
                print('eps: ' + str(eps))
                
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

                    reg = SVR(kernel=ker,
                              epsilon=eps,
                              C=c)
                    
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
                        print('#### MIN MSE SVR: ' + str(mse_mean))
                    
                    min_mse = mse_mean
                    min_mae = mae_mean
                    min_r2 = r2_mean

                    min_mse_std = mse_std
                    min_mae_std = mae_std
                    min_r2_std = r2_std

                    params_and_result['params']['C'] = c
                    params_and_result['params']['kernel'] = ker
                    params_and_result['params']['epsilon'] = eps

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

    reg = SVR(kernel = params_and_result['params']['kernel'],
              epsilon = params_and_result['params']['epsilon'],
              C = params_and_result['params']['C']
            )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    params_and_result['test_perf']['MSE'] = mean_squared_error(y_test, y_pred)
    params_and_result['test_perf']['MAE'] = mean_absolute_error(y_test, y_pred)
    params_and_result['test_perf']['R2'] = r2_score(y_test, y_pred) 


    import_result = permutation_importance(reg, X_train, y_train, n_repeats=10,
                                           random_state=3)
    
    df_import['import_mean'] = import_result.importances_mean
    df_import['import_std'] = import_result.importances_std

    return params_and_result, df_import


def use_svr(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = SVR()
    else:
        reg = SVR(kernel=best_params['kernel'],
                  epsilon=best_params['epsilon'],
                  gamma=best_params['gamma'],
                  C=best_params['C'])

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = list(y_pred)
    y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred