from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold


def tune_lgbm(df, gt, not_a_feature):
    boost_type = ['gbdt']
    learn_rate = [0.01, 0.1, 0.2, 0.4]
    n_estim = [10, 50, 100, 200]
    splits = 10

    min_mse = float('inf')
    best_params = {}
    feat = [f for f in df.columns if f not in not_a_feature]
    
    for boost in boost_type:
        for learn in learn_rate:
            for n in n_estim:

                X = df[feat].to_numpy()
                y = df[gt].to_numpy()
                skf = KFold(n_splits=splits, shuffle=True, random_state=1)
                skf.get_n_splits(X, y)
                mse = 0

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    reg = LGBMRegressor(boosting_type=boost,
                                        learning_rate=learn,
                                        n_estimators=n)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    y_pred = [y if y > 0 else 0 for y in y_pred]

                    mse += mean_squared_log_error(y_test, y_pred)

                mse /= splits

                if mse < min_mse:
                    print('#### MIN MSE LGBM: ' + str(min_mse))
                    
                    min_mse = mse

                    best_params['boosting_type'] = boost
                    best_params['learning_rate'] = learn
                    best_params['n_estimators'] = n

    print('### best params LGBM: ' +str(best_params))

    return best_params


def use_lgbm(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = LGBMRegressor()
    else:
        
        boost = best_params['boosting_type'] 
        learn = best_params['learning_rate'] 
        n = best_params['n_estimators'] 

        reg = LGBMRegressor(boosting_type=boost,
                            learning_rate=learn,
                            n_estimators=n)

        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        y_pred = list(y_pred)
        y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred