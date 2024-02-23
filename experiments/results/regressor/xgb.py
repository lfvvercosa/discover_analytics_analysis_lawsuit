from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold


def tune_xgb(df, gt, not_a_feature):
    # print('#### TUNING XGB ####')

    C = [1, 2, 4, 8, 16, 24, 32]
    # C = [1, 2, 4, 8, 16, 24, 32, 48]
    # C = [1, 2, 4, 8, 16]

    n_estim = [5, 50, 100, 200, 500]
    learn_rate = [0.1, 0.3, 0.5]
    boost = ['gbtree', 'gblinear']
    best_params = {}
    min_mse = float('inf')
    splits = 10
    feat = [f for f in df.columns if f not in not_a_feature]
        
    for n_est in n_estim:
        # print("### CURRENT 'N': " + str(n_est))

        for lear in learn_rate:
            for boo in boost:
                X = df[feat].to_numpy()
                y = df[gt].to_numpy()
                skf = KFold(n_splits=splits, shuffle=True, random_state=1)
                skf.get_n_splits(X, y)
                mse = 0

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    reg = XGBRegressor(n_estimators=n_est,
                                        booster=boo,
                                        learning_rate=lear)
                    reg.fit(X_train, y_train)

                    y_pred = reg.predict(X_test)
                    y_pred = [y if y > 0 else 0 for y in y_pred]
                    mse += mean_squared_log_error(y_test, y_pred)

                mse /= splits

                if mse < min_mse:
                    print('#### MIN MSE XGB: ' + str(min_mse))
                    min_mse = mse
                    best_params['n_estimators'] = n_est
                    best_params['learning_rate'] = lear
                    best_params['booster'] = boo

    print('### BEST_PARAMS XGB: ' +str(best_params))     
    return best_params


def use_xgb(X_train, y_train, X_test, best_params=None):
    if not best_params:
        reg = XGBRegressor()
    else:
        reg = XGBRegressor(n_estimators=best_params['n_estimators'],
                           booster=best_params['booster'],
                           learning_rate=best_params['learning_rate'])

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = list(y_pred)
    y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred