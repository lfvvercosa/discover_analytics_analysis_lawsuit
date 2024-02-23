from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold


def tune_svr(df, gt, not_a_feature):
    print('#### TUNING SVR ####')

    C = [1, 2, 4, 8, 16, 24, 32]
    # C = [1, 2, 4, 8, 16, 24, 32, 48]
    # C = [1, 2]

    epsilon = [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
    kernel = ['linear', 'poly']
    gamma = ['scale', 'auto']
    best_params = {}
    min_mse = float('inf')
    splits = 10
    feat = [f for f in df.columns if f not in not_a_feature]
        
    for curr_c in C:
        print("### CURRENT 'C': " + str(curr_c))
        for eps in epsilon:
            # print("### CURRENT 'EPS': " + str(eps))
            for ker in kernel:
                for ga in gamma:

                        X = df[feat].to_numpy()
                        y = df[gt].to_numpy()
                        skf = KFold(n_splits=splits, shuffle=True, random_state=1)
                        skf.get_n_splits(X, y)
                        mse = 0

                        for train_index, test_index in skf.split(X, y):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]

                            reg = SVR(kernel=ker,
                                    epsilon=eps,
                                    gamma=ga,
                                    C=curr_c)
                            reg.fit(X_train, y_train)

                            y_pred = reg.predict(X_test)
                            y_pred = [y if y > 0 else 0 for y in y_pred]
                            mse += mean_squared_log_error(y_test, y_pred)

                        mse /= splits

                        if mse < min_mse:
                            print('#### MIN MSE SVR: ' + str(min_mse))
                            min_mse = mse
                            best_params['C'] = curr_c
                            best_params['epsilon'] = eps
                            best_params['kernel'] = ker
                            best_params['gamma'] = ga

    print('### BEST_PARAMS SVR: ' +str(best_params))     
    return best_params


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