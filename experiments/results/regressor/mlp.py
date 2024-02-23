from sklearn.metrics import mean_squared_log_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
                        

def tune_mlp(df, gt, not_a_feature):
    print('#### TUNING MLP ####')

    hidden_layer_sizes=[(50), (100), (50,50), (100,100), (200,200), (100,100,100), \
                        (200,200,200)]
    max_iter=[200, 500, 1000, 1500, 2000]
    early_stopping = [True, False]
    best_params = {}
    min_mse = float('inf')
    splits = 10
    feat = [f for f in df.columns if f not in not_a_feature]

    for hidden in hidden_layer_sizes:
        for max_i in max_iter:
            for early in early_stopping:

                X = df[feat].to_numpy()
                y = df[gt].to_numpy()
                skf = KFold(n_splits=splits, shuffle=True, random_state=1)
                skf.get_n_splits(X, y)
                mse = 0

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]


                    reg = MLPRegressor(hidden_layer_sizes=hidden,
                                    max_iter=max_i,
                                    early_stopping=early)
                    reg.fit(X_train, y_train)

                    y_pred = reg.predict(X_test)
                    y_pred = [y if y > 0 else 0 for y in y_pred]
                    mse += mean_squared_log_error(y_test, y_pred)

                mse /= splits

                if mse < min_mse:
                    print('#### MIN MSE MLP: ' + str(min_mse))
                    min_mse = mse
                    best_params['hidden_layer_sizes'] = hidden
                    best_params['max_iter'] = max_i
                    best_params['early_stopping'] = early

    print('### BEST_PARAMS MLP: ' +str(best_params))     
    return best_params


def use_mlp(X_train, y_train, X_test, best_params=None):

    if not best_params:
        reg = MLPRegressor()
    else:
        reg = MLPRegressor(hidden_layer_sizes=\
                                best_params['hidden_layer_sizes'],
                           max_iter=\
                                best_params['max_iter'],
                           early_stopping=\
                                best_params['early_stopping'])

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = list(y_pred)
    y_pred = [x if x > 0 else 0 for x in y_pred]

    return y_pred