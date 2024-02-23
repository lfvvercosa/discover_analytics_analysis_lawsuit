##RandomForestRegressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
   


def tune_rfr(df, gt, not_a_feature):
    print('#### TUNING RFR ####')

    #HiperParametros
    tree_num = [5, 10, 25, 50]
    # crit = ["squared_error", "absolute_error", "poisson"]
    profundidades = [2, 3, 5, 7, 9]
    #min_amostras = [2, 3, 4, 5, 6]
    # min_amostras_leaf = [1,2,5,8,13,21]
    # min_weight_fraction_leaf=0.0
    max_fectu= ["auto", "sqrt", "log2"]
    splits = 10

    #Determinando os dados de teste e treinamento

    feat = [f for f in df.columns if f not in not_a_feature]

    #Procurando os melhores HiperParametros
    min_mse = float('inf')
    best_params = {}

    for tn in tree_num:
        print(str(tn) + " Trees")
        for pr in profundidades:
            # for mal in min_amostras_leaf:
            for mf in max_fectu :
                # for cr in crit: 

                X = df[feat].to_numpy()
                y = df[gt].to_numpy()
                skf = KFold(n_splits=splits, shuffle=True, random_state=1)
                skf.get_n_splits(X, y)
                mse = 0

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    arvore = RandomForestRegressor(n_estimators=tn,
                                                max_depth = pr, 
                                                max_features = mf,
                                                random_state=0, 
                                                n_jobs=-1)
                    arvore.fit(X_train, y_train)
                    y_pred = arvore.predict(X_test)
                    mse += mean_squared_log_error(y_test, y_pred)

                mse /= splits

                if mse < min_mse:
                    print('#### MIN MSE RFR: ' + str(min_mse))
                    min_mse = mse
                    best_params['tree_num'] = tn
                    # best_params['min_amostras_leaf'] = mal 
                    best_params['max_fectu'] = mf 
                    best_params['profundidades'] = pr
                    # best_params['crit'] = cr

    print('### BEST_PARAMS RFR: ' +str(best_params)) 
    return best_params


def use_rfr(X_train, y_train, X_test, best_params=None):
    if not best_params:
        arvore = RandomForestRegressor(n_estimators = 100, n_jobs=-1)
    else:
        arvore = RandomForestRegressor(n_estimators=best_params['tree_num'], 
                                       max_depth = best_params['profundidades'], 
                                       max_features = best_params['max_fectu'],
                                       random_state=0, 
                                       n_jobs=-1)
        
    arvore.fit(X_train, y_train)
    y_pred = arvore.predict(X_test)
    y_pred = list(y_pred)
    y_pred = [x if x > 0 else 0 for x in y_pred]
    
    return y_pred







