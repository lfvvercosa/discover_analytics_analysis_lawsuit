import pandas as pd
import experiments.results.regressor.svr as svr
import experiments.results.regressor.mlp as mlp
import experiments.results.regressor.rfr as rfr
import experiments.results.regressor.lgbm as lgbm
import statistics
from experiments.results.pre_process_dataset import pre_process_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy import stats
from utils.global_var import DEBUG
from experiments.features_creation import feat_manager


def get_mean_std_res(res):
    for k in res:
        print(k + ' mean : ' + str(statistics.mean(res[k])))
        print(k + ' std  : ' + str(statistics.stdev(res[k])))
        print('')


if __name__ == '__main__':
    gt = 'FITNESS'
    not_a_feature = ['EVENT_LOG', 'DISCOVERY_ALG', gt]
    k1_path = 'experiments/results/markov/k_1' + \
              '/df_markov_k_1.csv'
    k2_path = 'experiments/results/markov/k_2' + \
              '/df_markov_k_2.csv'
    k3_path = 'experiments/results/markov/k_3' + \
              '/df_markov_k_3.csv'

    df_k1 = pd.read_csv(k1_path, sep='\t')
    df_k2 = pd.read_csv(k2_path, sep='\t')
    df_k3 = pd.read_csv(k3_path, sep='\t')

    df_k1 = df_k1[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'FITNESS',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
    ]]

    df_k1 = df_k1.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K1',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K1'})

    df_k2 = df_k2[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
    ]]

    df_k2 = df_k2.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K2',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K2'})

    df_k3 = df_k3[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
    ]]

    df_k3 = df_k3.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K3',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K3'})

    df = df_k1.merge(df_k2, 
                     how='inner',
                     on=['EVENT_LOG','DISCOVERY_ALG'])
    
    df = df.merge(df_k3, 
                  how='inner',
                  on=['EVENT_LOG','DISCOVERY_ALG'])

    df.to_csv('df_all.csv', sep='\t')

    before = len(df.index)
    df = pre_process_dataset(df, gt, not_a_feature)
    after = len(df.index)

    if DEBUG:
        print('# rows before pre-processing: ' + str(before))
        print('# rows after pre-processing: ' + str(after))

    res = {
        'SVR': {
            'spearman':[],
            'MAE':[],
            'r2_score':[],
        },
        'Linear': {
            'spearman':[],
            'MAE':[],
            'r2_score':[],
        },
        'MLP': {
            'spearman':[],
            'MAE':[],
            'r2_score':[],
        },
        'RFR':{
            'spearman':[],
            'MAE':[],
            'r2_score':[],
        },
        'LGBM': {
            'spearman':[],
            'MAE':[],
            'r2_score':[],
        },
    }

    best_params_rfr = rfr.tune_rfr(df,gt,not_a_feature)
    best_params_svr = svr.tune_svr(df, gt, not_a_feature)
    best_params_mlp = mlp.tune_mlp(df, gt, not_a_feature)
    best_params_lgbm = lgbm.tune_lgbm(df, gt, not_a_feature)
    # best_params_svr = None
    # best_params_mlp = None
    # best_params_lgbm = None


    li_reg = LinearRegression()

    feat = [f for f in df.columns if f not in not_a_feature]
    X = df[feat].to_numpy()
    y = df[gt].to_numpy()

    skf = KFold(n_splits=10, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    count = 1

    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        print('test ' + str(count) + ' running...')
        count += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred_svr = svr.use_svr(X_train, 
                                 y_train, 
                                 X_test, 
                                 best_params_svr)
        
        li_reg = li_reg.fit(X_train, y_train)
        y_pred_reg = li_reg.predict(X_test)
        y_pred_mlp = mlp.use_mlp(X_train, 
                                 y_train, 
                                 X_test, 
                                 best_params_mlp)
        y_pred_rfr = rfr.use_rfr(X_train, 
                                 y_train, 
                                 X_test, 
                                 best_params_rfr)
        

        y_pred_lgb = lgbm.use_lgbm(X_train, 
                                   y_train, 
                                   X_test, 
                                   best_params_lgbm)

        y_pred_lgb = lgbm.use_lgbm(X_train, 
                                   y_train, 
                                   X_test, 
                                   best_params_lgbm)

        res['SVR']['spearman'].append(
            stats.spearmanr(list(y_pred_svr),list(y_test)).correlation
        )
        res['SVR']['MAE'].append(
            mean_absolute_error(list(y_pred_svr),list(y_test))
        )
        res['SVR']['r2_score'].append(
            r2_score(y_test, y_pred_svr)
        )

        res['Linear']['spearman'].append(
            stats.spearmanr(list(y_pred_reg),list(y_test)).correlation
        )
        res['Linear']['MAE'].append(
            mean_absolute_error(list(y_pred_reg),list(y_test))
        )
        res['Linear']['r2_score'].append(
            r2_score(y_test, y_pred_reg)
        )

        res['MLP']['spearman'].append(
            stats.spearmanr(list(y_pred_mlp),list(y_test)).correlation
        )
        res['MLP']['MAE'].append(
            mean_absolute_error(list(y_pred_mlp),list(y_test))
        )
        res['MLP']['r2_score'].append(
            r2_score(y_test, y_pred_mlp)
        )

        res['RFR']['spearman'].append(
            stats.spearmanr(list(y_pred_rfr),list(y_test)).correlation
        )
        res['RFR']['MAE'].append(
            mean_absolute_error(list(y_pred_rfr),list(y_test))
        )
        res['RFR']['r2_score'].append(
            r2_score(y_test, y_pred_rfr)
        )

        res['LGBM']['spearman'].append(
            stats.spearmanr(list(y_pred_lgb),list(y_test)).correlation
        )
        res['LGBM']['MAE'].append(
            mean_absolute_error(list(y_pred_lgb),list(y_test))
        )
        res['LGBM']['r2_score'].append(
            r2_score(y_test, y_pred_lgb)
        )

    print('')
    print('### Results SVR:')
    get_mean_std_res(res['SVR'])

    print('')
    print('### Results Linear Regression:')
    get_mean_std_res(res['Linear'])
    
    print('')
    print('### Results MLP:')
    get_mean_std_res(res['MLP'])
    
    print('')
    print('### Results RFR:')
    get_mean_std_res(res['RFR'])

    print('')
    print('### Results LGBM:')
    get_mean_std_res(res['LGBM'])