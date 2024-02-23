import pandas as pd
import experiments.results.regressor.svr as svr
import experiments.results.regressor.mlp as mlp
import experiments.results.regressor.rfr as rfr
#import experiments.results.regressor.lgbm as lgbm
import experiments.results.regressor.xgb as xgb
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


def get_path(k):
    if k == -1:
        k = 1
    
    return 'experiments/results/markov/k_' + str(k) + \
              '/df_markov_k_' + str(k) + '.csv'


if __name__ == '__main__':
    k = 2
    my_path = get_path(k)
    n_splits = 5

    print('#### Markov K = ' + str(k))

    aux_path = 'experiments/results/markov/k_3/df_markov_k_3.csv'
    gt = 'FITNESS'

    # gt ='PRECISION'
    not_a_feature = ['EVENT_LOG', 'DISCOVERY_ALG', 'PRECISION', 'FITNESS', 'TIME_MARKOV']
    # not_a_feature = [gt]
    # feat_selec_list = ['TOP1_ABS','TOP10', 'RF', 'XGB','SUB']
    feat_selec_list = ['TOP1']
    #'TOP7_RF', 'TOP5_RF', 'TOP3_RF',,'TOP7_XGB', 'TOP5_XGB','TOP3_XGB', 'TRSH15_XGB'
    xgb_preds = {'FITNESS':[], 'XGB_PRED':[]}
    # xgb_preds = {'PRECISION':[], 'XGB_PRED':[]}



    for feat_selec in feat_selec_list:

        print()
        print('##########################')
        print('#### FEAT SELECT: ' + str(feat_selec))
        print('##########################')
        print()

        if feat_selec == 'SUB':
            sel_feat = feat_manager.sel_subset_feat(gt)

        if feat_selec == 'TOP1':
            sel_feat = feat_manager.sel_top_feat(k, gt)

        if feat_selec == 'TOP10':
            sel_feat = feat_manager.sel_top_10_feat(k, gt)

        if feat_selec == 'RF':
            sel_feat = feat_manager.select_rf_features(k, gt)

        if feat_selec == 'XGB':
            sel_feat = feat_manager.select_xgb_features(k, gt)
        
        if feat_selec == 'TOP1_ABS':
            sel_feat = feat_manager.sel_top_abs_feat(k, gt)

        if feat_selec == 'TOP10_ABS':
            sel_feat = feat_manager.sel_top_10_abs_feat(k, gt)

        if feat_selec == 'TOP7_RF':
            sel_feat = feat_manager.sel_top_7_rf_feat(k,gt)
        
        if feat_selec == 'TOP5_RF':
            sel_feat = feat_manager.sel_top_5_rf_feat(k,gt)

        if feat_selec == 'TOP3_RF':
            sel_feat = feat_manager.sel_top_3_rf_feat(k,gt)

        if feat_selec == 'TRSH15_RF': 
            sel_feat = feat_manager.sel_treshold15_rf_feat(k,gt) 

        if feat_selec == 'TOP7_XGB':
            sel_feat = feat_manager.sel_top_7_xgb_feat(k,gt) 
        
        if feat_selec == 'TOP5_XGB':
            sel_feat = feat_manager.sel_top_5_xgb_feat(k,gt) 
        
        if feat_selec == 'TOP3_XGB':
            sel_feat = feat_manager.sel_top_3_xgb_feat(k,gt) 
        
        if feat_selec == 'TRSH15_XGB': 
            sel_feat = feat_manager.sel_treshold15_xgb_feat(k,gt)

        print('### selected features:')
        print(sel_feat)

        df = pd.read_csv(
                        my_path,
                        sep='\t'
                        )
        # df = df[df['DISCOVERY_ALG'] == 'IMd']
        df_aux = pd.read_csv(
                        aux_path,
                        sep='\t',
                )

        before = len(df.index)

        df_aux = df_aux[['EVENT_LOG', 'DISCOVERY_ALG']]


        df = df.merge(df_aux,
                    on=['EVENT_LOG', 'DISCOVERY_ALG'],
                    how='inner')

        after = len(df.index)

        if DEBUG:
            print('# rows before inner join: ' + str(before))
            print('# rows after inner join: ' + str(after))

        df = df[sel_feat + not_a_feature]

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
            'XGB': {
                'spearman':[],
                'MAE':[],
                'r2_score':[],
            },
        }

        best_params_rfr = None
        best_params_svr = None
        best_params_mlp = None
        best_params_xgb = None
        
        best_params_svr = svr.tune_svr(df, gt, not_a_feature)
        # best_params_rfr = rfr.tune_rfr(df,gt,not_a_feature)
        # best_params_mlp = mlp.tune_mlp(df, gt, not_a_feature)
        # best_params_lgbm = lgbm.tune_lgbm(df, gt, not_a_feature)
        best_params_xgb = xgb.tune_xgb(df, gt, not_a_feature)

        li_reg = LinearRegression()

        feat = [f for f in df.columns if f not in not_a_feature]
        X = df[feat].to_numpy()
        y = df[gt].to_numpy()

        skf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
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
            

            # y_pred_lgb = lgbm.use_lgbm(X_train, 
            #                            y_train, 
            #                            X_test, 
            #                            best_params_lgbm)

            y_pred_xgb = xgb.use_xgb(X_train, 
                                    y_train, 
                                    X_test, 
                                    best_params_xgb)

            # xgb_preds['FITNESS'] += list(y_test)

            # xgb_preds['PRECISION'] += list(y_test)

            # xgb_preds['XGB_PRED'] += list(y_pred_xgb)


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

            # res['LGBM']['spearman'].append(
            #     stats.spearmanr(list(y_pred_lgb),list(y_test)).correlation
            # )
            # res['LGBM']['MAE'].append(
            #     mean_absolute_error(list(y_pred_lgb),list(y_test))
            # )
            # res['LGBM']['r2_score'].append(
            #     r2_score(y_test, y_pred_lgb)
            # )

            res['XGB']['spearman'].append(
                stats.spearmanr(list(y_pred_xgb),list(y_test)).correlation
            )
            res['XGB']['MAE'].append(
                mean_absolute_error(list(y_pred_xgb),list(y_test))
            )
            res['XGB']['r2_score'].append(
                r2_score(y_test, y_pred_xgb)
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

        # print('')
        # print('### Results LGBM:')
        # get_mean_std_res(res['LGBM'])

        print('')
        print('### Results XGB:')
        get_mean_std_res(res['XGB'])

        df_preds = pd.DataFrame.from_dict(xgb_preds)
        df_preds = df_preds.round(4)
        df_preds.to_csv('preds_xgb.csv', sep='\t', index=False)
        print('done!')