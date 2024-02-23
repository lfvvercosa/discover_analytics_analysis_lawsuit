import pandas as pd
import numpy as np
import statsmodels.api as sm


def add_constant(X):
    X['CONST'] = 1
    first = X.pop('CONST')
    X.insert(0, 'CONST', first)

    return X


if __name__ == '__main__':
    k = 2
    ground_truth = 'FITNESS'
    not_a_feature = ['EVENT_LOG', 'DISCOVERY_ALG']
    dataset_path = 'experiments/results/markov/k_' + str(k) + \
                   '/df_markov_k_' +  str(k) + '.csv'
    output_path = 'experiments/results/reports/feat_importance/' + \
                  'all_feat_cor_k_' + str(k) + '_' + \
                   ground_truth[:3].lower() + '_backwards.csv'
    
    df = pd.read_csv(dataset_path, sep='\t')
    feats = list(df.columns)
    feats.remove(ground_truth)
    feats = [f for f in feats if f not in not_a_feature]

    X = df[feats]
    y = df[[ground_truth]]

    cols = list(X.columns)
    pmax = 1

    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        # X_1 = sm.add_constant(X_1)
        X_1 = add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()

        if(pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break

selected_features_BE = cols
print('selected features: ' + str(selected_features_BE))
    