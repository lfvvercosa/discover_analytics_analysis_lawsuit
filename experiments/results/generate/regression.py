import pandas as pd
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from experiments.results.pre_process_dataset import pre_process_dataset
from utils.creation.create_df_from_dict import df_from_dict


if __name__ == '__main__':
    k = 3
    ground_truth = 'PRECISION'
    dataset_path = 'experiments/results/markov/k_' + str(k) + '/df_markov_k_' + \
                   str(k) + '.csv'
    output_path = 'experiments/results/reports/metric_prediction/' + \
                  'feat_pred_k_' + str(k) + '_' + \
                  ground_truth[:3].lower() + '.csv'
    
    n_simu = 10
    test_percent = 0.25
    r2_score_feats = {}
    not_feat = [
        'EVENT_LOG',	
        'DISCOVERY_ALG',	
        'PRECISION',	
        'FITNESS',
    ]
    df = pd.read_csv(
                      dataset_path,
                      sep='\t'
                    )
    df = pre_process_dataset(df, ground_truth, not_feat)
    cols = [f for f in df.columns if f not in not_feat]

    for f in cols:
        res = []
        reg = LinearRegression()

        for n in range(n_simu):
            X = df[[f]]
            y = df[ground_truth]

            X_train, X_test, y_train, y_test = \
            train_test_split(X, 
                             y,
                             shuffle=True, 
                             test_size=test_percent,
                             random_state=n)
            
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            res.append(r2_score(y_test, y_pred))
        
        r2_score_feats[f] = {}
        r2_score_feats[f]['mean'] = round(statistics.mean(res), 4)
        r2_score_feats[f]['std'] = round(statistics.stdev(res), 4)
    
    df = pd.DataFrame.from_dict(r2_score_feats)
    df = df.T
    df = df.sort_values(by=['mean'], ascending=False)
    df.to_csv(output_path, sep='\t')

    print('done!')
