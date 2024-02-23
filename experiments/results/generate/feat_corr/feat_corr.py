import pandas as pd
from utils.global_var import DEBUG
from experiments.features_creation import feat_manager
from experiments.results.pre_process_dataset import pre_process_dataset


def get_dataset_cor(df, gt, cor='spearman'):
    
    if DEBUG:
        print('## total lines df: ' + str(len(df.index)))

    df = df[(df[gt] != -1) & (df[gt] != 0)]

    if DEBUG:
        print('## total lines df after removal: ' + \
              str(len(df.index)))

    df_cor = df.corr(cor)
    df_cor = df_cor.abs()
    df_cor = df_cor.sort_values(by = [gt], ascending=False)

    # return df_cor[[gt]]
    return df_cor


def get_most_cor_feat(df_cor, gt, thres=0, rename={}):
    df_cor = df_cor[(df_cor[gt] > thres) | (df_cor[gt] < -thres)]
    want = list(df_cor.index.values)
    dont_want = [c for c in list(df_cor.columns) if c not in want]
    df_cor = df_cor.drop(columns=dont_want)
    df_cor = df_cor[want]

    if rename:
        df_cor = df_cor.rename(columns=rename, index=rename)
    
    return rename


def save_to_csv(df, sel_feat=None, gt=None, my_path=None, cor='spearman'):
    if sel_feat is not None:
        df_save = df[sel_feat + [gt]]
    else:
        df_save = df

    df_save = get_dataset_cor(df_save, gt, cor=cor)
    df_save.to_csv(my_path, sep='\t')

    print('saved to CSV!')


def select_cols(df, gt):
    cols = [
        # 'ABS_LOG_MAX_DEGREE',
        # 'ABS_LOG_DEGREE_ENTROPY',
        # 'ABS_LOG_LINK_DENSITY',
        # 'ABS_LOG_NODE_LINK_RATIO',
        # 'ABS_LOG_MAX_CLUST_COEF',
        # 'ABS_LOG_MEAN_CLUST_COEF',
        # 'ABS_LOG_MAX_BTW',
        # 'ABS_LOG_MEAN_BTW',
        # 'ABS_LOG_BTW_ENTROPY',
        # 'ABS_LOG_BTW_ENTROPY_NORM',
        # 'ABS_LOG_DFT_ENTROPY',
        # 'ABS_LOG_CC_ENTROPY',
        # 'ABS_LOG_CC_ENTROPY_NORM',
        # 'ABS_LOG_DEGREE_ASSORT',

        # 'LOG_EVENTS',
        # 'LOG_EVENTS_TYPES',
        # 'LOG_SEQS',
        # 'LOG_UNIQUE_SEQS',
        # 'LOG_PERC_UNIQUE_SEQS',
        # 'LOG_AVG_SEQS_LENGTH',
        # 'LOG_AVG_EDIT_DIST',

        # 'variant_entropy',
        # 'variant_entropy_norm',
        # 'sequence_entropy',
        # 'sequence_entropy_norm',

        'LOG_PERC_UNIQUE_SEQS',

        gt
    ]
    return df[cols]


if __name__ == '__main__':
    k = 1
    ground_truth = 'FITNESS'
    dataset_path = 'experiments/results/markov/k_' + str(k) + '/df_markov_k_' + \
                   str(k) + '.csv'
    output_path = 'experiments/results/reports/feat_importance/' + \
                  'all_feat_cor_k_' + str(k) + '_' + ground_truth[:3].lower() + '.csv'
    not_a_feature = ['EVENT_LOG', 
                     'DISCOVERY_ALG', 
                     'PRECISION', 
                     'FITNESS', 
                     'TIME_MARKOV']
                  
    # dataset_path = 'experiments/results/markov/dataset_markov_k2.csv'
    # dataset_path = 'experiments/results/markov/dataset_markov_k3.csv'
    sel_feat = None
    cor = 'pearson'
    aux_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'

    df_aux = pd.read_csv(
                      aux_path,
                      sep='\t',
             )
    df_aux = df_aux[['EVENT_LOG', 'DISCOVERY_ALG']]

    df = pd.read_csv(dataset_path, sep='\t')
    df = df.merge(df_aux,
                  on=['EVENT_LOG', 'DISCOVERY_ALG'],
                  how='inner')
    
    df = df[df['DISCOVERY_ALG'] == 'IMf']
    
    df = pre_process_dataset(df, ground_truth, not_a_feature)

    # df = select_cols(df, ground_truth)
    df_cor = get_dataset_cor(df, ground_truth, cor=cor)

    print(df_cor[ground_truth])

    save_to_csv(df, 
                sel_feat, 
                ground_truth, 
                output_path,
                cor)
    
    print('done!')


