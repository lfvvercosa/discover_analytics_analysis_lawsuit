import pandas as pd


def rename_and_select_cols(df_gt):
    df_gt = df_gt.rename(columns={
                            'event_log':'EVENT_LOG',
                            'algorithm':'DISCOVERY_ALG',
                            'precision_alignment':'PRECISION',
                            'fitness_alignment':'FITNESS'
                        })
    
    df_gt = df_gt[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'PRECISION',
        'FITNESS',
    ]]

    return df_gt


if __name__ == '__main__':

    base_path = 'experiments/results/'
    metrics_path = base_path + 'metrics_mixed_dataset.csv'
    k_markov = '1'

    feat_path = 'experiments/features_creation/feat_markov/' + \
                            'feat_markov_k_' + k_markov + '.csv'
    feat_path_dfg = 'experiments/features_creation/feat_markov/' + \
                            'feat_markov_k_1_dfg.csv'
    log_feat_path = 'experiments/features_creation/feat_log/feat_log.csv'
    log_feat_dfg_path = 'experiments/features_creation/feat_log/feat_log_dfg.csv'
    pn_feat_path = 'experiments/features_creation/feat_pn/feat_pn.csv'
    subset_feat_path = 'experiments/features_creation/feat_subset/results_subset_metrics.csv'
    time_feat_path = 'experiments/results/markov/k_' + k_markov + \
                     '/model_perf_k_' + k_markov + '.csv'
    bl_feat_path = 'experiments/features_creation/feat_baseline/feat_baseline.csv'
    extra_feat_path = 'experiments/features_creation/feat_markov/' + \
                            'extra_feat_markov_k_' + k_markov + '.csv'
    align_1_path = 'experiments/features_creation/feat_markov/' + \
                   'feat_markov_align_1_k_1.csv'
    align_2_path = 'experiments/features_creation/feat_markov/' + \
                   'feat_markov_align_2_k_1.csv'
    align_3_path = 'experiments/features_creation/feat_markov/' + \
                   'feat_markov_align_3_k_1.csv'
    align_4_path = 'experiments/features_creation/feat_markov/' + \
                   'feat_markov_align_4_k_1.csv'

    entropy_feat_path = 'experiments/Automato prefix/Entropy_tabel.csv'

    output_path = base_path + 'markov/k_' + k_markov + '/df_markov_k_' + \
                            k_markov + '.csv'
    merge_cols = ['EVENT_LOG','DISCOVERY_ALG']

    # load dataset with GTs
    df_gt = pd.read_csv(metrics_path,
                        sep=',')

    df_gt = rename_and_select_cols(df_gt)

    df_gt = df_gt.drop_duplicates(merge_cols)

    # load features dataset
    df_feat = pd.read_csv(feat_path,
                        sep=',')
    df_feat = df_feat.drop_duplicates(merge_cols)

    size_df_feat = len(df_feat.index)

    df_new = df_gt.merge(df_feat,
                on=merge_cols,
                how='right')

    size_df_new = len(df_new.index)


    if size_df_new != size_df_feat:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_feat))

    # load features dataset DFG
    df_feat_dfg = pd.read_csv(feat_path_dfg,
                        sep='\t')
    df_feat_dfg = df_feat_dfg.drop_duplicates(merge_cols)

    size_df_feat_dfg = len(df_feat_dfg.index)

    df_new = df_new.merge(df_feat_dfg,
                		  on=merge_cols,
            			  how='left')

    size_df_new = len(df_new.index)

    if size_df_new != size_df_feat_dfg:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_feat_dfg))


    # remove duplicates
    cols = list(df_new.columns)
    cols = [c for c in cols if c not in merge_cols or 'LOG_' not in c]
    before = len(df_new.index)
    df_new = df_new.drop_duplicates(cols)
    after = len(df_new.index)

    print('removed ' + str(after-before) + ' duplicates')
    print(df_new)

    # load log features dataset
    df_log_feat = pd.read_csv(log_feat_path,
                              sep='\t')
    
    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_log_feat,
                          how='left',
                          on=['EVENT_LOG'])

    size_df_new_after = len(df_new.index)
    
    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load entropy features dataset
    df_entropy_feat = pd.read_csv(entropy_feat_path,
                                  sep=',')
    
    df_new = df_new.merge(df_entropy_feat,
                          how='left',
                          on=['EVENT_LOG'])
    
    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))


     # load petri-net features dataset
    df_pn = pd.read_csv(pn_feat_path,
                            sep='\t')

    size_df_new_before = len(df_new.index)
    df_new = df_new.merge(df_pn,
                          on=merge_cols,
                          how='left')
    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load markov time feature
    df_time = pd.read_csv(time_feat_path,
                          sep='\t')
    df_time = df_time.rename(columns={'event_log':'EVENT_LOG',
                                      'algorithm':'DISCOVERY_ALG',
                                      'total_time':'TIME_MARKOV'})
    df_time = df_time.drop(['tran_sys_time',
                            'dfa_time',
                            'reduc_time',
                            'markov_time'], axis=1)

    size_df_new_before = len(df_new.index)
    df_new = df_new.merge(df_time,
                          on=merge_cols,
                          how='left')
    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load baseline features dataset
    df_bl = pd.read_csv(bl_feat_path,
                        sep='\t')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_bl,
                          on=merge_cols,
                          how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load extra features dataset
    df_extra = pd.read_csv(extra_feat_path,
                        sep='\t')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_extra,
                          on=merge_cols,
                          how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # if k_markov == '1':

    # load alignment markov 2 features dataset
    df_align2 = pd.read_csv(align_2_path,
                            sep='\t')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_align2,
                        on=merge_cols,
                        how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))
    
    # load alignment markov 1 features dataset
    df_align1 = pd.read_csv(align_1_path,
                            sep=',')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_align1,
                        on=merge_cols,
                        how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))


    # load alignment markov 3 features dataset
    df_align3 = pd.read_csv(align_3_path,
                            sep='\t')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_align3,
                        on=merge_cols,
                        how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))


    

    # load subset features dataset
    df_subset = pd.read_csv(subset_feat_path,
                            sep=',')
    df_subset = df_subset.drop(['FITNESS_ALIGNMENT','PRECISION_ALIGNMENT'], axis=1)


    size_df_new_before = len(df_new.index)

    df_new['EVENT_LOG'] = df_new['EVENT_LOG'].str.replace(r'\.xes\.gz$', '')
    df_new['EVENT_LOG'] = df_new['EVENT_LOG'].str.replace(r'\.xes$', '')

    df_new = df_new.merge(df_subset,
                          on=merge_cols,
                          how='left')


    size_df_new_after = len(df_new.index)
    
    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load log DFG features dataset
    df_dfg_feat = pd.read_csv(log_feat_dfg_path,
                              sep='\t')
    
    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_dfg_feat,
                          how='left',
                          on=['EVENT_LOG'])

    size_df_new_after = len(df_new.index)
    
    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # load alignment markov 4 features dataset
    df_align4 = pd.read_csv(align_4_path,
                            sep='\t')

    size_df_new_before = len(df_new.index)

    df_new = df_new.merge(df_align4,
                          on=merge_cols,
                          how='left')

    size_df_new_after = len(df_new.index)

    if size_df_new_before != size_df_new_after:
        raise Exception('more lines than it should!')
    else:
        print('OK! # lines: ' + str(size_df_new_after))

    # check duplicated cols
    cols = list(df_new.columns)

    if len(cols) > len(set(cols)):
        raise Exception('repeated cols!')

    df_new = df_new.round(4)

    # save new dataset
    df_new.to_csv(path_or_buf = output_path,
                sep='\t',
                header=True,
                index=False)

    print('done!')