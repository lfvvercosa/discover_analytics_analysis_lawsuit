from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
import pm4py

import pandas as pd

import core.my_loader as my_loader
from core import my_create_features


if __name__ == "__main__": 
    base_path = 'dataset/'
    input_path = 'dataset/tribunais_trabalho/dataset_raw.csv'
    out_path = 'dataset/tribunais_trabalho/dataset_model_v2.csv'
    ngram = 1
    trunc_min = 0.02
    trunc_max = 0.02
    infreq_thres = 0.02
    test_size = 0.2
    random_seed = 3
    metric_encode = 'median'
    DEBUG = True

    
    df_feat = pd.read_csv(input_path, sep='\t')
    
    # Prepair feature dataset to the model
    df_feat = my_create_features.process_features(df_feat,
                                                  base_path, 
                                                  ngram,
                                                  trunc_min,
                                                  trunc_max,
                                                  infreq_thres,
                                                  test_size,
                                                  random_seed,
                                                  metric_encode,
                                                  is_statistics=False)

    # Save it
    df_feat = df_feat.drop_duplicates()
    df_feat.to_csv(out_path, sep='\t', index=False)

    if DEBUG:
        print('dataset created!')

