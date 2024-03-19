import pandas as pd
import re
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from models.DatasetSplit import DatasetSplit
import models.lgbm_clas as lgbm_clas


def remove_a_posteriori_features(df):
    regexp = re.compile('EXTRAJUDICIAL')
    jud_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('SUSPENSOS')
    susp_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('RECURSOS')
    rec_cols = [c for c in df.columns if regexp.search(c)]

    rem_cols = jud_cols + susp_cols + rec_cols
    rem_cols += [
        'case:concept:name',
        'CASE:LAWSUIT:PERCENT_KEY_MOV',
        'TAXA_DE_CONGESTIONAMENTO_LIQUIDA',
        'TAXA_DE_CONGESTIONAMENTO_TOTAL',
        'ESTOQUE',
    ]

    df = df[[c for c in df.columns if c not in rem_cols]]
    

    return df


if __name__ == "__main__":
    DEBUG = True

    input_path = 'dataset/tribunais_trabalho/mini/dataset_trt_model.csv'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    dataset_split = DatasetSplit()
    test_size = 0.2
    random_seed = 3
    splits_kfold = 5
    number_cores = 6

    df = pd.read_csv(input_path, sep='\t')
    df = df.drop_duplicates()

    # df = dataset_split.gen_categories(df,None,None,gt)
    # df = df.rename(columns={'cat':'TEMPO_PROCESSUAL_CATEGORIA'})
    # df = df.drop(columns='TEMPO_PROCESSUAL_TOTAL_DIAS')

    df = remove_a_posteriori_features(df)

    if DEBUG:
        print('### Total records: ' + str(len(df.index)))
        print('### Total features: ' + str(len(df.columns) - 1))

    # params_and_results = lgbm_clas.get_best_params(
    #                                                df, 
    #                                                gt, 
    #                                                [gt], 
    #                                                None, 
    #                                                random_seed, 
    #                                                splits_kfold,
    #                                                number_cores,
    #                                                test_size
    #                                               )

    params = {'params': {'boosting_type': 'dart', 
                         'learning_rate': 0.1, 
                         'n_estimators': 1000, 
                         'objective': 'multiclassova',
                        }
             }

    print('best parameters: ' + str(params))

    
    # Show time of most severe bottlenecks for trace
    t1 = 0
    t2 = 365*1
    t3 = 365*4
    t4 = 365*8
    t5 = 365*100

    bins = pd.IntervalIndex.from_tuples([
                                         (t1,t2),
                                         (t2,t3),
                                         (t3,t4),
                                         (t4,t5),
                                         ]
                                        )
    # labels = ['very fast','fast','medium','slow','very slow']
    labels = [0,1,2,3]

    model, y_test_, y_pred_ = lgbm_clas.run_model(
                                   df, 
                                   gt,
                                   bins,
                                   labels,  
                                   params['params'], 
                                   random_seed, 
                                   number_cores,
                                   test_size
                              )   
    

    report = classification_report(y_test_, y_pred_)
    print('Classification report:\n', report)
    
    cm = confusion_matrix(y_test_, y_pred_, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()

    plt.show()



    print()
