import multiprocessing
import pandas as pd
from os import listdir
from os.path import isfile, join
from experiments.models.get_markov import get_markov_log
from features import features
from features import fitness_feat
from features import precision_feat
from features import LogFeatures
from features import petri_net_feat

from utils.read_and_write.files_handle import remove_suffix
from utils.creation.create_df_from_dict import df_from_dict
from utils.global_var import DEBUG

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from libraries import networkx_graph


def calcMetrics(Gm_log, ret_dict):
    ret_dict['ABS_LOG_MAX_DEGREE'] = \
        networkx_graph.calcOutDegree(Gm_log, 'max')
    ret_dict['ABS_LOG_MEAN_DEGREE'] = \
        networkx_graph.calcOutDegree(Gm_log, 'mean')
    ret_dict['ABS_LOG_DEGREE_ENTROPY'] = \
        networkx_graph.calcEntropyAux(Gm_log.degree(), 
                                        normalized=False)
    ret_dict['ABS_LOG_LINK_DENSITY'] = \
        networkx_graph.calcLinkDensity(Gm_log)
    ret_dict['ABS_LOG_MAX_CLUST_COEF'] = \
        networkx_graph.calcClustCoef(Gm_log, stat='max')
    ret_dict['ABS_LOG_MEAN_CLUST_COEF'] = \
        networkx_graph.calcClustCoef(Gm_log, stat='mean')
    ret_dict['ABS_LOG_MAX_BTW'] = \
        networkx_graph.calcBetweenness(Gm_log, stat='max')
    ret_dict['ABS_LOG_MEAN_BTW'] = \
        networkx_graph.calcBetweenness(Gm_log, stat='mean')
    ret_dict['ABS_LOG_NODE_LINK_RATIO'] = \
        networkx_graph.calcNodeLinkRatio(Gm_log)
    ret_dict['ABS_LOG_BTW_ENTROPY'] = \
        networkx_graph.calcBetweennessEntropy(Gm_log,
                                                'weight',
                                                False)
    ret_dict['ABS_LOG_BTW_ENTROPY_NORM'] = \
        networkx_graph.calcBetweennessEntropy(Gm_log,
                                                'weight',
                                                True)
    ret_dict['ABS_LOG_DFT_ENTROPY'] = \
        networkx_graph.calcEntropyDFTEigenvalues(Gm_log,
                                                    'weight',
                                                    False)
    ret_dict['ABS_LOG_CC_ENTROPY'] = \
        networkx_graph.calcEntropyClustering(Gm_log,
                                                'weight',
                                                False)
    ret_dict['ABS_LOG_CC_ENTROPY_NORM'] = \
        networkx_graph.calcEntropyClustering(Gm_log,
                                                'weight',
                                                True)
    ret_dict['ABS_LOG_DEGREE_ASSORT'] = \
        networkx_graph.degreeAssort(Gm_log,
                                    None)


def calcMetricsTimeout(Gm_log, timeout):
        manager = multiprocessing.Manager()
        ret_dict = manager.dict()

        p = multiprocessing.Process(target=calcMetrics,
                                    args=(
                                            Gm_log, 
                                            ret_dict,
                                          )
                                   )
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
        
            return None
    
        p.join()

        return ret_dict


if __name__ == '__main__':
    base_path = 'xes_files/'
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/features_creation/feat_log/' + \
               'feat_log_dfg.csv'
    count_rec = -1
    k_markov = 1
    timeout = 30*60
    # timeout = 1
    ref_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'
    
    feat = {
        'EVENT_LOG':{},

        # 'LOG_EVENTS':{},
        # 'LOG_EVENTS_TYPES':{},
        # 'LOG_SEQS':{},
        # 'LOG_UNIQUE_SEQS':{},
        # 'LOG_PERC_UNIQUE_SEQS':{},
        # 'LOG_AVG_SEQS_LENGTH':{},
        # 'LOG_AVG_EDIT_DIST':{},

        'ABS_LOG_MAX_DEGREE':{},
        'ABS_LOG_MEAN_DEGREE':{},
        'ABS_LOG_DEGREE_ENTROPY':{},
        'ABS_LOG_LINK_DENSITY':{},
        'ABS_LOG_NODE_LINK_RATIO':{},
        'ABS_LOG_MAX_CLUST_COEF':{},
        'ABS_LOG_MEAN_CLUST_COEF':{},
        'ABS_LOG_MAX_BTW':{},
        'ABS_LOG_MEAN_BTW':{},
        'ABS_LOG_BTW_ENTROPY':{},
        'ABS_LOG_BTW_ENTROPY_NORM':{},
        'ABS_LOG_DFT_ENTROPY':{},
        'ABS_LOG_CC_ENTROPY':{},
        'ABS_LOG_CC_ENTROPY_NORM':{},
        'ABS_LOG_DEGREE_ASSORT':{},    
    }
    
    df_ref = pd.read_csv(ref_path, sep='\t')

    for fol in folders:
            my_path = base_path + fol
            my_files = [f for f in listdir(my_path) \
                        if isfile(join(my_path, f))]

            for fil in my_files:
                record = remove_suffix(fil)

                if len(df_ref[
                    (df_ref[['EVENT_LOG']].values == \
                        record).all(axis=1)]) == 0:
                        print('Skipping record (not in df_markov_k1)...')
                        continue

                # df_ex = pd.read_csv(ref_path, sep='\t')

                # if len(df_ex[
                #     (df_ex[['EVENT_LOG']].values == \
                #         record).all(axis=1)]) == 1:
                #         print('Skipping record (already there)...')
                #         continue 

                Gm_log = get_markov_log(fil, k_markov)
                log = xes_importer.apply(my_path + fil)

                if Gm_log:
                    print('### Folder: ' + str(fol))
                    print('### Log: ' + str(fil))

                    ret_dict = calcMetricsTimeout(Gm_log, timeout)

                    if not ret_dict:
                        print('Skipping record (Timeout!)...')
                        continue

                    count_rec += 1

                    feat['EVENT_LOG'][count_rec] = remove_suffix(fil)
                    
                    # feat['LOG_EVENTS'][count_rec] = \
                    #     log_feat.number_events(log)
                    # feat['LOG_EVENTS_TYPES'][count_rec] = \
                    #     log_feat.number_events_types(log)
                    # feat['LOG_SEQS'][count_rec] = \
                    #     log_feat.number_sequences(log)
                    # feat['LOG_UNIQUE_SEQS'][count_rec] = \
                    #     log_feat.number_unique_seqs(log)
                    # feat['LOG_PERC_UNIQUE_SEQS'][count_rec] = \
                    #     log_feat.percent_unique_seqs(log)
                    # feat['LOG_AVG_SEQS_LENGTH'][count_rec] = \
                    #     log_feat.avg_sequence_length(log)
                    # feat['LOG_AVG_EDIT_DIST'][count_rec] = \
                    #     log_feat.log_edit_distance(log)

                    feat['ABS_LOG_MAX_DEGREE'][count_rec] = \
                        ret_dict['ABS_LOG_MAX_DEGREE']
                    feat['ABS_LOG_MEAN_DEGREE'][count_rec] = \
                        ret_dict['ABS_LOG_MEAN_DEGREE']
                    feat['ABS_LOG_DEGREE_ENTROPY'][count_rec] = \
                        ret_dict['ABS_LOG_DEGREE_ENTROPY']
                    feat['ABS_LOG_LINK_DENSITY'][count_rec] = \
                        ret_dict['ABS_LOG_LINK_DENSITY']
                    feat['ABS_LOG_MAX_CLUST_COEF'][count_rec] = \
                        ret_dict['ABS_LOG_MAX_CLUST_COEF']
                    feat['ABS_LOG_MEAN_CLUST_COEF'][count_rec] = \
                        ret_dict['ABS_LOG_MEAN_CLUST_COEF']
                    feat['ABS_LOG_MAX_BTW'][count_rec] = \
                        ret_dict['ABS_LOG_MAX_BTW']
                    feat['ABS_LOG_MEAN_BTW'][count_rec] = \
                        ret_dict['ABS_LOG_MEAN_BTW']
                    feat['ABS_LOG_NODE_LINK_RATIO'][count_rec] = \
                        ret_dict['ABS_LOG_NODE_LINK_RATIO']
                    feat['ABS_LOG_BTW_ENTROPY'][count_rec] = \
                        ret_dict['ABS_LOG_BTW_ENTROPY']
                    feat['ABS_LOG_BTW_ENTROPY_NORM'][count_rec] = \
                        ret_dict['ABS_LOG_BTW_ENTROPY_NORM']
                    feat['ABS_LOG_DFT_ENTROPY'][count_rec] = \
                        ret_dict['ABS_LOG_DFT_ENTROPY']
                    feat['ABS_LOG_CC_ENTROPY'][count_rec] = \
                        ret_dict['ABS_LOG_CC_ENTROPY']
                    feat['ABS_LOG_CC_ENTROPY_NORM'][count_rec] = \
                        ret_dict['ABS_LOG_CC_ENTROPY_NORM']
                    feat['ABS_LOG_DEGREE_ASSORT'][count_rec] = \
                        ret_dict['ABS_LOG_DEGREE_ASSORT']

                    df_from_dict(feat, out_path)
                else:
                    print('### Skipping log: ' + fil)


    print('done!')             

                    

