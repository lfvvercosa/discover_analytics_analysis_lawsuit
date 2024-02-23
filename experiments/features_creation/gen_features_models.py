import multiprocessing
from os import listdir
from os.path import isfile, join
from experiments.models.get_markov import get_markov_log, \
                                          get_markov_model
import pandas as pd

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


def calcMetrics(Gm_model, Gm_log, ret_dict):
    ret_dict['ABS_MEAN_DEGREE'] = \
        features.out_degree_diff(Gm_model, Gm_log, 'mean')
    ret_dict['ABS_MEAN_DEGREE_DIV'] = \
        features.out_degree_div(Gm_model, Gm_log, 'mean')
    
    ret_dict['ABS_NODE_LINK_RATIO'] = \
        features.node_link_ratio_diff(Gm_model, Gm_log)
    ret_dict['ABS_NODE_LINK_RATIO_DIV'] = \
        features.node_link_ratio_div(Gm_model, Gm_log)

    ret_dict['ABS_LINK_DENSITY'] = \
        features.link_density_diff(Gm_model, Gm_log)
    ret_dict['ABS_LINK_DENSITY_DIV'] = \
        features.link_density_div(Gm_model, Gm_log)
    
    ret_dict['ABS_MEAN_CLUST'] = \
        features.clust_diff(Gm_model, Gm_log, stat='mean')
    ret_dict['ABS_MEAN_CLUST_DIV'] = \
        features.clust_div(Gm_model, Gm_log, stat='mean')
    
    ret_dict['ABS_MAX_BTW'] = \
        features.between_diff(Gm_model, Gm_log, stat='max')
    ret_dict['ABS_MAX_BTW_DIV'] = \
        features.between_div(Gm_model, Gm_log, stat='max')

    ret_dict['ABS_MEAN_BTW'] = \
        features.between_diff(Gm_model, Gm_log, stat='mean')
    ret_dict['ABS_MEAN_BTW_DIV'] = \
        features.between_div(Gm_model, Gm_log, stat='mean')

    ret_dict['ABS_NODE_LINK_RATIO'] = \
        features.node_link_ratio_diff(Gm_model, Gm_log)
    ret_dict['ABS_NODE_LINK_RATIO_DIV'] = \
        features.node_link_ratio_div(Gm_model, Gm_log)

    ret_dict['ABS_BTW_ENTROPY'] = \
        features.betwee_entropy_diff(Gm_model, Gm_log, 'weight', False)
    ret_dict['ABS_BTW_ENTROPY_DIV'] = \
        features.betwee_entropy_div(Gm_model, Gm_log, 'weight', False)

    ret_dict['ABS_BTW_ENTROPY_NORM'] = \
        features.betwee_entropy_diff(Gm_model, Gm_log, 'weight', True)
    ret_dict['ABS_BTW_ENTROPY_NORM_DIV'] = \
        features.betwee_entropy_div(Gm_model, Gm_log, 'weight', True)

    ret_dict['ABS_DFT_ENTROPY'] = \
        features.fourier_entropy_diff_weighted(Gm_model, Gm_log)
    ret_dict['ABS_DFT_ENTROPY_DIV'] = \
        features.fourier_entropy_div_weighted(Gm_model, Gm_log)
        
    ret_dict['ABS_CC_ENTROPY'] = \
        features.clust_entropy_diff(Gm_model, Gm_log, 'weight', False)
    ret_dict['ABS_CC_ENTROPY_DIV'] = \
        features.clust_entropy_div(Gm_model, Gm_log, 'weight', False)

    ret_dict['ABS_CC_ENTROPY_NORM'] = \
        features.clust_entropy_diff(Gm_model, Gm_log, 'weight', True)

    ret_dict['ABS_DEGREE_ASSORT'] = \
        features.assortativity_diff(Gm_model, Gm_log)
    ret_dict['ABS_DEGREE_ASSORT_DIV'] = \
        features.assortativity_div(Gm_model, Gm_log)


def calcMetricsTimeout(Gm_model, Gm_log, timeout):
        manager = multiprocessing.Manager()
        ret_dict = manager.dict()

        p = multiprocessing.Process(target=calcMetrics,
                                    args=(
                                             Gm_model, 
                                             Gm_log, 
                                             ret_dict,
                                          )
                                   )

        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
        
            return -1
    
        p.join()

        return ret_dict

        
if __name__ == '__main__':
    base_path = 'xes_files/'
    base_path_pn = 'petri_nets/'
    k_markov = 1
    algs = ['IMf', 'IMd', 'ETM']
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/features_creation/feat_markov/' + \
               'TEST_feat_markov_k_' + str(k_markov) + '.csv'
    ref_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'
    count_rec = -1
    timeout = 30*60
    df_ref = pd.read_csv(ref_path, sep='\t')
    feat = {
        'EVENT_LOG':{},
        'DISCOVERY_ALG':{},

        # 'ALIGNMENTS_MARKOV':{},

        # 'ABS_EDGES_ONLY_IN_LOG':{},
        # 'ABS_EDGES_ONLY_IN_LOG_W':{},
        # 'ABS_EDGES_ONLY_IN_LOG_BTW':{},
        # 'ABS_EDGES_ONLY_IN_LOG_BTW_NORM':{},
        # 'ABS_EDGES_ONLY_IN_LOG_CC':{},
        # 'ABS_EDGES_ONLY_IN_MODEL':{},
        # 'ABS_EDGES_ONLY_IN_MODEL_W':{},
        
        # 'ABS_MAX_DEGREE':{},
        # 'ABS_MAX_DEGREE_DIV':{},
        # 'ABS_NODE_LINK_RATIO':{},
        
        # 'ABS_MAX_CLUSTER':{},
        # 'ABS_MAX_CLUSTER_DIV':{},
        
        # 'ABS_DEGREE_ENTROPY':{},
        # 'ABS_DEGREE_ENTROPY_DIV':{},
        
        # 'ABS_LINK_DENSITY':{},
        # 'ABS_LINK_DENSITY_DIV':{},

        'ABS_MEAN_DEGREE':{},
        'ABS_MEAN_DEGREE_DIV':{},

        'ABS_NODE_LINK_RATIO':{},
        'ABS_NODE_LINK_RATIO_DIV':{},

        'ABS_LINK_DENSITY':{},
        'ABS_LINK_DENSITY_DIV':{},
        
        'ABS_MEAN_CLUST':{},
        'ABS_MEAN_CLUST_DIV':{},

        'ABS_MAX_BTW':{},
        'ABS_MAX_BTW_DIV':{},

        'ABS_MEAN_BTW':{},
        'ABS_MEAN_BTW_DIV':{},

        'ABS_BTW_ENTROPY':{},
        'ABS_BTW_ENTROPY_DIV':{},

        'ABS_BTW_ENTROPY_NORM':{},
        'ABS_BTW_ENTROPY_NORM_DIV':{},

        'ABS_DFT_ENTROPY':{},
        'ABS_DFT_ENTROPY_DIV':{},

        'ABS_CC_ENTROPY':{},
        'ABS_CC_ENTROPY_DIV':{},

        'ABS_CC_ENTROPY_NORM':{},

        'ABS_DEGREE_ASSORT':{},
        'ABS_DEGREE_ASSORT_DIV':{},

        # 'ABS_MODEL_MAX_DEGREE':{},
        # 'ABS_MODEL_DEGREE_ENTROPY':{},
        # 'ABS_MODEL_LINK_DENSITY':{},

        # 'PN_SILENT_TRANS':{},
        # 'PN_TRANS':{},
        # 'PN_PLACES':{},
        # 'PN_ARCS_MEAN':{},
        # 'PN_ARCS_MAX':{},

    }

    for fol in folders:
            my_path = base_path + fol
            my_files = [f for f in listdir(my_path) \
                        if isfile(join(my_path, f))]

            for fil in my_files:
                Gm_log = get_markov_log(fil, k_markov)
                log = xes_importer.apply(my_path + fil)
                
                for alg in algs:

                    if DEBUG:
                        print('### Algorithm: ' + str(alg))
                        print('### Folder: ' + str(fol))
                        print('### Log: ' + str(fil))

                    record = (remove_suffix(fil),alg)

                    if len(df_ref[
                    (df_ref[['EVENT_LOG','DISCOVERY_ALG']].values == \
                        record).all(axis=1)]) == 0:
                        print('Skipping record (not in df_markov_k1)...')
                        continue

                    Gm_model = get_markov_model(fil, alg, k_markov)

                    if Gm_model:
                        ret_dict = calcMetricsTimeout(Gm_model, 
                                                      Gm_log,
                                                      timeout)
                        if not ret_dict:
                            print('Skipping record (Timeout!)...')
                            continue

                        count_rec += 1

                        pn_path = base_path_pn + alg + '/' + \
                            remove_suffix(fil) + '.pnml'
                        net, im, fm = pnml_importer.apply(pn_path)

                        feat['EVENT_LOG'][count_rec] = fil
                        feat['DISCOVERY_ALG'][count_rec] = alg

                        feat['ABS_MEAN_DEGREE'][count_rec] = \
                            ret_dict['ABS_MEAN_DEGREE']
                        feat['ABS_MEAN_DEGREE_DIV'][count_rec] = \
                            ret_dict['ABS_MEAN_DEGREE_DIV']
                        
                        feat['ABS_NODE_LINK_RATIO'][count_rec] = \
                            ret_dict['ABS_NODE_LINK_RATIO']
                        feat['ABS_NODE_LINK_RATIO_DIV'][count_rec] = \
                            ret_dict['ABS_NODE_LINK_RATIO_DIV']

                        feat['ABS_LINK_DENSITY'][count_rec] = \
                            ret_dict['ABS_LINK_DENSITY']
                        feat['ABS_LINK_DENSITY_DIV'][count_rec] = \
                            ret_dict['ABS_LINK_DENSITY_DIV']
                        
                        feat['ABS_MEAN_CLUST'][count_rec] = \
                            ret_dict['ABS_MEAN_CLUST']
                        feat['ABS_MEAN_CLUST_DIV'][count_rec] = \
                            ret_dict['ABS_MEAN_CLUST_DIV']
                        
                        feat['ABS_MAX_BTW'][count_rec] = \
                            ret_dict['ABS_MAX_BTW']
                        feat['ABS_MAX_BTW_DIV'][count_rec] = \
                            ret_dict['ABS_MAX_BTW_DIV']

                        feat['ABS_MEAN_BTW'][count_rec] = \
                            ret_dict['ABS_MEAN_BTW']
                        feat['ABS_MEAN_BTW_DIV'][count_rec] = \
                            ret_dict['ABS_MEAN_BTW_DIV']

                        feat['ABS_NODE_LINK_RATIO'][count_rec] = \
                            ret_dict['ABS_NODE_LINK_RATIO']
                        feat['ABS_NODE_LINK_RATIO_DIV'][count_rec] = \
                            ret_dict['ABS_NODE_LINK_RATIO_DIV']

                        feat['ABS_BTW_ENTROPY'][count_rec] = \
                            ret_dict['ABS_BTW_ENTROPY']
                        feat['ABS_BTW_ENTROPY_DIV'][count_rec] = \
                            ret_dict['ABS_BTW_ENTROPY_DIV']

                        feat['ABS_BTW_ENTROPY_NORM'][count_rec] = \
                            ret_dict['ABS_BTW_ENTROPY_NORM']
                        feat['ABS_BTW_ENTROPY_NORM_DIV'][count_rec] = \
                            ret_dict['ABS_BTW_ENTROPY_DIV']

                        feat['ABS_DFT_ENTROPY'][count_rec] = \
                            ret_dict['ABS_DFT_ENTROPY']
                        feat['ABS_DFT_ENTROPY_DIV'][count_rec] = \
                            ret_dict['ABS_DFT_ENTROPY_DIV']
                            
                        feat['ABS_CC_ENTROPY'][count_rec] = \
                            ret_dict['ABS_CC_ENTROPY']
                        feat['ABS_CC_ENTROPY_DIV'][count_rec] = \
                            ret_dict['ABS_CC_ENTROPY_DIV']

                        feat['ABS_CC_ENTROPY_NORM'][count_rec] = \
                            ret_dict['ABS_CC_ENTROPY_NORM']

                        feat['ABS_DEGREE_ASSORT'][count_rec] = \
                            ret_dict['ABS_DEGREE_ASSORT']
                        feat['ABS_DEGREE_ASSORT_DIV'][count_rec] = \
                            ret_dict['ABS_DEGREE_ASSORT_DIV']

                        # feat['ALIGNMENTS_MARKOV'][count_rec] = \
                        #     fitness_feat.alignments_markov_log(Gm_model, 
                        #                                        log, 
                        #                                        k_markov)

                        # feat['ABS_EDGES_ONLY_IN_LOG'][count_rec] = \
                        #     fitness_feat.edges_only_log(Gm_model, Gm_log)
                        # feat['ABS_EDGES_ONLY_IN_LOG_W'][count_rec] = \
                        #     fitness_feat.edges_only_log_w(Gm_model, Gm_log)
                        # feat['ABS_EDGES_ONLY_IN_LOG_BTW'][count_rec] = \
                        #     fitness_feat.edges_only_log_btw(Gm_model, 
                        #                                     Gm_log,
                        #                                     norm=False)
                        # feat['ABS_EDGES_ONLY_IN_LOG_BTW_NORM'][count_rec] = \
                        #     fitness_feat.edges_only_log_btw(Gm_model, 
                        #                                     Gm_log,
                        #                                     norm=True)
                        # feat['ABS_EDGES_ONLY_IN_LOG_CC'][count_rec] = \
                        #     fitness_feat.edges_only_log_cc(Gm_model, 
                        #                                    Gm_log)
                        # feat['ABS_EDGES_ONLY_IN_MODEL'][count_rec] = \
                        #     precision_feat.edges_only_model(Gm_model, Gm_log)
                        # feat['ABS_EDGES_ONLY_IN_MODEL_W'][count_rec] = \
                        #     precision_feat.edges_only_model_w(Gm_model, Gm_log)

                        # feat['ABS_MAX_DEGREE'][count_rec] = \
                        #     features.out_degree_diff(Gm_model, Gm_log, 'max')
                        # feat['ABS_MAX_DEGREE_DIV'][count_rec] = \
                        #     features.out_degree_div(Gm_model, Gm_log, 'max')
                        # feat['ABS_MAX_CLUSTER'][count_rec] = \
                        #     features.clust_diff(Gm_model, Gm_log, 'max')
                        # feat['ABS_MAX_CLUSTER_DIV'][count_rec] = \
                        #     features.clust_div(Gm_model, Gm_log, 'max')
                        # feat['ABS_DEGREE_ENTROPY'][count_rec] = \
                        #     features.entropy_diff(Gm_model, 
                        #                           Gm_log, 
                        #                           normalized=False)
                        # feat['ABS_DEGREE_ENTROPY_DIV'][count_rec] = \
                        #     features.entropy_div(Gm_model, 
                        #                          Gm_log, 
                        #                          normalized=False)
                        


                        # feat['ABS_MODEL_MAX_DEGREE'][count_rec] = \
                        #     networkx_graph.calcOutDegree(Gm_model, 'max')
                        # feat['ABS_MODEL_DEGREE_ENTROPY'][count_rec] = \
                        #     networkx_graph.calcEntropyAux(Gm_model.degree(), 
                        #                                   normalized=False)
                        # feat['ABS_MODEL_LINK_DENSITY'][count_rec] = \
                        #     networkx_graph.calcLinkDensity(Gm_model)


                        # feat['PN_SILENT_TRANS'][count_rec] = \
                        #     petri_net_feat.countInvisibleTransitions(net)
                        # feat['PN_TRANS'][count_rec] = \
                        #     petri_net_feat.countTransitions(net)
                        # feat['PN_PLACES'][count_rec] = \
                        #     petri_net_feat.countPlaces(net)
                        # feat['PN_ARCS_MEAN'][count_rec] = \
                        #     petri_net_feat.countArcsPlaces(net, 'mean')
                        # feat['PN_ARCS_MAX'][count_rec] = \
                        #     petri_net_feat.countArcsPlaces(net, 'max')

                        df_from_dict(feat, out_path)
                    else:
                        print('### Skipping log: ' + fil)


    print('done!')             

                    

