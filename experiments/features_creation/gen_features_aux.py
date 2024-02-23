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

from libraries import networkx_graph


if __name__ == '__main__':
    base_path = 'xes_files/'
    base_path_pn = 'petri_nets/'
    k_markov = 3
    algs = ['IMf', 'IMd', 'ETM']
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/features_creation/feat_markov/' + \
               'extra_feat_markov_k_' + str(k_markov) + '.csv'
    count_rec = -1
    count_errors = 0
    feat = {
        'EVENT_LOG':{},
        'DISCOVERY_ALG':{},

        'ABS_EDGES_ONLY_IN_MODEL_W_2':{},
        'DIST_NODES_PERCENT':{},
        'DIST_EDGES_PERCENT':{},
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

                    Gm_model = get_markov_model(fil, alg, k_markov)

                    if Gm_model:
                        try:
                            count_rec += 1

                            pn_path = base_path_pn + alg + '/' + \
                                remove_suffix(fil) + '.pnml'
                            net, im, fm = pnml_importer.apply(pn_path)

                            feat['EVENT_LOG'][count_rec] = fil
                            feat['DISCOVERY_ALG'][count_rec] = alg

                            feat['ABS_EDGES_ONLY_IN_MODEL_W_2'][count_rec] = \
                                precision_feat.edges_only_model_w_2(Gm_model, 
                                                                    Gm_log,
                                                                    log,
                                                                    k_markov)
                            feat['DIST_NODES_PERCENT'][count_rec] = \
                                features.dist_nodes_percent(Gm_model, Gm_log)
                            feat['DIST_EDGES_PERCENT'][count_rec] = \
                                features.dist_edges_percent(Gm_model, Gm_log)


                            df_from_dict(feat, out_path)

                        except Exception as e:
                            print('#### ERROR: ' + str(e))
                            count_errors += 1
                    else:
                        print('### Skipping log: ' + fil)


    print('done!')             

                    

