from genericpath import exists
from os import listdir
from os.path import isfile, join
from experiments.models.get_markov import get_markov_log, \
                                          get_markov_model
import pandas as pd

from features import baseline_feat

from utils.read_and_write.files_handle import remove_suffix
from utils.creation.create_df_from_dict import df_from_dict
from utils.global_var import DEBUG

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer


if __name__ == '__main__':
    base_path = 'xes_files/'
    base_path_pn = 'petri_nets/'
    k_markov = 2
    algs = ['IMf', 'IMd', 'ETM']
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/features_creation/feat_baseline/' + \
               'feat_baseline.csv'
    count_rec = -1
    count_errors = 0
    feat = {
        'EVENT_LOG':{},
        'DISCOVERY_ALG':{},

        'FOOTPRINT_COST_FIT_W':{},
        'FOOTPRINT_COST_FIT':{},
        'FOOTPRINT_COST_PRE_W':{},
        'FOOTPRINT_COST_PRE':{},
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

                    pn_path = base_path_pn + alg + '/' + \
                        remove_suffix(fil) + '.pnml'

                    if exists(pn_path):
                        try:
                            count_rec += 1
                            net, im, fm = pnml_importer.apply(pn_path)

                            feat['EVENT_LOG'][count_rec] = fil
                            feat['DISCOVERY_ALG'][count_rec] = alg

                            feat['FOOTPRINT_COST_FIT_W'][count_rec] = \
                                baseline_feat.footprint_cost_fit_w(log,
                                                                net, 
                                                                im, 
                                                                fm) 

                            feat['FOOTPRINT_COST_FIT'][count_rec] = \
                                baseline_feat.footprint_cost_fit(log,
                                                                net, 
                                                                im, 
                                                                fm) 


                            feat['FOOTPRINT_COST_PRE_W'][count_rec] = \
                                baseline_feat.footprint_cost_pre_w(log,
                                                                net, 
                                                                im, 
                                                                fm)

                            feat['FOOTPRINT_COST_PRE'][count_rec] = \
                                baseline_feat.footprint_cost_pre(log,
                                                                net, 
                                                                im, 
                                                                fm)

                            df_from_dict(feat, out_path)
                        except Exception as e:
                            print('#### ERROR: ' + str(e))
                            count_errors += 1

                    else:
                        print('### Skipping log: ' + fil)


    print('done!') 
    print('total errors: ' + str(count_errors))            

                    

