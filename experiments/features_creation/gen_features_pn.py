from os import listdir
from os.path import isfile, join, exists

from features import petri_net_feat
from utils.read_and_write.files_handle import remove_suffix
from utils.creation.create_df_from_dict import df_from_dict
from utils.global_var import DEBUG

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer


if __name__ == '__main__':
    base_path = 'xes_files/'
    base_path_pn = 'petri_nets/'
    algs = ['IMf', 'IMd', 'ETM']
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/features_creation/feat_pn/' + \
               'feat_pn.csv'
    count_rec = -1
    feat = {
        'EVENT_LOG':{},
        'DISCOVERY_ALG':{},

        'PN_SILENT_TRANS':{},
        'PN_PERC_SILENT_TRANS':{},
        'PN_TRANS':{},
        'PN_PLACES':{},
        'PN_ARCS_MEAN':{},
        'PN_ARCS_MAX':{},
        'PN_IN_ARCS_MEAN':{},
        'PN_OUT_ARCS_MEAN':{},
        'PN_IN_ARCS_TRAN_MEAN':{},
        'PN_OUT_ARCS_TRAN_MEAN':{},
        'PN_IN_ARCS_INV_TRAN_MEAN':{},
        'PN_OUT_ARCS_INV_TRAN_MEAN':{},
        'PN_IN_ARCS_MAX':{},
        'PN_OUT_ARCS_MAX':{},
        'PN_IN_ARCS_TRAN_MAX':{},
        'PN_OUT_ARCS_TRAN_MAX':{},
        'PN_IN_ARCS_INV_TRAN_MAX':{},
        'PN_OUT_ARCS_INV_TRAN_MAX':{},

    }

    for fol in folders:
            my_path = base_path + fol
            my_files = [f for f in listdir(my_path) \
                        if isfile(join(my_path, f))]

            for fil in my_files:
                for alg in algs:

                    if DEBUG:
                        print('### Algorithm: ' + str(alg))
                        print('### Folder: ' + str(fol))
                        print('### Log: ' + str(fil))

                    pn_path = base_path_pn + alg + '/' + \
                        remove_suffix(fil) + '.pnml'
                    
                    if exists(pn_path):
                        count_rec += 1
                        net, im, fm = pnml_importer.apply(pn_path)

                        feat['EVENT_LOG'][count_rec] = fil
                        feat['DISCOVERY_ALG'][count_rec] = alg

                        feat['PN_SILENT_TRANS'][count_rec] = \
                            petri_net_feat.countInvisibleTransitions(net)
                        feat['PN_PERC_SILENT_TRANS'][count_rec] = \
                            petri_net_feat.percentInvisTran(net)
                        feat['PN_TRANS'][count_rec] = \
                            petri_net_feat.countTransitions(net)
                        feat['PN_PLACES'][count_rec] = \
                            petri_net_feat.countPlaces(net)
                        feat['PN_ARCS_MEAN'][count_rec] = \
                            petri_net_feat.countArcsPlaces(net, 'mean')
                        feat['PN_ARCS_MAX'][count_rec] = \
                            petri_net_feat.countArcsPlaces(net, 'max')
                        feat['PN_IN_ARCS_MEAN'][count_rec] = \
                            petri_net_feat.countInArcsPlaces(net, 'mean')
                        feat['PN_OUT_ARCS_MEAN'][count_rec] = \
                            petri_net_feat.countOutArcsPlaces(net, 'mean')
                        feat['PN_IN_ARCS_TRAN_MEAN'][count_rec] = \
                            petri_net_feat.countInArcsTran(net, 'mean')
                        feat['PN_OUT_ARCS_TRAN_MEAN'][count_rec] = \
                            petri_net_feat.countOutArcsTran(net, 'mean')
                        feat['PN_IN_ARCS_INV_TRAN_MEAN'][count_rec] = \
                            petri_net_feat.countInArcsInvTran(net, 'mean')
                        feat['PN_OUT_ARCS_INV_TRAN_MEAN'][count_rec] = \
                            petri_net_feat.countOutArcsInvTran(net, 'mean')
                        feat['PN_IN_ARCS_MAX'][count_rec] = \
                            petri_net_feat.countInArcsPlaces(net, 'max')
                        feat['PN_OUT_ARCS_MAX'][count_rec] = \
                            petri_net_feat.countOutArcsPlaces(net, 'max')
                        feat['PN_IN_ARCS_TRAN_MAX'][count_rec] = \
                            petri_net_feat.countInArcsTran(net, 'max')
                        feat['PN_OUT_ARCS_TRAN_MAX'][count_rec] = \
                            petri_net_feat.countOutArcsTran(net, 'max')
                        feat['PN_IN_ARCS_INV_TRAN_MAX'][count_rec] = \
                            petri_net_feat.countInArcsInvTran(net, 'max')
                        feat['PN_OUT_ARCS_INV_TRAN_MAX'][count_rec] = \
                            petri_net_feat.countOutArcsInvTran(net, 'max')

                        df_from_dict(feat, out_path)
                    else:
                        print('### PN doesnt exist. Skipping log: ' + fil)


    print('done!')             

                    

