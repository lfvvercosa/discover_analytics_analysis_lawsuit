### create new features for CNJ dataset ###
import os
from os import listdir
from os.path import isfile, join, exists
import pandas as pd
import networkx as nx
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants.im_d.dfg_based import Parameters
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.petri_net import visualizer as pt_visualizer
from pm4py.algo.discovery.inductive.variants.im_f.algorithm import \
    Parameters as Parameters_IMf
from experiments.models.discover_dep_graph import discover_dep_graph
from experiments.features_creation.get_filtered_log_graph import \
    get_filtered_weighted_graph
from experiments.log.discover_dfg import discover_dfg
from experiments.models.get_markov import get_markov_model
from utils.converter.pn_to_dfg import petri_net_to_dfg, \
    petri_net_to_dfg_with_timeout
from utils.converter.pn_to_dep_graph import petri_net_to_dep_graph
from utils.converter.markov.create_markov_log import create_mk_abstraction_log
from features import features
from features import aux_feat
from features import fitness_feat
from features import precision_feat
from features import escap_edge_dfg
from libraries import pm4py_metrics, petri_net_metrics, networkx_graph


def remove_suffix_if_needed(f, alg):
    if alg == 'ETM':
        if '.xes.gz' in f:
            f = f.replace('.xes.gz', '')
        if '.xes' in f:
            f = f.replace('.xes', '')
        
    return f


def existing_record(f, alg):
    if exists(output_path):
        df = pd.read_csv(output_path, sep='\t')
        df_temp = df[(df['EVENT_LOG'] == f) & \
                     (df['DISCOVERY_ALG'] == alg)]
        if not df_temp.empty:
            return True
    
    return False


def in_blacklist(f, alg):
    if (f, alg) in black_list:
        return True
    
    return False


def load_records(my_dict):
    if exists(output_path):
        df = pd.read_csv(output_path, sep='\t')
        my_dict = df.to_dict()

        k = list(my_dict.keys())[0]
        count = len(my_dict[k])

        return (my_dict, count)

    return (my_dict, 0)



black_list = [
    ('JUIZO_DA_1a_ESCRIVANIA_CRIMINAL_DE_ARAPOEMA_-_TJTO.xes', 'IMf'),
    ('5a_VARA_CIVEL_-_TJMT.xes', 'IMf'),
    ('VITORIA_-_5a_VARA_CIVEL_-_TJES.xes', 'IMd'),
    ('Production_Data.xes.gz', 'IMd'),
    ('3a_VARA_DE_EXECUCOES_FISCAIS_DA_COMARCA_DE_FORTALEZA_-_TJCE.xes', 'IMd'),
    ('4a_VARA_DE_SUCESSOES_E_REGISTROS_PUBLICOS_DA_CAPITAL_-_TJPE.xes', 'IMd'),
    ('edited_hh104_labour.xes.gz', 'IMd'),
    ('PRIMEIRA_VARA_CRIMINAL_-_COMARCA_DE_VARZEA_GRANDE_-_SDCR_-_TJMT.xes', 'IMd'),
    ('QUARTA_VARA_CRIMINAL_-_COMARCA_DE_VARZEA_GRANDE_-_SDCR_-_TJMT.xes', 'IMd'),
    ('1a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.xes', 'IMd'),
    ('1a_VARA_CRIMINAL_DA_CAPITAL_-_TJAM.xes', 'IMd'),
    ('VARA_DE_SUCESSOES_DE_CAMPINA_GRANDE_-_TJPB.xes', 'IMd'),
    ('COLOMBO_-_VARA_DA_FAZENDA_PUBLICA_-_TJPR.xes', 'IMd'),
    ('2A._VARA_DE_EXECUTIVO_FISCAL_JOAO_PESSOA_-_TJPB.xes', 'IMd'),
    ('JUIZO_DA_1a_ESCRIVANIA_CRIMINAL_DE_ARAPOEMA_-_TJTO.xes', 'IMd'),
    ('5a_VARA_CIVEL_-_TJMT.xes', 'IMd'),
    
]
base_path = 'xes_files/'
folders = ['1/', '2/', '3/', '4/', '5/']
petri_nets_path = [
    ('petri_nets/IMf', 'IMf'),
    ('petri_nets/Heu_Miner/', 'HEU_MINER'),
    ('petri_nets/IMd/', 'IMd'),
    ('petri_nets/ETM/', 'ETM'),
]
output_path = 'experiments/results/markov/features_markov_k2.csv'
k_markov = 2

features_dict = {
    'EVENT_LOG':{},
    'EDIT_DISTANCE':{},
    'EDIT_DISTANCE_WEIGHTED':{},
    'EDIT_DISTANCE_WEIGHTED_MODIF':{},
    'EDIT_DISTANCE_WEIGHTED_FIT':{},
    'ALIGNMENT_DFG':{},
    'DIST_NODES_PERCENT':{},
    'DIST_EDGES_PERCENT':{},
    'DIST_EDGES_WEIGHTED_FIT':{},
    'DIST_EDGES_BTW':{},
    'DIST_EDGES_CC':{},
    'DIST_EDGES_DEGREE':{},
    'CLUST_DIFF':{},
    'CLUST_MAX_DIFF':{},
    'DEGREE_DIFF':{},
    'DEGREE_MAX_DIFF':{},
    'BETWEEN_DIFF':{},
    'BETWEEN_MAX_DIFF':{},
    'ASSORT_DIFF':{},
    'ENTROPY_DIFF':{},
    'ENTROPY_DIFF_BTW':{},
    'EXCEEDING_EDGES':{},
    'EXCEEDING_EDGES_BTW':{},
    'EXCEEDING_EDGES_CC':{},
    'EXCEEDING_EDGES_DEGREE':{},
    # 'ESCAPING_EDGES_DFG':{},
    # 'ESCAPING_EDGES_DFG_NO_ALGN':{},
    'FOURIER_ENTROPY_DIFF':{},
    'FOURIER_ENTROPY_DIFF_WEIGHTED':{},
    'CLUST_DIFF_WEIGHTED':{},
    'BETWEEN_DIFF_WEIGHTED':{},
    'BETWEEN_DIFF_WEIGHTED_NORM':{},
    'PN_SILENT_TRANS':{},
    'PN_TRANS':{},
    'PN_PLACES':{},
    'PN_ARCS_MEAN':{},
    'PN_ARCS_MAX':{},
    'PN_SIMPLICITY':{},
    'PN_GENERALIZATION':{},
    'LOG_BTW_MAX':{},
    'LOG_BTW_MEDIAN':{},
    'LOG_BTW_ENTROPY':{},
    'LOG_CC_MAX':{},
    'LOG_CC_MEDIAN':{},
    'LOG_CC_ENTROPY':{},
    'LOG_DEGREE_MAX':{},
    'LOG_DEGREE_MEDIAN':{},
    'LOG_DEGREE_ENTROPY':{},
    'LOG_LINK_DENSITY':{},
    'LOG_NODE_DEGREE_RATIO':{},
    'LOG_FOURIER_ENTROPY':{},
    'LOG_TRACES':{},
    'LOG_VARIANTS':{},
    'LOG_ACTIVITIES':{},
    'DISCOVERY_ALG':{},
}

features_dict, count_rec = load_records(features_dict)

for alg in petri_nets_path:
    for fol in folders:
        my_path = base_path + fol
        files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

        for f in files:

            if existing_record(f, alg[1]):
                print('### Existing record. Skipping...')
                
                continue
            
            if in_blacklist(f, alg[1]):
                print('### In Blacklist. Skipping...')
                
                continue

            if f != 'edited_hh104_labour.xes.gz' or alg[1] != 'IMf':
                continue

            try:
                net, im, fm = \
                    pnml_importer.apply(os.path.join(alg[0],
                                        remove_suffix_if_needed(f,alg[1]) + \
                                        '.pnml')
                                    )
            except Exception as e:
                print(e)
                continue
            
            print('### current folder: ' + str(fol))
            print('### current algorithm: ' + str(alg[1]))
            print('### current file: ' + str(f))

            G_model = get_markov_model(f=f, alg=alg[1], k=k_markov)

            if not G_model:
                print('Not able to obtain graph from model, skipping this one!')
                continue
            elif len(G_model.edges) == 0:
                print('Empty model! Skipping...')
                continue

            print('### number nodes G_model: ' + str(len(G_model.nodes)))
            print('### number edges G_model: ' + str(len(G_model.edges)))

            file_path = my_path + str(f)
            log = xes_importer.apply(file_path)

            # G_model = petri_net_to_dep_graph(net, im, fm)
            
            # timeout = 60
            # G_model = petri_net_to_dfg_with_timeout(net, im, fm, timeout)
            # G_model = petri_net_to_dfg(net, im, fm)

            # dep_thr = 0.85
            # G_log = discover_dep_graph(log, dep_thr)
            # G_log = discover_dfg(log)

            G_log = create_mk_abstraction_log(log=log, k=k_markov)
            
            if not G_log:
                print('Not able to obtain graph from log, skipping this one!')
                continue
            elif len(G_log.edges) == 0:
                print('Empty graph from log! Skipping...')
                continue

            print('Able to obtain graphs, proceeding...')

            # act_occur = aux_feat.activities_occurrence(log)
            act_occur = aux_feat.activities_occurrence_model_based(G_log)

            # create features
            features_dict['EVENT_LOG'][count_rec] = f
            features_dict['EDIT_DISTANCE'][count_rec] = -1
            features_dict['EDIT_DISTANCE_WEIGHTED'][count_rec] = -1
            features_dict['EDIT_DISTANCE_WEIGHTED_MODIF'][count_rec] = -1
            features_dict['EDIT_DISTANCE_WEIGHTED_FIT'][count_rec] = -1
            # features_dict['ALIGNMENT_DFG'][count_rec] = (
            #      fitness_feat.alignments_dfg_log(G_model, log))  
            features_dict['ALIGNMENT_DFG'][count_rec] = (
                 fitness_feat.alignments_markov_log(G_model, log, k_markov))  
            features_dict['DIST_NODES_PERCENT'][count_rec] = \
                features.dist_nodes_percent(G_model, G_log)
            features_dict['DIST_EDGES_PERCENT'][count_rec] = \
                features.dist_edges_percent(G_model, G_log)
            features_dict['DIST_EDGES_WEIGHTED_FIT'][count_rec] = \
                fitness_feat.edges_only_log_w(G_model, G_log)
            features_dict['DIST_EDGES_BTW'][count_rec] = \
                fitness_feat.\
                    dist_edges_double_weighted(G_model, 
                                            G_log, 
                                            nx.betweenness_centrality)
            features_dict['DIST_EDGES_CC'][count_rec] = \
                fitness_feat.\
                    dist_edges_double_weighted(G_model, 
                                            G_log, 
                                            nx.clustering)
            features_dict['DIST_EDGES_DEGREE'][count_rec] = \
                fitness_feat.\
                    dist_edges_double_weighted(G_model, 
                                            G_log, 
                                            aux_feat.func_degree)
            features_dict['CLUST_DIFF'][count_rec] = \
                features.clust_diff(G_model, G_log)
            features_dict['CLUST_MAX_DIFF'][count_rec] = \
                features.clust_diff(G_model, G_log, stat='max')
            features_dict['DEGREE_DIFF'][count_rec] = \
                features.out_degree_diff(G_model, G_log)
            features_dict['DEGREE_MAX_DIFF'][count_rec] = \
                features.out_degree_diff(G_model, G_log, stat='max')
            features_dict['BETWEEN_DIFF'][count_rec] = -1
            features_dict['BETWEEN_MAX_DIFF'][count_rec] = -1
            features_dict['ASSORT_DIFF'][count_rec] = \
                features.assortativity_diff(G_model, G_log)
            features_dict['ENTROPY_DIFF'][count_rec] = \
                features.entropy_diff(G_model, G_log)
            features_dict['ENTROPY_DIFF_BTW'][count_rec] = -1
            features_dict['EXCEEDING_EDGES'][count_rec] = \
                precision_feat.edges_only_model_w(G_model, G_log)
            features_dict['EXCEEDING_EDGES_BTW'][count_rec] = precision_feat.\
                    exceding_edges_model_weighted(G_model, 
                                                G_log, 
                                                act_occur,
                                                nx.betweenness_centrality)
            features_dict['EXCEEDING_EDGES_CC'][count_rec] = precision_feat.\
                    exceding_edges_model_weighted(G_model, 
                                                G_log, 
                                                act_occur,
                                                nx.clustering)
            features_dict['EXCEEDING_EDGES_DEGREE'][count_rec] = precision_feat.\
                    exceding_edges_model_weighted(G_model, 
                                                G_log, 
                                                act_occur,
                                                aux_feat.func_degree)
            # features_dict['ESCAPING_EDGES_DFG'][count_rec] = escap_edge_dfg.\
            #      calc_precision(log, 
            #                  G_model, 
            #                  align=True)
            # features_dict['ESCAPING_EDGES_DFG_NO_ALGN'][count_rec] = escap_edge_dfg.\
            #      calc_precision(log, 
            #                  G_model, 
            #                  align=False)

            features_dict['FOURIER_ENTROPY_DIFF'][count_rec] = \
                features.fourier_entropy_diff(G_model, G_log)
            features_dict['FOURIER_ENTROPY_DIFF_WEIGHTED'][count_rec] = \
                features.fourier_entropy_diff_weighted(G_model, G_log)
            features_dict['CLUST_DIFF_WEIGHTED'][count_rec] = \
                features.clust_diff_weighted(G_model, G_log)
            features_dict['BETWEEN_DIFF_WEIGHTED'][count_rec] = -1
            features_dict['BETWEEN_DIFF_WEIGHTED_NORM'][count_rec] = -1

            features_dict['PN_SILENT_TRANS'][count_rec] = \
                petri_net_metrics.countInvisibleTransitions(net)
            features_dict['PN_TRANS'][count_rec] = \
                petri_net_metrics.countTransitions(net)
            features_dict['PN_PLACES'][count_rec] = \
                petri_net_metrics.countPlaces(net)
            features_dict['PN_ARCS_MEAN'][count_rec] = \
                petri_net_metrics.countArcs(net, stat='mean')
            features_dict['PN_ARCS_MAX'][count_rec] = \
                petri_net_metrics.countArcs(net, stat='max')
            features_dict['PN_SIMPLICITY'][count_rec] = \
                petri_net_metrics.calcSimplicity(log, net, im, fm)
            features_dict['PN_GENERALIZATION'][count_rec] = -1

            (G_log_filt, filt_log) = get_filtered_weighted_graph(f, log, alg[1])

            features_dict['LOG_BTW_MAX'][count_rec] = \
                networkx_graph.calcBetweenness(G_log_filt, 
                                               stat='max', 
                                               weight='weight')    
            features_dict['LOG_BTW_MEDIAN'][count_rec] = \
                networkx_graph.calcBetweenness(G_log_filt, 
                                               stat='median', 
                                               weight='weight')
            features_dict['LOG_BTW_ENTROPY'][count_rec] = \
                networkx_graph.calcBetweennessEntropy(G_log_filt, 
                                               weight='weight')  
            features_dict['LOG_CC_MAX'][count_rec] = \
                networkx_graph.calcClustCoef(G_log_filt, 
                                             stat='max', 
                                             weight='weight')  
            features_dict['LOG_CC_MEDIAN'][count_rec] = \
                networkx_graph.calcClustCoef(G_log_filt, 
                                             stat='median', 
                                             weight='weight')  
            features_dict['LOG_CC_ENTROPY'][count_rec] = \
                networkx_graph.calcEntropyClustering(G_log_filt, 
                                                     weight='weight',
                                                     normalized=True,
                                                    )                                            
            features_dict['LOG_DEGREE_MAX'][count_rec] = \
                networkx_graph.calcOutDegree(G_log_filt, 
                                             stat='max', 
                                             weight='weight')  
            features_dict['LOG_DEGREE_MEDIAN'][count_rec] = \
                networkx_graph.calcOutDegree(G_log_filt, 
                                             stat='median', 
                                             weight='weight')  
            features_dict['LOG_DEGREE_ENTROPY'][count_rec] = \
                networkx_graph.calcEntropyAux(G_log_filt.degree(weight='weight'), 
                                              normalized=True)
            features_dict['LOG_LINK_DENSITY'][count_rec] = \
                networkx_graph.calcLinkDensity(G_log_filt)
            features_dict['LOG_NODE_DEGREE_RATIO'][count_rec] = \
                networkx_graph.calcNodeLinkRatio(G_log_filt)
            features_dict['LOG_FOURIER_ENTROPY'][count_rec] = \
                networkx_graph.calcEntropyDFTEigenvalues(G_log_filt, 
                                                         weight='weight')
            features_dict['LOG_TRACES'][count_rec] = \
                pm4py_metrics.number_of_traces(log)
            features_dict['LOG_VARIANTS'][count_rec] = \
                pm4py_metrics.number_of_variants(log)
            features_dict['LOG_ACTIVITIES'][count_rec] = \
                pm4py_metrics.number_of_activities(log)    
            features_dict['DISCOVERY_ALG'][count_rec] = alg[1]

            # save to csv file
            df = pd.DataFrame.from_dict(features_dict)
            df.to_csv(path_or_buf=output_path,
                    sep='\t',
                    header=True,
                    index=False)

            print('done!')
            count_rec += 1