import pandas as pd
from sklearn.cluster import KMeans
from pm4py import convert_to_dataframe
from pm4py import convert_to_event_log
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.heuristics.parameters import Parameters
from pm4py.algo.discovery.inductive.variants.im_f import algorithm as IMf
from pm4py.algo.discovery.inductive.variants.im_clean import algorithm as IMc
from pm4py.algo.conformance.alignments.petri_net import variants
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

from pm4py.visualization.petri_net import visualizer as pn_visualizer

from experiments.clustering.create_n_gram import create_1_gram


def assign_feat_weight(df, feat_weight):
    for f in feat_weight:
        df[f] = df[f] * feat_weight[f]

    
    return df


def split_clusters(df, df_map_cluster, col):
    df_clusters = {}
    df_work = df.merge(df_map_cluster, 
                      on='case:concept:name',
                      how='left')
    ids = df_map_cluster[col].tolist()
    ids = list(set(ids))
    ids.sort()

    for i in ids:
        df_temp = df_work[df_work[col] == i]
        df_clusters[i] = df_temp.drop(columns=col)

    
    return df_clusters


def get_model_cost_func(pn, log_cost_func):
    model_cost_function = dict()
    
    for t in pn.transitions:
        if t.label is not None:
            model_cost_function[t] = log_cost_func[t.label] 
        else:
            model_cost_function[t] = 0


    return model_cost_function


act_weight = {'A':2, 
              'B':1, 
              'C':4, 
              'D':1, 
              'E':1, 
              'F':1, 
              'G':1
             }


### Load event log
log_path = 'xes_files/clustering/clust_test2.xes'
log = xes_importer.apply(log_path)
df = convert_to_dataframe(log)


### Create 1-gram
df_gram = create_1_gram(df, 'case:concept:name', 'concept:name')
df_gram.index = df_gram.index.map(int)
df_gram = df_gram.sort_index()


### Assign feature weight
# df_gram = assign_feat_weight(df_gram, act_weight)

print(df_gram)


### Cluster with K-means
X = df_gram.values
kmeans = KMeans(n_clusters=2).fit(X)
print(kmeans.labels_)


### Split dataframe based on cluster id
cluster_ids = kmeans.labels_
trace_ids = df_gram.index.to_list()
trace_ids = [str(i) for i in trace_ids]
col = 'cluster_id'
map_cluster = {'case:concept:name':trace_ids, 
               col:cluster_ids}
df_map_cluster = pd.DataFrame.from_dict(map_cluster)
df_clusters = split_clusters(df, df_map_cluster, col)
log_clusters = {}

for id in df_clusters:
    log_clusters[id] = convert_to_event_log(df_clusters[id])


### Discover process model for each cluster
pn_clusters = {}
thresh = 0.8

for id in df_clusters:
    log_temp = log_clusters[id]

    params = {Parameters.DEPENDENCY_THRESH:thresh}
    net, im, fm = heuristics_miner.apply(log_temp,
                                         parameters=params)

    # params = {IMc.Parameters.NOISE_THRESHOLD:thresh}
    # net, im, fm = IMc.apply(log_temp, params)

    pn_clusters[id] = (net, im, fm)

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)


### Calculate alignment fitness for each cluster

log_cost = act_weight
model_cost = act_weight

align_variant = variants.my_dijkstra
fitness_pns = {}


for id in pn_clusters:

    (net, im, fm) = pn_clusters[id]
    log_cluster = log_clusters[id]

    params = {}
    params[replay_fitness.Parameters.ALIGN_VARIANT] = align_variant
    params[align_variant.Parameters.PARAM_MODEL_COST_FUNCTION] = \
                    get_model_cost_func(net, model_cost)
    params[align_variant.Parameters.PARAM_STD_SYNC_COST] = 0
    params[align_variant.Parameters.PARAM_TRACE_COST_FUNCTION] = log_cost

    v = replay_fitness.apply(log_cluster, net, im, fm, 
                            variant=replay_fitness.Variants.ALIGNMENT_BASED,
                            parameters=params)
    
    fitness_pns[id] = v['log_fitness']
    
    aligned_traces = alignments.apply_log(log_cluster, net, im, fm,
                            variant=align_variant,
                            parameters=params)
    
    print()
    
print('fitness clusters: ' + str(fitness_pns))

### Calculate combined alignment fitness
combined_fitness = 0
total_traces = 0

for id in fitness_pns:
    combined_fitness += fitness_pns[id] * len(df_clusters[id])
    total_traces += len(df_clusters[id])

combined_fitness /= total_traces

print('combined fitness: ' + str(combined_fitness))
