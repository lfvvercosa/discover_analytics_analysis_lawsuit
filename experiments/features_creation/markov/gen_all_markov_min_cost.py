import pandas as pd
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.petri_net import variants


path = 'experiments/features_creation/feat_markov/feat_markov_k_all.csv'
out_path = 'experiments/features_creation/feat_markov/feat_markov_k_all_min.csv'
algs = ['IMf','IMd','ETM']
variant_a_star = variants.state_equation_a_star



df = pd.read_csv(path, sep='\t')
min_cost = {'EVENT_LOG':[], 'DISCOVERY_ALG':[], 'MIN_COST':[]}


def remove_suffix_name(name):
    name = name.replace('.xes.gz','')
    name = name.replace('.xes','')

    return name


for a in algs:
    filename = df[df['DISCOVERY_ALG'] == a]['EVENT_LOG'].to_list()

    for f in filename:
        print(f)
        print(a)
        print()

        name = remove_suffix_name(f)
        pn_path = "models/petri_nets/" + a + "/" + str(name) + ".pnml"
        net, im, fm = pnml_importer.apply(pn_path)
        parameters = {}
        best_worst = variant_a_star.get_best_worst_cost(net, im, fm, parameters)

        min_cost['EVENT_LOG'].append(f)
        min_cost['DISCOVERY_ALG'].append(a)
        min_cost['MIN_COST'].append(best_worst)

df_cost = pd.DataFrame.from_dict(min_cost)

df = df.merge(df_cost, on=['EVENT_LOG','DISCOVERY_ALG'],how='inner')
df.to_csv(out_path, sep='\t')

print('done!')



