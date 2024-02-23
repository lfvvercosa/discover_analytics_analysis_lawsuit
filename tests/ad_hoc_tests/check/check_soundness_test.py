from pm4py.objects.petri_net.utils.check_soundness \
    import check_easy_soundness_net_in_fin_marking

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
import time
                

# log_path = 'xes_files/1/activitylog_uci_detailed_labour.xes.gz'
pn_path = 'petri_nets/tests/'+\
              'activitylog_uci_detailed_labour.xes.gz.pnml'

# log = xes_importer.apply(log_path)
net, im, fm = pnml_importer.apply(pn_path)

print(check_easy_soundness_net_in_fin_marking(net, im, fm))

