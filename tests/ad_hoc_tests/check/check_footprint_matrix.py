from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery


pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
log_path = 'xes_files/test/test_feat_markov_ks_diff5.xes'

net, im, fm = pnml_importer.apply(pn_path)
log = xes_importer.apply(log_path)

fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)

print('#### Footprint Log')
print(fp_log)
print()

fp_net = footprints_discovery.apply(net, im, fm)

print('#### Footprint Model')
print(fp_net)
print()



