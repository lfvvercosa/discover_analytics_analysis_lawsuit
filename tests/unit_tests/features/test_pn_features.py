from pm4py.objects.petri_net.importer import importer as pnml_importer
from features import petri_net_feat


my_file = 'petri_nets/tests/pn_parallel.pnml'
net, im, fm = pnml_importer.apply(my_file)

print('################################')
print('###### TEST "pn_parallel" ######')
print('################################')
print()
print('#countInvisibleTransitions: ' + str(petri_net_feat.countInvisibleTransitions(net)))
print('#percentInvisTran: ' + str(petri_net_feat.percentInvisTran(net)))
print('#countTransitions: ' + str(petri_net_feat.countTransitions(net)))
print('#countPlaces: ' + str(petri_net_feat.countPlaces(net)))
print('#countArcsPlaces: ' + str(petri_net_feat.countArcsPlaces(net)))
print('#countArcsPlacesMax: ' + str(petri_net_feat.countArcsPlaces(net, stat='max')))
print('#countInArcsPlaces: ' + str(petri_net_feat.countInArcsPlaces(net)))
print('#countOutArcsPlaces: ' + str(petri_net_feat.countOutArcsPlaces(net)))
print('#countInArcsTran: ' + str(petri_net_feat.countInArcsTran(net)))
print('#countOutArcsTran: ' + str(petri_net_feat.countOutArcsTran(net)))
print('#countInArcsInvTran: ' + str(petri_net_feat.countInArcsInvTran(net)))
print('#countOutArcsInvTran: ' + str(petri_net_feat.countOutArcsInvTran(net)))
print()

my_file = 'petri_nets/tests/pn_parallel3.pnml'
net, im, fm = pnml_importer.apply(my_file)

print('#################################')
print('###### TEST "pn_parallel3" ######')
print('#################################')
print()
print('#countInvisibleTransitions: ' + str(petri_net_feat.countInvisibleTransitions(net)))
print('#percentInvisTran: ' + str(petri_net_feat.percentInvisTran(net)))
print('#countTransitions: ' + str(petri_net_feat.countTransitions(net)))
print('#countPlaces: ' + str(petri_net_feat.countPlaces(net)))
print('#countArcsPlaces: ' + str(petri_net_feat.countArcsPlaces(net)))
print('#countArcsPlacesMax: ' + str(petri_net_feat.countArcsPlaces(net, stat='max')))
print('#countInArcsPlaces: ' + str(petri_net_feat.countInArcsPlaces(net)))
print('#countOutArcsPlaces: ' + str(petri_net_feat.countOutArcsPlaces(net)))
print('#countInArcsTran: ' + str(petri_net_feat.countInArcsTran(net)))
print('#countOutArcsTran: ' + str(petri_net_feat.countOutArcsTran(net)))
print('#countInArcsInvTran: ' + str(petri_net_feat.countInArcsInvTran(net)))
print('#countOutArcsInvTran: ' + str(petri_net_feat.countOutArcsInvTran(net)))
print()
