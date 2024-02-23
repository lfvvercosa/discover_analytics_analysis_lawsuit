import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator


my_file = "models/petri_nets/tests/variants/p1_var1.pnml"
net, im, fm = pnml_importer.apply(my_file)

simulated_log = simulator.apply(net, 
                im, 
                variant=simulator.Variants.BASIC_PLAYOUT, 
                parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 50})

df = pm4py.convert_to_dataframe(simulated_log)



print()