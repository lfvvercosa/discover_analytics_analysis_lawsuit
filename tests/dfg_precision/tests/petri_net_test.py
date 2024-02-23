from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


net = PetriNet("petri_net_test_precision")

source = PetriNet.Place("source")
sink = PetriNet.Place("sink")

p1 = PetriNet.Place("p1")
p2 = PetriNet.Place("p2")
p3 = PetriNet.Place("p3")
p4 = PetriNet.Place("p4")
p6 = PetriNet.Place("p6")
p7 = PetriNet.Place("p7")

# add the places to the Petri Net
net.places.add(source)
net.places.add(sink)
net.places.add(p1)
net.places.add(p2)
net.places.add(p3)
net.places.add(p4)
net.places.add(p6)
net.places.add(p7)

# create activities (transitions)
a = PetriNet.Transition("a", "a")
b = PetriNet.Transition("b", "b")
c = PetriNet.Transition("c", "c")
d = PetriNet.Transition("d", "d")
e = PetriNet.Transition("e", "e")
f = PetriNet.Transition("f", "f")
g = PetriNet.Transition("g", "g")
h = PetriNet.Transition("h", "h")
i = PetriNet.Transition("i", "i")

# add created transitions to net
net.transitions.add(a)
net.transitions.add(b)
net.transitions.add(c)
net.transitions.add(d)
net.transitions.add(e)
net.transitions.add(f)
net.transitions.add(g)
net.transitions.add(h)
net.transitions.add(i)

# add arcs
petri_utils.add_arc_from_to(source, a, net)
petri_utils.add_arc_from_to(a, p1, net)
petri_utils.add_arc_from_to(a, p2, net)
petri_utils.add_arc_from_to(p1, b, net)
petri_utils.add_arc_from_to(p1, f, net)
petri_utils.add_arc_from_to(p2, c, net)
petri_utils.add_arc_from_to(p2, f, net)
petri_utils.add_arc_from_to(b, p3, net)
petri_utils.add_arc_from_to(c, p4, net)
petri_utils.add_arc_from_to(f, p6, net)
petri_utils.add_arc_from_to(p3, d, net)
petri_utils.add_arc_from_to(p3, i, net)
petri_utils.add_arc_from_to(p3, e, net)
petri_utils.add_arc_from_to(p4, d, net)
petri_utils.add_arc_from_to(p4, e, net)
petri_utils.add_arc_from_to(p6, g, net)
petri_utils.add_arc_from_to(g, p7, net)
petri_utils.add_arc_from_to(p7, h, net)
petri_utils.add_arc_from_to(i, p1, net)
petri_utils.add_arc_from_to(d, sink, net)
petri_utils.add_arc_from_to(e, sink, net)
petri_utils.add_arc_from_to(h, sink, net)

# adding tokens
initial_marking = Marking()
initial_marking[source] = 1
final_marking = Marking()
final_marking[sink] = 1

# visualize
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)

# export
pnml_exporter.apply(net, 
                    initial_marking, 
                    "simul_qual_metr/tests/dfg_precision/petri_net_test.pnml", 
                    final_marking=final_marking)

print('done!')
