from distutils.spawn import find_executable
import unittest

from features.aux_feat import activities_occurrence_2
from utils.converter.all_dist import create_all_dist_ts
from utils.converter.reach_graph_to_dfg import find_end_acts, reach_graph_to_dfg, \
                                               find_start_acts, \
                                               find_end_acts

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.transition_system.obj import TransitionSystem


class ReachGraphToDFG(unittest.TestCase):

    def test1(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')
        q4 = TransitionSystem.State(name='q4')

        q0_q1 = TransitionSystem.Transition(name='(anything1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything2, b)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q3 = TransitionSystem.Transition(name='(anything3, a)',
                                            from_state=q2,
                                            to_state=q3)
        q3_q4 = TransitionSystem.Transition(name='(anything4, a)',
                                            from_state=q3,
                                            to_state=q4)
       
        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q3)
        q3.outgoing.add(q3_q4)

        q1.incoming.add(q0_q1)
        q2.incoming.add(q1_q2)
        q3.incoming.add(q2_q3)
        q4.incoming.add(q3_q4)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)
        ts.states.add(q3)
        ts.states.add(q4)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q2_q3)
        ts.transitions.add(q3_q4)

        suffix = '!#@#!'
        create_all_dist_ts(ts, suffix)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = reach_graph_to_dfg(ts)
        start_acts = find_start_acts(ts)
        end_acts = find_end_acts(ts)

        print()
        print()
        print('####### start_acts #######: ' + str(start_acts))
        print()
        print()

        print()
        print()
        print('####### end_acts #######: ' + str(end_acts))
        print()
        print()


    def test2(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        suffix = '!#@#!'
        create_all_dist_ts(ts, suffix)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)
        
        G = reach_graph_to_dfg(ts)
        start_acts = find_start_acts(ts)
        end_acts = find_end_acts(ts)

        print()
        print()
        print('####### start_acts #######: ' + str(start_acts))
        print()
        print()

        print()
        print()
        print('####### end_acts #######: ' + str(end_acts))
        print()
        print()


    def test3(self):
        my_file = 'petri_nets/IMf/' + \
                  'QUARTA_VARA_CRIMINAL_-_COMARCA_DE_VARZEA'+\
                  '_GRANDE_-_SDCR_-_TJMT.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        gviz = ts_visualizer.apply(ts)
        ts_visualizer.view(gviz)

        G = reach_graph_to_dfg(ts)
        start_acts = find_start_acts(ts)
        end_acts = find_end_acts(ts)

        print()
        print()
        print('####### start_acts #######: ' + str(start_acts))
        print()
        print()

        print()
        print()
        print('####### end_acts #######: ' + str(end_acts))
        print()
        print()

    
if __name__ == '__main__':
    unittest.main()
