from os import listdir
from os.path import isfile, join, exists
import unittest
import matplotlib.pyplot as plt
import libraries.networkx_graph as networkx_graph
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair, \
                                     is_path_possible_in_trans_system
from experiments.models.get_markov import get_markov_model, \
    find_first_n_paths_markov

from utils.converter.markov.markov_utils import are_markov_paths_possible_2
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa, \
                                                      remove_empty_state
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_first_n_paths_from_vertex_pair 

from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.petri_net import visualizer as pn_visualizer



class TestMarkovModel(unittest.TestCase):
    # def test1(self):
    #     q0 = TransitionSystem.State(name='q0')
    #     q1 = TransitionSystem.State(name='q1')
    #     q2 = TransitionSystem.State(name='q2')
    #     q3 = TransitionSystem.State(name='q3')
    #     q4 = TransitionSystem.State(name='q4')

    #     q0_q1 = TransitionSystem.Transition(name='(anything1, a)',
    #                                         from_state=q0,
    #                                         to_state=q1)
    #     q1_q2 = TransitionSystem.Transition(name='(anything2, b)',
    #                                         from_state=q1,
    #                                         to_state=q2)
    #     q2_q3 = TransitionSystem.Transition(name='(anything3, a)',
    #                                         from_state=q2,
    #                                         to_state=q3)
    #     q3_q4 = TransitionSystem.Transition(name='(anything4, a)',
    #                                         from_state=q3,
    #                                         to_state=q4)
       
    #     q0.outgoing.add(q0_q1)
    #     q1.outgoing.add(q1_q2)
    #     q2.outgoing.add(q2_q3)
    #     q3.outgoing.add(q3_q4)

    #     ts = TransitionSystem(name='ts')
        
    #     ts.states.add(q0)
    #     ts.states.add(q1)
    #     ts.states.add(q2)
    #     ts.states.add(q3)
    #     ts.states.add(q4)

    #     ts.transitions.add(q0_q1)
    #     ts.transitions.add(q1_q2)
    #     ts.transitions.add(q2_q3)
    #     ts.transitions.add(q3_q4)

    #     # gviz = ts_visualizer.apply(ts)
    #     # ts_visualizer.view(gviz)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=q0,
    #                             final_states=[q4],
    #                             include_empty_state=False)

    #     # edges_list = list(Gd.edges)

    #     # for e in edges_list:
    #     #     print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)

    #     # edges_list = list(Gm.edges)

    #     # for e in edges_list:
    #     #     print('edge: ' + str(e))

    #     self.assertTrue(Gm.has_edge("-", "['-', 'a']"))
    #     self.assertTrue(Gm.has_edge("['a', 'b']", "['b', 'a']"))
    #     self.assertTrue(Gm.has_edge("['b', 'a']", "['a', 'a']"))
    #     self.assertTrue(Gm.has_edge("['a', 'a']", "['a', '-']"))
    #     self.assertTrue(Gm.has_edge("['a', '-']", "-"))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=3)

    #     # edges_list = list(Gm.edges)

    #     # for e in edges_list:
    #     #     print('edge: ' + str(e))        


    # def test2(self):
    #     q0 = TransitionSystem.State(name='q0')
    #     q1 = TransitionSystem.State(name='q1')
    #     q2 = TransitionSystem.State(name='q2')
    #     q3 = TransitionSystem.State(name='q3')

    #     q0_q1 = TransitionSystem.Transition(name='(anything0, a)',
    #                                         from_state=q0,
    #                                         to_state=q1)
    #     q0_q2 = TransitionSystem.Transition(name='(anything1, b)',
    #                                         from_state=q0,
    #                                         to_state=q2)
    #     q1_q2 = TransitionSystem.Transition(name='(anything2, a)',
    #                                         from_state=q1,
    #                                         to_state=q2)
    #     q2_q3 = TransitionSystem.Transition(name='(anything3, b)',
    #                                         from_state=q2,
    #                                         to_state=q3)
               
    #     q0.outgoing.add(q0_q1)
    #     q0.outgoing.add(q0_q2)
    #     q1.outgoing.add(q1_q2)
    #     q2.outgoing.add(q2_q3)

    #     ts = TransitionSystem(name='ts')
        
    #     ts.states.add(q0)
    #     ts.states.add(q1)
    #     ts.states.add(q2)
    #     ts.states.add(q3)

    #     ts.transitions.add(q0_q1)
    #     ts.transitions.add(q0_q2)
    #     ts.transitions.add(q1_q2)
    #     ts.transitions.add(q2_q3)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=q0,
    #                             final_states=[q3],
    #                             include_empty_state=False)

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=1)

    #     self.assertTrue(Gm.has_edge("['a']", "['a']"))
    #     self.assertTrue(Gm.has_edge("['b']", "-"))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)

    #     self.assertTrue(Gm.has_edge("['-', 'a']", "['a', 'a']"))
    #     self.assertTrue(Gm.has_edge("['b', 'b']", "['b', '-']"))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=3)


    # def test3(self):
    #     q0 = TransitionSystem.State(name='q0')
    #     q1 = TransitionSystem.State(name='q1')
    #     q2 = TransitionSystem.State(name='q2')

    #     q0_q1 = TransitionSystem.Transition(name='(anything0, a)',
    #                                         from_state=q0,
    #                                         to_state=q1)
    #     q0_q2 = TransitionSystem.Transition(name='(anything1, b)',
    #                                         from_state=q0,
    #                                         to_state=q2)
    #     q1_q0 = TransitionSystem.Transition(name='(anything1_5, b)',
    #                                         from_state=q1,
    #                                         to_state=q0)
    #     q1_q1 = TransitionSystem.Transition(name='(anything2, c)',
    #                                         from_state=q1,
    #                                         to_state=q1)
    #     q1_q2 = TransitionSystem.Transition(name='(anything2, a)',
    #                                         from_state=q1,
    #                                         to_state=q2)
               
    #     q0.outgoing.add(q0_q1)
    #     q0.outgoing.add(q0_q2)
    #     q1.outgoing.add(q1_q0)
    #     q1.outgoing.add(q1_q1)
    #     q1.outgoing.add(q1_q2)

    #     ts = TransitionSystem(name='ts')
        
    #     ts.states.add(q0)
    #     ts.states.add(q1)
    #     ts.states.add(q2)

    #     ts.transitions.add(q0_q1)
    #     ts.transitions.add(q0_q2)
    #     ts.transitions.add(q1_q0)
    #     ts.transitions.add(q1_q1)
    #     ts.transitions.add(q1_q2)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=q0,
    #                             final_states=[q2],
    #                             include_empty_state=False)

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)

    #     edges_list = list(Gm.edges)

    #     for e in edges_list:
    #         print('edge: ' + str(e))
        
    #     self.assertTrue(Gm.has_edge("-", "['-', 'a']"))
    #     self.assertTrue(Gm.has_edge("['c', 'c']", "['c', 'a']"))
    #     self.assertTrue(Gm.has_edge("['c', 'c']", "['c', 'c']"))
    #     self.assertTrue(Gm.has_edge("['a', 'b']", "['b', 'b']"))
    #     self.assertTrue(Gm.has_edge("['a', 'a']", "['a', '-']"))


    # def test4(self):
    #     my_file = 'petri_nets/IMf/' + \
    #           'ElectronicInvoicingENG.xes.gz.pnml'

    #     net, im, fm = pnml_importer.apply(my_file)
    #     ts = reachability_graph.construct_reachability_graph(net, im)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=None,
    #                             final_states=None,
    #                             include_empty_state=False)

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)

    #     self.assertTrue(Gm.has_edge(
    #                                 "['-', 'Register']", 
    #                                 "['Register', 'Invoice Scanning']")
    #                                )
    #     self.assertTrue(Gm.has_edge(
    #                                 "['Invoice Scanning', "+\
    #                                 "'Scanning of Extra Documentation']", 
    #                                 "['Scanning of Extra Documentation', "+\
    #                                 "'Approve Invoice']")
    #                                )
    #     self.assertTrue(Gm.has_edge(
    #                                 "['Approve Invoice', 'Approve Invoice']", 
    #                                 "['Approve Invoice', 'Approve Invoice']")
    #                                )
    #     self.assertTrue(Gm.has_edge(
    #                                 "['Approve Invoice', 'End']", 
    #                                 "['End', '-']")
    #                                )

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=3)

    #     # edges_list = list(Gm.edges)

    #     # for e in edges_list:
    #     #     print('edge: ' + str(e))

    #     origin = "['Invoice Scanning', "+\
    #              "'Scanning of Extra Documentation', "+\
    #              "'Scanning of Extra Documentation']"
    #     dest = "['Scanning of Extra Documentation', "+\
    #            "'Scanning of Extra Documentation', "+\
    #            "'Approve Invoice']"
    #     self.assertTrue(Gm.has_edge(origin, dest))

    #     origin = "['Register', 'Invoice Scanning', 'Invoice Scanning']"
    #     dest = "['Invoice Scanning', 'Invoice Scanning', 'Invoice Scanning']"
    #     self.assertTrue(Gm.has_edge(origin, dest))

    #     origin = "['Scanning of Extra Documentation', 'Approve Invoice', 'End']"
    #     dest = "['Approve Invoice', 'End', '-']"
    #     self.assertTrue(Gm.has_edge(origin, dest))


    # def test5(self):
    #     q0 = TransitionSystem.State(name='q0')
    #     q1 = TransitionSystem.State(name='q1')
    #     q2 = TransitionSystem.State(name='q2')
    #     q3 = TransitionSystem.State(name='q3')
    #     q4 = TransitionSystem.State(name='q4')

    #     q0_q1 = TransitionSystem.Transition(name='(anything1, a)',
    #                                         from_state=q0,
    #                                         to_state=q1)
    #     q1_q2 = TransitionSystem.Transition(name='(anything2, b)',
    #                                         from_state=q1,
    #                                         to_state=q2)
    #     q2_q3 = TransitionSystem.Transition(name='(anything3, a)',
    #                                         from_state=q2,
    #                                         to_state=q3)
    #     q3_q4 = TransitionSystem.Transition(name='(anything4, a)',
    #                                         from_state=q3,
    #                                         to_state=q4)
       
    #     q0.outgoing.add(q0_q1)
    #     q1.outgoing.add(q1_q2)
    #     q2.outgoing.add(q2_q3)
    #     q3.outgoing.add(q3_q4)

    #     q1.incoming.add(q0_q1)
    #     q2.incoming.add(q1_q2)
    #     q3.incoming.add(q2_q3)
    #     q4.incoming.add(q3_q4)

    #     ts = TransitionSystem(name='ts')
        
    #     ts.states.add(q0)
    #     ts.states.add(q1)
    #     ts.states.add(q2)
    #     ts.states.add(q3)
    #     ts.states.add(q4)

    #     ts.transitions.add(q0_q1)
    #     ts.transitions.add(q1_q2)
    #     ts.transitions.add(q2_q3)
    #     ts.transitions.add(q3_q4)

    #     # gviz = ts_visualizer.apply(ts)
    #     # ts_visualizer.view(gviz)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=q0,
    #                             final_states=[q4],
    #                             include_empty_state=False)

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)

    #     G = tran_sys_to_nx_graph.convert(ts)
    #     n = 200
    #     paths_G = networkx_graph.\
    #                 find_first_n_paths_from_vertex_pair(G, 
    #                                                     v1=None, 
    #                                                     v2=None,
    #                                                     n=n)

    #     self.assertTrue(are_markov_paths_possible_2(Gm, paths_G, 2))


    # def test6(self):
    #     my_file = 'petri_nets/IMf/' + \
    #           'ElectronicInvoicingENG.pnml'

    #     net, im, fm = pnml_importer.apply(my_file)
    #     ts = reachability_graph.construct_reachability_graph(net, im)

    #     G = tran_sys_to_nx_graph.convert(ts)
    #     n = 200
    #     paths_G = networkx_graph.find_first_n_paths_from_vertex_pair(G, 
    #                                                                  v1=None, 
    #                                                                  v2=None,
    #                                                                  n=n)

    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=None,
    #                             final_states=None,
    #                             include_empty_state=True)


    #     Gr = reduceDFA(Gd, include_empty_state=False)
        
    #     # Gd = readable_copy(Gd)
    #     Gd = remove_empty_state(Gd)

    #     Gm = create_mk_abstraction_dfa_2(Gr, k=1)
    #     Gmd = create_mk_abstraction_dfa_2(Gd, k=1)


    #     # edges_list = list(Gr.edges)

    #     # for e in edges_list:
    #     #     print('edge: ' + str(e) + ', activity: ' + str(Gr.edges[e]['activity']))

    #     self.assertTrue(are_markov_paths_possible_2(Gm, paths_G, k=1))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)
    #     self.assertTrue(are_markov_paths_possible_2(Gm, paths_G, k=2))

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=3)
    #     self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=3))


    # def test7(self):
    #     my_file = 'petri_nets/IMf/' + \
    #     'BPI_Challenge_2014_Geo parcel document.pnml'

    #     net, im, fm = pnml_importer.apply(my_file)

    #     print('### Building Reachability Graph ...')
    #     ts = reachability_graph.construct_reachability_graph(net, im)

    #     G = tran_sys_to_nx_graph.convert(ts)

    #     n = 500
    #     paths_G = find_first_n_paths_from_vertex_pair(G, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)

    #     print('### Building DFA ...')
    #     Gd = convert_nfa_to_dfa(ts, 
    #                             init_state=None,
    #                             final_states=None,
    #                             include_empty_state=True)

    #     Gd = remove_empty_state(Gd)

    #     print('### Building Markov Model (k=1)...')

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=1)
    #     self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=1))

    #     print('### Building Markov Model (k=2)...')

    #     Gm = create_mk_abstraction_dfa_2(Gd, k=2)
    #     self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=2))

    #     # print('### Building Markov Model (k=3)...')

    #     # Gm = create_mk_abstraction_dfa_2(Gd, k=3)
    #     # self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=3))
    

    # def test9(self):
    #     my_file = 'petri_nets/IMf/' + \
    #               'BPI_Challenge_2020_InternationalDeclarations.pnml'
    #     net, im, fm = pnml_importer.apply(my_file)
    #     ts = reachability_graph.construct_reachability_graph(net, im)

    #     # gviz = ts_visualizer.apply(ts)
    #     # ts_visualizer.view(gviz)

    #     Gt = tran_sys_to_nx_graph.convert(ts)
    #     n = 10
    #     paths_G = find_first_n_paths_from_vertex_pair(Gt, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)

    #     my_file = my_file[my_file.rfind('/') + 1 : my_file.rfind('.')]
    #     Gm = get_markov_model(my_file, 'IMf', 3)

    #     has_edge = Gm.has_edge(
    #             "['Permit SUBMITTED by EMPLOYEE'," + \
    #             " 'Permit APPROVED by ADMINISTRATION',"+ \
    #             " 'Permit FINAL_APPROVED by SUPERVISOR']",

    #             "['Permit APPROVED by ADMINISTRATION',"+\
    #             " 'Permit FINAL_APPROVED by SUPERVISOR',"+\
    #             " 'Request Payment']"
    #             )

    #     self.assertTrue(has_edge)


    def test8(self):
        my_file = 'petri_nets/IMf/' + \
                  'QUARTA_VARA_CRIMINAL_-_COMARCA_DE_VARZEA'+\
                  '_GRANDE_-_SDCR_-_TJMT.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)

        n = 2000
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        Gd = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)

        print('### Building Markov Model (k=1)...')

        Gm = create_mk_abstraction_dfa_2(Gd, k=1)
        self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=1))

        print('### Building Markov Model (k=2)...')

        Gm = create_mk_abstraction_dfa_2(Gd, k=2)
        self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=2))

        print('### Building Markov Model (k=3)...')

        Gm = create_mk_abstraction_dfa_2(Gd, k=3)
        self.assertTrue(are_markov_paths_possible_2(Gm=Gm, paths=paths_G, k=3))

if __name__ == '__main__':
    unittest.main()