from os import listdir
from os.path import isfile, join, exists
import unittest
import matplotlib.pyplot as plt
import libraries.networkx_graph as networkx_graph
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair, \
                                     is_path_possible_in_trans_system
from experiments.models.get_markov import get_markov_model, \
    find_first_n_paths_markov

from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa, \
                                                      remove_empty_state
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.markov.dfa_to_markov import create_mk_abstraction_dfa
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

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q4],
                                include_empty_state=False)

        edges_list = list(Gd.edges)

        for e in edges_list:
            print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        # print('test')

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(Gm.has_edge("-", "['a', 'b']"))
        self.assertTrue(Gm.has_edge("['a', 'b']", "['b', 'a']"))
        self.assertTrue(Gm.has_edge("['b', 'a']", "['a', 'a']"))
        self.assertTrue(Gm.has_edge("['a', 'a']", "-"))

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e))


    def test2(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')

        q0_q1 = TransitionSystem.Transition(name='(anything0, a)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q2 = TransitionSystem.Transition(name='(anything1, b)',
                                            from_state=q0,
                                            to_state=q2)
        q1_q2 = TransitionSystem.Transition(name='(anything2, a)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q3 = TransitionSystem.Transition(name='(anything3, b)',
                                            from_state=q2,
                                            to_state=q3)
               
        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q2)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q3)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)
        ts.states.add(q3)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q2)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q2_q3)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q3],
                                include_empty_state=False)

        Gm = create_mk_abstraction_dfa(Gd, k=1)

        self.assertTrue(Gm.has_edge("-", "['a']"))
        self.assertTrue(Gm.has_edge("['a']", "['a']"))
        self.assertTrue(Gm.has_edge("['a']", "['b']"))
        self.assertTrue(Gm.has_edge("['b']", "['b']"))
        self.assertTrue(Gm.has_edge("['b']", "-"))
        self.assertFalse(Gm.has_edge("['a']", "-"))

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(Gm.has_edge("['a', 'a']", "['a', 'b']"))
        self.assertTrue(Gm.has_edge("['a', 'b']", "-"))
        self.assertTrue(Gm.has_edge("-", "['b', 'b']"))
        self.assertTrue(Gm.has_edge("['b', 'b']", "-"))

        Gm = create_mk_abstraction_dfa(Gd, k=3)

        self.assertTrue(Gm.has_edge("-", "['a', 'a', 'b']"))
        self.assertTrue(Gm.has_edge("-", "['a', 'a', 'b']"))
        self.assertTrue(Gm.has_edge("-", "['b', 'b']"))
        self.assertTrue(Gm.has_edge("['b', 'b']", "-"))

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e))


    def test3(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(anything0, a)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q2 = TransitionSystem.Transition(name='(anything1, b)',
                                            from_state=q0,
                                            to_state=q2)
        q1_q0 = TransitionSystem.Transition(name='(anything1_5, b)',
                                            from_state=q1,
                                            to_state=q0)
        q1_q1 = TransitionSystem.Transition(name='(anything2, c)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything2, a)',
                                            from_state=q1,
                                            to_state=q2)
               
        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q2)
        q1.outgoing.add(q1_q0)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q2)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q2)
        ts.transitions.add(q1_q0)
        ts.transitions.add(q1_q1)
        ts.transitions.add(q1_q2)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q2],
                                include_empty_state=False)

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(Gm.has_edge("-", "['b']"))
        self.assertTrue(Gm.has_edge("['b']", "-"))
        self.assertTrue(Gm.has_edge("['c', 'c']", "['c', 'a']"))
        self.assertTrue(Gm.has_edge("['c', 'c']", "['c', 'c']"))
        self.assertTrue(Gm.has_edge("['a', 'b']", "['b', 'b']"))
        self.assertFalse(Gm.has_edge("['b', 'a']", "['c', 'b']"))
        self.assertFalse(Gm.has_edge("['c', 'c']", "-"))

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e))


    def test4(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(anything0, a)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q2 = TransitionSystem.Transition(name='(anything1, b)',
                                            from_state=q0,
                                            to_state=q2)
        q1_q0 = TransitionSystem.Transition(name='(anything1_5, b)',
                                            from_state=q1,
                                            to_state=q0)
        q1_q1 = TransitionSystem.Transition(name='(anything2, c)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything2, a)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q1 = TransitionSystem.Transition(name='(anything2, c)',
                                            from_state=q2,
                                            to_state=q1)

        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q2)
        q1.outgoing.add(q1_q0)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q1)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q2)
        ts.transitions.add(q1_q0)
        ts.transitions.add(q1_q1)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q2_q1)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q2],
                                include_empty_state=False)

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(Gm.has_edge("['a', 'c']", "['c', 'a']"))
        self.assertTrue(Gm.has_edge("['c', 'a']", "-"))
        self.assertTrue(Gm.has_edge("['b', 'b']", "-"))

        self.assertFalse(Gm.has_edge("['a', 'c']", "-"))
        self.assertFalse(Gm.has_edge("['b', 'a']", "-"))


        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e))

        # print('test')


    def test4_7(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')

        q0_q1 = TransitionSystem.Transition(name='(anything1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q1 = TransitionSystem.Transition(name='(anything1, b)',
                                            from_state=q1,
                                            to_state=q1)
       
        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q1)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q1_q1)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q1],
                                include_empty_state=False)

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        edges_list = list(Gd.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        # print('test')
        self.assertTrue(Gm.has_edge("['a']", "-"))


    def test4_8(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')
        q4 = TransitionSystem.State(name='q4')

        q0_q1 = TransitionSystem.Transition(name='(anything1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything1, b)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q3 = TransitionSystem.Transition(name='(anything3, a)',
                                            from_state=q2,
                                            to_state=q3)
        q3_q4 = TransitionSystem.Transition(name='(anything3, a)',
                                            from_state=q3,
                                            to_state=q4)


        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q3)
        q3.outgoing.add(q3_q4)

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

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q3],
                                include_empty_state=False)

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        edges_list = list(Gd.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        # print('test')

        self.assertTrue(Gm.has_edge("['b', 'a']", "-"))


    def test5(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.xes.gz.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=False)

        # Gr = readable_copy(Gd)

        # edges_list = list(Gd.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(Gm.has_edge(
                                    "-", 
                                    "['Register', 'Invoice Scanning']")
                                   )
        self.assertTrue(Gm.has_edge(
                                    "['Invoice Scanning', 'Scanning of Extra Documentation']", 
                                    "['Scanning of Extra Documentation', 'Approve Invoice']")
                                   )
        self.assertTrue(Gm.has_edge(
                                    "['Approve Invoice', 'Approve Invoice']", 
                                    "['Approve Invoice', 'Approve Invoice']")
                                   )
        self.assertTrue(Gm.has_edge(
                                    "['Approve Invoice', 'End']", 
                                    "-")
                                   )

        Gm = create_mk_abstraction_dfa(Gd, k=3)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e))

        origin = "['Invoice Scanning', 'Scanning of Extra Documentation', 'Scanning of Extra Documentation']"
        dest = "['Scanning of Extra Documentation', 'Scanning of Extra Documentation', 'Approve Invoice']"
        self.assertTrue(Gm.has_edge(origin, dest))

        origin = "['Register', 'Invoice Scanning', 'Invoice Scanning']"
        dest = "['Invoice Scanning', 'Invoice Scanning', 'Invoice Scanning']"
        self.assertTrue(Gm.has_edge(origin, dest))

        origin = "['Scanning of Extra Documentation', 'Approve Invoice', 'End']"
        dest = "-"
        self.assertTrue(Gm.has_edge(origin, dest))


    def test6(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.xes.gz.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 200
        paths_G = networkx_graph.find_first_n_paths_from_vertex_pair(G, 
                                                                     v1=None, 
                                                                     v2=None,
                                                                     n=n)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=True)
        Gd = remove_empty_state(Gd)

        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(networkx_graph.\
            are_markov_paths_possible(Gm, paths_G, 2))

        Gm = create_mk_abstraction_dfa(Gd, k=3)

        self.assertTrue(networkx_graph.\
            are_markov_paths_possible(Gm=Gm, paths=paths_G, k=3))


    def test7(self):
        my_file = 'petri_nets/IMf/' + \
        'BPI_Challenge_2014_Geo parcel document.xes.gz.pnml'

        net, im, fm = pnml_importer.apply(my_file)

        print('### Building Reachability Graph ...')
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)

        n = 500
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        print('### Building DFA ...')
        Gd = convert_nfa_to_dfa(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=True)

        Gd = remove_empty_state(Gd)

        print('### Building Markov Model ...')
        Gm = create_mk_abstraction_dfa(Gd, k=2)

        self.assertTrue(networkx_graph.\
            are_markov_paths_possible(Gm=Gm, paths=paths_G, k=2))

    
    def test8(self):
        my_file = 'petri_nets/IMf/' + \
                  'QUARTA_VARA_CRIMINAL_-_COMARCA_DE_VARZEA_GRANDE_-_SDCR_-_TJMT.xes.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)

        n = 2000
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        Gd = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)

        # edges_list = list(Gr.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + str(Gr.edges[e]['activity']))

        Gm = create_mk_abstraction_dfa(Gr, k=2)

        self.assertTrue(networkx_graph.\
            are_markov_paths_possible(Gm=Gm, paths=paths_G, k=2))

        Gm = create_mk_abstraction_dfa(Gr, k=3)

        self.assertTrue(networkx_graph.\
            are_markov_paths_possible(Gm=Gm, paths=paths_G, k=3))
        
        print(str(len(paths_G)) + ' paths tested')

    
    def test9(self):
        my_file = 'petri_nets/IMf/' + \
                  '5a_VARA_CIVEL_-_TJMT.xes.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        my_file = my_file[my_file.rfind('/') + 1 : my_file.rfind('.')]
        Gm = get_markov_model(my_file, 'IMf', 3)

        self.assertTrue(
            Gm.has_edge("['Devolução', 'Escrivão/Diretor de Secretaria/Secretário Jurídico', 'Publicação']",
                        "['Escrivão/Diretor de Secretaria/Secretário Jurídico', 'Publicação', 'Decurso de Prazo']")
        )
        
    
    def test10(self):
        my_file = 'petri_nets/IMf/' + \
                  'BPI_Challenge_2020_InternationalDeclarations.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gt = tran_sys_to_nx_graph.convert(ts)
        n = 10
        paths_G = find_first_n_paths_from_vertex_pair(Gt, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        my_file = my_file[my_file.rfind('/') + 1 : my_file.rfind('.')]
        Gm = get_markov_model(my_file, 'IMf', 3)

        has_edge = Gm.has_edge(
                "['Permit SUBMITTED by EMPLOYEE', 'Permit APPROVED by ADMINISTRATION', 'Permit FINAL_APPROVED by SUPERVISOR']",
                "['Permit APPROVED by ADMINISTRATION', 'Permit FINAL_APPROVED by SUPERVISOR', 'Request Payment']"
                )

        self.assertTrue(has_edge)


    def test11(self):
        # my_file = 'petri_nets/IMf/' + \
        #           'BPI_Challenge_2020_InternationalDeclarations.xes.gz.pnml'
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        my_file = my_file[my_file.rfind('/') + 1 : my_file.rfind('.')]
        Gm = get_markov_model(my_file, 'IMf', 2)

        n = 5
        paths_Gm = find_first_n_paths_markov(Gm, n)

        for p in paths_Gm:
            self.assertTrue(is_path_possible_in_trans_system(ts, p))


    def test12(self):
        base_path = 'experiments/features_creation/markov/from_dfa_reduced/k_3/IMf/'
        files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

        for f in files:
            print('         ####### curr file: ' + str(f))
            my_file = 'petri_nets/IMf/' + f[ : f.rfind('.')] + '.pnml'
            net, im, fm = pnml_importer.apply(my_file)

            print('         ####### obtaining reach graph...')
            ts = reachability_graph.construct_reachability_graph(net, im)
            
            # gviz = ts_visualizer.apply(ts)
            # ts_visualizer.view(gviz)

            print('         ####### get markov model...')
            my_file = my_file[my_file.rfind('/') + 1 : my_file.rfind('.')]
            Gm = get_markov_model(my_file, 'IMf', 3)

            print('         ####### find first n paths in markov model...')
            n = 5
            paths_Gm = find_first_n_paths_markov(Gm, n)

            print('         ####### check if path is possible...')
            for p in paths_Gm:
                print('         ####### path being tested:')
                print(p)
                # if not is_path_possible_in_trans_system(ts, p):
                #     print('')
                self.assertTrue(is_path_possible_in_trans_system(ts, p))
            



        



if __name__ == '__main__':
    unittest.main()