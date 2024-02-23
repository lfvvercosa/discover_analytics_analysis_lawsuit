import unittest
import networkx as nx
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair, \
                                     is_path_possible
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import find_distinguish_pairs, \
                                                     get_distinct_sets, \
                                                     create_reduced_graph, \
                                                     get_final_state_dict, \
                                                     init_distinguish, \
                                                     distinguish_final_states, \
                                                     find_nodes_reachability, \
                                                     reduceDFA


class TestNFAToDFA(unittest.TestCase):
    def test1(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')
        q4 = TransitionSystem.State(name='q4')

        q0_q1 = TransitionSystem.Transition(name='(1, 0)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q3 = TransitionSystem.Transition(name='(1.5, 1)',
                                            from_state=q0,
                                            to_state=q3)
        q1_q2 = TransitionSystem.Transition(name='(3, 0)',
                                            from_state=q1,
                                            to_state=q2)
        q1_q4 = TransitionSystem.Transition(name='(4, 1)',
                                            from_state=q1,
                                            to_state=q4)
        q2_q1 = TransitionSystem.Transition(name='(5, 0)',
                                            from_state=q2,
                                            to_state=q1)                                
        q2_q4 = TransitionSystem.Transition(name='(6, 1)',
                                            from_state=q2,
                                            to_state=q4)
        q3_q2 = TransitionSystem.Transition(name='(7, 0)',
                                            from_state=q3,
                                            to_state=q2)
        q3_q4 = TransitionSystem.Transition(name='(8, 1)',
                                            from_state=q3,
                                            to_state=q4)
        q4_q4 = TransitionSystem.Transition(name='(9, 0)',
                                            from_state=q4,
                                            to_state=q4)
        q4_q4_2 = TransitionSystem.Transition(name='(10, 1)',
                                            from_state=q4,
                                            to_state=q4)                                                                                                                                      

        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q3)
        q1.outgoing.add(q1_q4)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q1)
        q2.outgoing.add(q2_q4)
        q3.outgoing.add(q3_q2)
        q3.outgoing.add(q3_q4)
        q4.outgoing.add(q4_q4)
        q4.outgoing.add(q4_q4_2)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)
        ts.states.add(q3)
        ts.states.add(q4)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q3)
        ts.transitions.add(q1_q4)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q2_q1)
        ts.transitions.add(q2_q4)
        ts.transitions.add(q3_q2)
        ts.transitions.add(q3_q4)
        ts.transitions.add(q4_q4)
        ts.transitions.add(q4_q4_2)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q4])
        
        reach = find_nodes_reachability(Gd)
        distinguish = init_distinguish(reach)
        distinguish = distinguish_final_states(distinguish, Gd)

        self.assertTrue(distinguish['{q4}']['{q0}'])
        self.assertTrue(distinguish['{q4}']['{q1}'])
        self.assertFalse(distinguish['{q4}']['{q4}'])

        distinguish = find_distinguish_pairs(Gd)

        self.assertFalse(distinguish['{q1}']['{q2}'])
        self.assertFalse(distinguish['{q3}']['{q2}'])
        self.assertTrue(distinguish['{q0}']['{q1}'])
        self.assertTrue(distinguish['{q2}']['{q4}'])

        sets = get_distinct_sets(distinguish)

        for s in sets:
            if '{q1}' in s:
                self.assertTrue('{q2}' in s)
                self.assertTrue('{q3}' in s)
                self.assertFalse('{q4}' in s)
                self.assertFalse('{q0}' in s)

        Gr = create_reduced_graph(Gd, sets)
        final_states = get_final_state_dict(Gr)

        self.assertTrue(final_states['{{q4}}'])
        self.assertFalse(final_states['{{q1}#{q2}#{q3}}'])

        l1 = Gr.edges[('{{q0}}','{{q1}#{q2}#{q3}}')]['activity']
        l1.sort()

        l2 = Gr.edges[('{{q1}#{q2}#{q3}}',
                       '{{q1}#{q2}#{q3}}')]['activity']
        l2.sort()

        self.assertListEqual(l1, ['0','1'])
        self.assertListEqual(l2, ['0'])

        Gr2 = reduceDFA(Gd)

        self.assertTrue(nx.is_isomorphic(Gr, Gr2))


    def test2(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')
        q4 = TransitionSystem.State(name='q4')
        q5 = TransitionSystem.State(name='q5')

        q0_q1 = TransitionSystem.Transition(name='(1, 0)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q2 = TransitionSystem.Transition(name='(2, 1)',
                                            from_state=q0,
                                            to_state=q2)
        q1_q2 = TransitionSystem.Transition(name='(3, 0)',
                                            from_state=q1,
                                            to_state=q2)
        q1_q3 = TransitionSystem.Transition(name='(4, 1)',
                                            from_state=q1,
                                            to_state=q3)
        q2_q2 = TransitionSystem.Transition(name='(5, 0)',
                                            from_state=q2,
                                            to_state=q2)                                
        q2_q4 = TransitionSystem.Transition(name='(6, 1)',
                                            from_state=q2,
                                            to_state=q4)
        q3_q3 = TransitionSystem.Transition(name='(7, 0)',
                                            from_state=q3,
                                            to_state=q3)
        q3_q3_2 = TransitionSystem.Transition(name='(8, 1)',
                                            from_state=q3,
                                            to_state=q3)
        q4_q4 = TransitionSystem.Transition(name='(9, 0)',
                                            from_state=q4,
                                            to_state=q4)
        q4_q4_2 = TransitionSystem.Transition(name='(10, 1)',
                                            from_state=q4,
                                            to_state=q4)
        q5_q4 = TransitionSystem.Transition(name='(11, 1)',
                                            from_state=q5,
                                            to_state=q4)
        q5_q5 = TransitionSystem.Transition(name='(12, 0)',
                                            from_state=q5,
                                            to_state=q5)                                                                                                                                        

        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q2)
        q1.outgoing.add(q1_q3)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q2)
        q2.outgoing.add(q2_q4)
        q3.outgoing.add(q3_q3)
        q3.outgoing.add(q3_q3_2)
        q4.outgoing.add(q4_q4)
        q4.outgoing.add(q4_q4_2)
        q5.outgoing.add(q5_q4)
        q5.outgoing.add(q5_q5)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)
        ts.states.add(q3)
        ts.states.add(q4)
        ts.states.add(q5)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q2)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q1_q3)
        ts.transitions.add(q2_q2)
        ts.transitions.add(q2_q4)
        ts.transitions.add(q3_q3)
        ts.transitions.add(q3_q3_2)
        ts.transitions.add(q4_q4)
        ts.transitions.add(q4_q4_2)
        ts.transitions.add(q5_q4)
        ts.transitions.add(q5_q5)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=q0,
                                final_states=[q3, q4])
        
        reach = find_nodes_reachability(Gd)
        distinguish = init_distinguish(reach)
        distinguish = distinguish_final_states(distinguish, Gd)

        self.assertTrue(distinguish['{q3}']['{q0}'])
        self.assertTrue(distinguish['{q4}']['{q1}'])
        self.assertFalse(distinguish['{q4}']['{q3}'])

        distinguish = find_distinguish_pairs(Gd)

        self.assertFalse(distinguish['{q3}']['{q4}'])
        self.assertFalse(distinguish['{q1}']['{q2}'])
        self.assertTrue(distinguish['{q0}']['{q1}'])
        self.assertTrue(distinguish['{q2}']['{q4}'])
      
        sets = get_distinct_sets(distinguish)

        for s in sets:
            if '{q1}' in s:
                self.assertTrue('{q2}' in s)
                self.assertFalse('{q3}' in s)
                self.assertFalse('{q4}' in s)

            if '{q3}' in s:
                self.assertTrue('{q4}' in s)
                self.assertFalse('{q2}' in s)
                self.assertFalse('{q1}' in s)
                self.assertFalse('{q0}' in s)
        
        Gr = create_reduced_graph(Gd, sets)
        final_states = get_final_state_dict(Gr)

        self.assertTrue(final_states['{{q3}#{q4}}'])
        self.assertFalse(final_states['{{q1}#{q2}}'])

        Gr2 = reduceDFA(Gd)

        self.assertTrue(nx.is_isomorphic(Gr, Gr2))

        # edges_list = list(Gr.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gr.edges[e]['activity']))
        
        # print('test')


    def test3(self):
        my_file = 'petri_nets/IMf/' + \
                  'Production_Data.xes.gz.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 100
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      '{source1}', 
                                                      '{sink1}',
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=True)


        Gd = readable_copy(Gd)   
        Gr = reduceDFA(Gd, include_empty_state=False)
        # Gs = readable_copy(Gr)

        for i in range(len(paths_G)):
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
            self.assertTrue(is_path_possible(Gr, paths_G[i]))


        # edges_list = list(Gr.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gr.edges[e]['activity']))


    # check whether DFA has null or duplicated transitions
    def test4(self):
        my_file = 'petri_nets/IMf/' + \
                  'activitylog_uci_detailed_labour.xes.gz.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)
        G = tran_sys_to_nx_graph.convert(ts)

        n = 300
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        Gd = readable_copy(Gd)
        # Gr1 = reduceDFA(Gd, include_empty_state=True)
        Gr = reduceDFA(Gd, include_empty_state=False)
        # Gs = readable_copy(Gr)

        edges_list = list(Gr.edges)

        for e in edges_list:
            print('edge: ' + str(e) + ', activity: ' + str(Gr.edges[e]['activity']))


        for v in Gr.nodes:
            d = {}
            for e in Gr.edges(v.name):
                acts = Gr.edges[e]['activity']
                for a in acts:
                    if a == 'None':
                        print('test')

                    self.assertTrue(a != 'None')

                    if a not in d:
                        d[a] = True
                    else:
                        print('test')
                        self.assertTrue(False)
    
        for i in range(len(paths_G)):
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
            self.assertTrue(is_path_possible(Gr, paths_G[i]))

    
    def test5(self):
        my_file = 'petri_nets/IMf/' + \
                  '1a_VARA_CRIMINAL_DA_CAPITAL_-_TJAM.xes.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 200
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        # Gs = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)
        # Gr = readable_copy(Gr)

        for i in range(len(paths_G)):
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
            self.assertTrue(is_path_possible(Gr, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')


    def test6(self):
        my_file = 'petri_nets/ETM/' + \
                  '1a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 200
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        Gd = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)
        # Gr = readable_copy(Gr)

        for i in range(len(paths_G)):
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
            self.assertTrue(is_path_possible(Gr, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')


    def test7(self):
        my_file = 'petri_nets/ETM/' + \
                  'BPI_Challenge_2020_DomesticDeclarations.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        G = tran_sys_to_nx_graph.convert(ts)

        n = 200
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
        Gd = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)

        for i in range(len(paths_G)):
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
            self.assertTrue(is_path_possible(Gr, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')


    




if __name__ == '__main__':
    unittest.main()
