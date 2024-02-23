import unittest
import networkx as nx
import pickle
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.Vertex import Vertex
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_3 import get_final_states, \
                                                      get_all_reachable_states, \
                                                      convert_nfa_to_dfa_3, \
                                                      get_source
# from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy 
from libraries.networkx_graph import find_paths_from_vertex_pair, \
                                     find_shortest_path_vertex_pair, \
                                     find_first_n_paths_from_vertex_pair, \
                                     is_path_possible
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.visualization.transition_system import visualizer as ts_visualizer


class TestNFAToDFA(unittest.TestCase):
    def test1(self):
        my_file = 'petri_nets/IMf/' + \
          'BPI_Challenge_2014_Control summary.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)
        s0 = get_source(ts.states)
        res = get_final_states(ts.states, s0, {})

        self.assertEqual(1, len(res))
        self.assertEqual('sink1', res[0].name)


    def test2(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q1 = TransitionSystem.Transition(name='(2, a)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(3, None)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q0 = TransitionSystem.Transition(name='(4, b)',
                                            from_state=q2,
                                            to_state=q0)

        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q0)

        q0.incoming.add(q2_q0)
        q1.incoming.add(q0_q1)
        q1.incoming.add(q1_q1)
        q2.incoming.add(q1_q2)

        r = get_all_reachable_states({q0}, 'a', {})
        self.assertCountEqual({q1,q2}, r)

        r = get_all_reachable_states({q0}, 'b', {})
        self.assertCountEqual(set(), r)

        r = get_all_reachable_states({q1,q2}, 'a', {})
        self.assertCountEqual({q1,q2}, r)

        r = get_all_reachable_states({q1,q2}, 'b', {})
        self.assertCountEqual({q0}, r)

        r = get_all_reachable_states({q1}, 'b', {})
        self.assertCountEqual({q0}, r)


    def test2_5(self):
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')
        q3 = TransitionSystem.State(name='q3')
        q4 = TransitionSystem.State(name='q4')

        q1_q2 = TransitionSystem.Transition(name='(1, None)',
                                            from_state=q1,
                                            to_state=q2)
        q1_q3 = TransitionSystem.Transition(name='(2, a)',
                                            from_state=q1,
                                            to_state=q3)
        q3_q2 = TransitionSystem.Transition(name='(3, None)',
                                            from_state=q3,
                                            to_state=q2)
        q2_q4 = TransitionSystem.Transition(name='(4, None)',
                                            from_state=q2,
                                            to_state=q4)
        q4_q1 = TransitionSystem.Transition(name='(5, None)',
                                            from_state=q4,
                                            to_state=q1)

        q1.outgoing.add(q1_q2)
        q1.outgoing.add(q1_q3)
        q2.outgoing.add(q2_q4)
        q3.outgoing.add(q3_q2)
        q4.outgoing.add(q4_q1)

        q1.incoming.add(q4_q1)
        q2.incoming.add(q1_q2)
        q2.incoming.add(q3_q2)
        q3.incoming.add(q1_q3)
        q4.incoming.add(q2_q4)

        r = get_all_reachable_states({q1}, 'a', {})
        self.assertCountEqual({q1,q2,q3,q4}, r)
        
    
    def test3(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q1 = TransitionSystem.Transition(name='(2, a)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(3, None)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q0 = TransitionSystem.Transition(name='(4, b)',
                                            from_state=q2,
                                            to_state=q0)

        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q0)

        q0.incoming.add(q2_q0)
        q1.incoming.add(q0_q1)
        q1.incoming.add(q1_q1)
        q2.incoming.add(q1_q2)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q1_q1)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q2_q0)

        Gd = convert_nfa_to_dfa_3(ts, 
                                init_state=q0,
                                final_states=[q1])

        self.assertListEqual(Gd.edges[('{q0}','{q1,q2}')]['activity'], ['a'])
        self.assertListEqual(Gd.edges[('{q1,q2}','{q1,q2}')]['activity'], ['a'])
        self.assertListEqual(Gd.edges[('{q1,q2}','{q0}')]['activity'], ['b'])
        self.assertListEqual(Gd.edges[('{q0}','{}')]['activity'], ['b'])

        l1 = Gd.edges[('{}','{}')]['activity']
        l1.sort()

        self.assertListEqual(l1, ['a','b'])


    def test4(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q0 = TransitionSystem.Transition(name='(anything1, 0)',
                                            from_state=q0,
                                            to_state=q0)
        q0_q1 = TransitionSystem.Transition(name='(anything2, 0)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q1_2 = TransitionSystem.Transition(name='(anything3, 1)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything4, 0)',
                                            from_state=q1,
                                            to_state=q2)
        q1_q2_2 = TransitionSystem.Transition(name='(anything5, 1)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q2 = TransitionSystem.Transition(name='(anything6, 1)',
                                            from_state=q2,
                                            to_state=q2)

        q0.outgoing.add(q0_q0)
        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q1_2)
        q1.outgoing.add(q1_q2)
        q1.outgoing.add(q1_q2_2)
        q2.outgoing.add(q2_q2)

        q0.incoming.add(q0_q0)
        q1.incoming.add(q0_q1)
        q1.incoming.add(q0_q1_2)
        q2.incoming.add(q1_q2)
        q2.incoming.add(q1_q2_2)
        q2.incoming.add(q2_q2)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)

        ts.transitions.add(q0_q0)
        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q1_2)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q1_q2_2)
        ts.transitions.add(q2_q2)

        Gd = convert_nfa_to_dfa_3(ts, 
                                init_state=q0,
                                final_states=[q1])
        
        self.assertListEqual(Gd.edges[('{q0}','{q0,q1}')]['activity'], ['0'])
        self.assertListEqual(Gd.edges[('{q0,q1}','{q0,q1,q2}')]['activity'], ['0'])

        l1 = Gd.edges[('{q1}','{q2}')]['activity']
        l1.sort()

        self.assertListEqual(l1, ['0','1'])




    def test5(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(1, a)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q1 = TransitionSystem.Transition(name='(2, a)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(3, None)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q0 = TransitionSystem.Transition(name='(4, a)',
                                            from_state=q2,
                                            to_state=q0)

        q0.outgoing.add(q0_q1)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q2)
        q2.outgoing.add(q2_q0)

        q0.incoming.add(q2_q0)
        q1.incoming.add(q0_q1)
        q1.incoming.add(q1_q1)
        q2.incoming.add(q1_q2)

        r = get_all_reachable_states({q0}, 'a', {})
        self.assertCountEqual({q1,q2}, r)


    def test6(self):
        q0 = TransitionSystem.State(name='q0')
        q1 = TransitionSystem.State(name='q1')
        q2 = TransitionSystem.State(name='q2')

        q0_q1 = TransitionSystem.Transition(name='(anything1, 0)',
                                            from_state=q0,
                                            to_state=q1)
        q0_q1_2 = TransitionSystem.Transition(name='(anything2, 1)',
                                            from_state=q0,
                                            to_state=q1)
        q1_q2 = TransitionSystem.Transition(name='(anything4, 0)',
                                            from_state=q1,
                                            to_state=q2)
        q1_q2_2 = TransitionSystem.Transition(name='(anything5, None)',
                                            from_state=q1,
                                            to_state=q2)
        q2_q1 = TransitionSystem.Transition(name='(anything6, 1)',
                                            from_state=q2,
                                            to_state=q2)
        q1_q1 = TransitionSystem.Transition(name='(anything7, 1)',
                                            from_state=q1,
                                            to_state=q1)
        q1_q0 = TransitionSystem.Transition(name='(anything8, 0)',
                                            from_state=q1,
                                            to_state=q0)

        q0.outgoing.add(q0_q1)
        q0.outgoing.add(q0_q1_2)
        q1.outgoing.add(q1_q2)
        q1.outgoing.add(q1_q2_2)
        q1.outgoing.add(q1_q1)
        q1.outgoing.add(q1_q0)
        q2.outgoing.add(q2_q1)

        q0.incoming.add(q1_q0)
        q1.incoming.add(q0_q1)
        q1.incoming.add(q0_q1_2)
        q1.incoming.add(q1_q1)
        q1.incoming.add(q2_q1)
        q2.incoming.add(q1_q2)
        q2.incoming.add(q1_q2_2)

        ts = TransitionSystem(name='ts')
        
        ts.states.add(q0)
        ts.states.add(q1)
        ts.states.add(q2)

        ts.transitions.add(q0_q1)
        ts.transitions.add(q0_q1_2)
        ts.transitions.add(q1_q2)
        ts.transitions.add(q1_q2_2)
        ts.transitions.add(q2_q1)
        ts.transitions.add(q1_q1)
        ts.transitions.add(q1_q0)

        Gd = convert_nfa_to_dfa_3(ts, 
                                init_state=q0,
                                final_states=[q1])
        
        l1 = Gd.edges[('{q0}','{q1,q2}')]['activity']
        l1.sort()

        l2 = Gd.edges[('{q0,q2}','{q1,q2}')]['activity']
        l2.sort()

        self.assertListEqual(l1, ['0','1'])
        self.assertListEqual(l2, ['0','1'])
    
    def test7(self):
        my_file = 'petri_nets/IMf/' + \
              'BPI_Challenge_2014_Entitlement application.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                init_state=None,
                                final_states=None)
        
        edges_list = list(Gd.edges)

        for e in edges_list:
            print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        # print('test')

        self.assertListEqual(Gd.out_edges[('{source1}','{p_121,p_61,p_71,p_91}')]
                             ['activity'], ["mail valid"])
       
    
    def test8(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=False)
        
        # edges_list = list(Gd.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + str(Gd.edges[e]['activity']))

        # print('test')

        self.assertListEqual(Gd.edges[('{p_101,p_131,p_91}','{p_101,p_131,p_91}')]
                             ['activity'], ["Scanning of Extra Documentation"])
        self.assertListEqual(Gd.edges[('{p_51,p_61,p_91}', '{p_51,p_61,p_91}')]
                             ['activity'], ["Invoice Scanning"])

        
    def test10(self):
        my_file = 'petri_nets/IMf/' + \
              'BPI_Challenge_2014_Entitlement application.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        paths_G = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)
        paths_Gd = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

        self.assertTrue(sorted(paths_G) == sorted(paths_Gd))

    
    def test11(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        paths_G = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)
        paths_Gd = find_paths_from_vertex_pair(Gd, '{source1}', '{sink1}')

        if sorted(paths_G) != sorted(paths_Gd):
            print('testing...')

        self.assertTrue(sorted(paths_G) == sorted(paths_Gd))


    def test12(self):
        my_file = 'petri_nets/IMf/' + \
              'ElectronicInvoicingENG.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        short_path_G = find_shortest_path_vertex_pair(G, '{source1}', '{sink1}')

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)
        short_path_Gd = find_shortest_path_vertex_pair(Gd, '{source1}', '{sink1}')

        self.assertTrue(short_path_G == short_path_Gd)

    
    def test12_5(self):
        my_file = 'petri_nets/IMf/' + \
              'VITORIA_-_5a_VARA_CIVEL_-_TJES.pnml'
        
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 200
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)

        for i in range(len(paths_G)):

            self.assertTrue(is_path_possible(Gd, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')


    # check whether DFA has null or duplicated transitions
    def test13(self):
        my_file = 'petri_nets/IMf/' + \
              'Production_Data.pnml'

        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)

        G = tran_sys_to_nx_graph.convert(ts)
        n = 100
        paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                      '{source1}', 
                                                      '{sink1}',
                                                      n=n)

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)

        for i in range(len(paths_G)):

            self.assertTrue(is_path_possible(Gd, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')


    # def test14(self):
    #     my_file = 'petri_nets/HEU_MINER/' + \
    #           'Receipt phase of an environmental permit application process' + \
    #           ' (_WABO_) CoSeLoG project.pnml'

    #     net, im, fm = pnml_importer.apply(my_file)
    #     ts = reachability_graph.construct_reachability_graph(net, im)

    #     G = tran_sys_to_nx_graph.convert(ts)
    #     n = 200
    #     paths_G = find_first_n_paths_from_vertex_pair(G, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)

    #     Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)

    #     for i in range(len(paths_G)):            
    #         self.assertTrue(is_path_possible(Gd, paths_G[i]))
        
    #     print(str(len(paths_G)) + ' paths tested')


    def test15(self):
        my_file = 'petri_nets/IMd/' + \
              'JUIZO_DA_1a_ESCRIVANIA_CRIMINAL_DE_ARAPOEMA_-_TJTO.pnml'
        my_graph = 'experiments/features_creation/dfa/IMd/' + \
              'JUIZO_DA_1a_ESCRIVANIA_CRIMINAL_DE_ARAPOEMA_-_TJTO.xes.txt'

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

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)
        # Gd = pickle.load(open(my_graph, 'rb'))
        # Gs = readable_copy(Gd)

        for i in range(len(paths_G)):            
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested')    


    def test16(self):
        my_file = 'petri_nets/ETM/' + \
              '1a_VARA_CRIMINAL_DA_CAPITAL_-_TJAM.pnml'
        
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

        Gd = convert_nfa_to_dfa_3(ts, include_empty_state=False)
        # Gd = pickle.load(open(my_graph, 'rb'))
        # Gs = readable_copy(Gd)

        for i in range(len(paths_G)):            
            self.assertTrue(is_path_possible(Gd, paths_G[i]))
        
        print(str(len(paths_G)) + ' paths tested') 


if __name__ == '__main__':
    unittest.main()
