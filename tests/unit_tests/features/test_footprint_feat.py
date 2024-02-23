import unittest
import networkx as nx
from features.baseline_feat import footprint_cost_fit_w, \
                                   footprint_cost_fit, \
                                   footprint_cost_pre_w, \
                                   footprint_cost_pre
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer

class TestFootprintFeature(unittest.TestCase):

    def test1(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),1)
        self.assertEqual(footprint_cost_fit(log,net,im,fm),1)


    def test2(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),1)
        self.assertEqual(footprint_cost_fit(log,net,im,fm),1)
        self.assertEqual(footprint_cost_pre_w(log,net,im,fm), 0.6)


    def test3(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6_2.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + \
        # str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),0.5)
        self.assertEqual(footprint_cost_fit(log,net,im,fm),0.5)
        self.assertEqual(footprint_cost_pre_w(log,net,im,fm),0.4)


    
    def test4(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6_3.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),0.5)
        self.assertEqual(footprint_cost_fit(log,net,im,fm),0.5)
        self.assertEqual(footprint_cost_pre_w(log,net,im,fm),0.6)
    

    def test5(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff5.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_2.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        res = round(footprint_cost_fit_w(log,net,im,fm),2)
        self.assertEqual(res,0.67)

    
    def test6(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff5.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_3.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),0.6)

    
    def test7(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff5.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_3.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),0.6)


    def test8(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),1)
    

    def test9(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff3.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff3.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        # print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))

        self.assertEqual(footprint_cost_fit_w(log,net,im,fm),1)
    

    def test10(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6_4.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res = round(footprint_cost_fit_w(log,net,im,fm),2)
        self.assertEqual(res,0.44)

    
    # def test11(self):
    #     pn_path = 'petri_nets/IMf/ElectronicInvoicingENG.pnml'
    #     log_path = 'xes_files/2/ElectronicInvoicingENG.xes.gz'

    #     net, im, fm = pnml_importer.apply(pn_path)
    #     log = xes_importer.apply(log_path)
        
    #     print('footprint_cost_fit_w Eletronic Invoicing: ' + \
    #         str(footprint_cost_fit_w(log,net,im,fm)))

    def test12(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6_5.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res = round(footprint_cost_fit_w(log,net,im,fm),2)
        self.assertEqual(res,0.44)
        print('footprint_cost_fit_w: ' + str(footprint_cost_fit_w(log,net,im,fm)))
        print('footprint_cost_fit: ' + str(footprint_cost_fit(log,net,im,fm)))


    def test13(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff6_6.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res = round(footprint_cost_pre_w(log,net,im,fm),2)
        self.assertEqual(res,1)

    def test14(self):
        pn_path = 'petri_nets/tests/pm_flower1.pnml'
        log_path = 'xes_files/tests/test_flower_model.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res = round(footprint_cost_pre_w(log,net,im,fm),2)
        self.assertEqual(res,0.17)
    

    def test14(self):
        pn_path = 'petri_nets/tests/pm_flower1.pnml'
        log_path = 'xes_files/tests/test_flower_model2.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res_w = round(footprint_cost_pre_w(log,net,im,fm),2)
        res = round(footprint_cost_pre(log,net,im,fm),2)

        # self.assertEqual(res,0.17)
        print('res_w: ' + str(res_w))
        print('res: ' + str(res))
    

    def test15(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff5.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_4.xes'

        net, im, fm = pnml_importer.apply(pn_path)
        log = xes_importer.apply(log_path)
        
        res = round(footprint_cost_pre_w(log,net,im,fm),2)
        self.assertEqual(res,1)

if __name__ == '__main__':
    unittest.main()