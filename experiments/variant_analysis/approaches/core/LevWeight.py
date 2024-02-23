import numpy as np
from weighted_levenshtein import lev, osa, dam_lev
from sklearn.cluster import dbscan

from pm4py import util as pm_util
from pm4py.util import constants
from pm4py.util import exec_utils
from pm4py.algo.discovery.alpha.utils import endpoints
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.algo.discovery.alpha.data_structures import alpha_classic_abstraction
from pm4py.objects.log.importer.xes import importer as xes_importer


class LevWeight:
    variants = None
    clustering = None
    relation_log = None
    
    weight_parallel = 1
    weight_new_act = 1
    weight_substitute = 1

    transpose_costs = None
    substitute_costs = None

    def __init__(self, 
                 variants, 
                 log,
                 weight_parallel,
                 weight_new_act,
                 weight_substitute):
        
        self.variants = variants
        self.relation_log = self.get_relation(log)

        self.weight_parallel = weight_parallel
        self.weight_new_act = weight_new_act
        self.weight_substitute = weight_substitute

        self.transpose_costs = self.get_transpose_costs(self.relation_log, 
                                                        self.weight_parallel)
        self.substitute_costs = np.full((128, 128), 
                                        self.weight_substitute, 
                                        dtype=np.float64)
        

    def get_relation(self, log):
        parameters = {}
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, 
                                                parameters,
                                                pm_util.xes_constants.DEFAULT_NAME_KEY)
        
        dfg = {k: v for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0}
        start_activities = endpoints.derive_start_activities_from_log(log, activity_key)
        end_activities = endpoints.derive_end_activities_from_log(log, activity_key)

        alpha_abstraction = alpha_classic_abstraction.\
            ClassicAlphaAbstraction(start_activities, 
                                    end_activities, 
                                    dfg,
                                    activity_key=activity_key)


        return alpha_abstraction


    def get_insert_cost(self, s1, s2, weight):
        insert_costs = np.ones(128, dtype=np.float64)
        new_acts = [s for s in s2 if s not in s1]

        for act in new_acts:
            insert_costs[ord(act)] = weight

        
        return insert_costs


    def get_delete_cost(self, s1, s2, weight):
        delete_costs = np.ones(128, dtype=np.float64)
        new_acts = [s for s in s1 if s not in s2]

        for act in new_acts:
                delete_costs[ord(act)] = weight

            
        return delete_costs


    def get_transpose_costs(self, relation_log, weight):
        transpose_costs = np.ones((128, 128), dtype=np.float64)

        for k in relation_log.parallel_relation:
            transpose_costs[ord(k[0]), ord(k[1])] = weight
            transpose_costs[ord(k[1]), ord(k[0])] = weight


        return transpose_costs


    def lev_metric_weight(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices

        delete_costs = self.get_delete_cost(self.variants[i], 
                                            self.variants[j], 
                                            self.weight_new_act)
        insert_costs = self.get_insert_cost(self.variants[i], 
                                            self.variants[j], 
                                            self.weight_new_act)

        return dam_lev(
                    self.variants[i], 
                    self.variants[j],
                    insert_costs=insert_costs, 
                    delete_costs=delete_costs,
                    transpose_costs=self.transpose_costs,
                    substitute_costs=self.substitute_costs
                   )

