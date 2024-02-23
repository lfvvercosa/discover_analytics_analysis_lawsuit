import numpy as np
from weighted_levenshtein import lev, osa, dam_lev

from pm4py import util as pm_util
from pm4py.util import constants
from pm4py.util import exec_utils
from pm4py.algo.discovery.alpha.utils import endpoints
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.algo.discovery.alpha.data_structures import alpha_classic_abstraction
from pm4py.objects.log.importer.xes import importer as xes_importer


def get_relation(log):
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


def get_insert_cost(s1, s2, weight):
    insert_costs = np.ones(128, dtype=np.float64)
    new_acts = [s for s in s2 if s not in s1]

    for act in new_acts:
        insert_costs[ord(act)] = weight

    
    return insert_costs


def get_delete_cost(s1, s2, weight):
   delete_costs = np.ones(128, dtype=np.float64)
   new_acts = [s for s in s1 if s not in s2]

   for act in new_acts:
        delete_costs[ord(act)] = weight

    
   return delete_costs


def get_transpose_costs(relation_log, weight):
    transpose_costs = np.ones((128, 128), dtype=np.float64)

    for k in relation_log.parallel_relation:
        transpose_costs[ord(k[0]), ord(k[1])] = weight
        transpose_costs[ord(k[1]), ord(k[0])] = weight


    return transpose_costs



if __name__ == '__main__':
    weight_parallel = 0.5
    weight_new_act = 1.5
    weight_substitute = 1

    log_path = 'xes_files/variant_analysis/exp2/p1_v2.xes'
    log = xes_importer.apply(log_path)

    relation_log = get_relation(log)
    transpose_costs = get_transpose_costs(relation_log, weight_parallel)
    substitute_costs = np.full((128, 128), weight_substitute, dtype=np.float64)

    s1 = 'AZZF'
    s2 = 'AJZF'

    delete_costs = get_delete_cost(s1, s2, weight_new_act)
    insert_costs = get_insert_cost(s1, s2, weight_new_act)


    print(
            dam_lev(
                    s1, 
                    s2,
                    insert_costs=insert_costs, 
                    delete_costs=delete_costs,
                    transpose_costs=transpose_costs,
                    substitute_costs=substitute_costs
                   )
         )






