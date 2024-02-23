from pm4py.algo.discovery.alpha.utils import endpoints
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.algo.discovery.alpha.data_structures import alpha_classic_abstraction
from pm4py import util as pm_util
from pm4py.util import exec_utils
from pm4py.objects.log.importer.xes import importer as xes_importer

from pm4py.util import constants
from enum import Enum


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

parameters = {}
activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters,
                                              pm_util.xes_constants.DEFAULT_NAME_KEY)

log_path = 'xes_files/test_variants/p1_v2.xes'
log = xes_importer.apply(log_path)


dfg = {k: v for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0}
start_activities = endpoints.derive_start_activities_from_log(log, activity_key)
end_activities = endpoints.derive_end_activities_from_log(log, activity_key)

alpha_abstraction = alpha_classic_abstraction.\
    ClassicAlphaAbstraction(start_activities, 
                            end_activities, 
                            dfg,
                            activity_key=activity_key)

print(alpha_abstraction.parallel_relation)