'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import heapq
import sys
import uuid
from enum import Enum
from copy import deepcopy

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.util import constants, xes_constants, exec_utils
from pm4py.util import variants_util
from pm4py.objects.petri_net.utils import align_utils
from pm4py.objects.log.obj import EventLog, Trace
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.util import typing
from pm4py.objects.conversion.log import converter as log_converter


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    SYNC_COST_FUNCTION = "sync_cost_function"
    MODEL_MOVE_COST_FUNCTION = "model_move_cost_function"
    LOG_MOVE_COST_FUNCTION = "log_move_cost_function"
    INTERNAL_LOG_MOVE_COST_FUNCTION = "internal_log_move_cost_function"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"


class Outputs(Enum):
    ALIGNMENT = "alignment"
    COST = "cost"
    VISITED = "visited_states"
    CLOSED = "closed"
    INTERNAL_COST = "internal_cost"


POSITION_G = 0
POSITION_LOG = 1
POSITION_NUM_VISITED = 2
POSITION_H = 3
POSITION_REAL_F = 4
POSITION_F = 5
POSITION_MODEL = 6
POSITION_IS_LM = 7
POSITION_IS_MM = 8
POSITION_PREV = 9


def apply(obj: Union[EventLog, Trace], dfg: Dict[Tuple[str, str], int], sa: Dict[str, int], ea: Dict[str, int], parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> Union[typing.AlignmentResult, typing.ListAlignments]:
    """
    Applies the alignment algorithm provided a log/trace object, and a *connected* DFG

    Parameters
    --------------
    obj
        Event log / Trace
    dfg
        *Connected* directly-Follows Graph
    sa
        Start activities
    ea
        End activities
    parameters
        Parameters of the algorithm:
        - Parameters.SYNC_COST_FUNCTION: for each activity that is in both the trace and the model, provide the
        non-negative cost of a sync move
        - Parameters.MODEL_MOVE_COST_FUNCTION: for each activity that is in the model, provide the non-negative
        cost of a model move
        - Parameters.LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost of a log move
        that is returned in the alignment to the user (but not used internally for ordering the states)
        - Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost
        of a log move that is used internally for ordering the states in the search algorithm.
        - Parameters.ACTIVITY_KEY: the attribute of the log that is the activity

    Returns
    --------------
    ali
        Result of the alignment
    """
    if isinstance(obj, Trace):
        return apply_trace(obj, dfg, sa, ea, parameters=parameters)
    else:
        return apply_log(obj, dfg, sa, ea, parameters=parameters)


def apply_log(log, dfg, sa, ea, parameters=None):
    """
    Applies the alignment algorithm provided a log object, and a *connected* DFG

    Parameters
    ----------------
    log
        Event log
    dfg
        *Connected* DFG
    sa
        Start activities
    ea
        End activities
    parameters
        Parameters of the algorithm:
        - Parameters.SYNC_COST_FUNCTION: for each activity that is in both the trace and the model, provide the
        non-negative cost of a sync move
        - Parameters.MODEL_MOVE_COST_FUNCTION: for each activity that is in the model, provide the non-negative
        cost of a model move
        - Parameters.LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost of a log move
        that is returned in the alignment to the user (but not used internally for ordering the states)
        - Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost
        of a log move that is used internally for ordering the states in the search algorithm.
        - Parameters.ACTIVITY_KEY: the attribute of the log that is the activity

    Returns
    ----------------
    aligned_traces
        For each trace, contains a dictionary describing the alignment
    """
    if parameters is None:
        parameters = {}

    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    aligned_traces = []
    align_dict = {}
    
    if 'accepts_empty' in parameters and parameters['accepts_empty']:
        al_empty_cost = 0
    else:
        al_empty_cost = \
            __apply_list_activities([], dfg, sa, ea, parameters=parameters)["cost"]
            
    for trace in log:
        trace_act = tuple(x[activity_key] for x in trace)
        if trace_act in align_dict:
            aligned_traces.append(align_dict[trace_act])
        else:
            log_move_cost_function = exec_utils.get_param_value(Parameters.LOG_MOVE_COST_FUNCTION, parameters,
                                                                {x: align_utils.STD_MODEL_LOG_MOVE_COST for x in
                                                                 trace_act})
            trace_bwc_cost = sum(log_move_cost_function[x] for x in trace_act)
            al_tr = __apply_list_activities(trace_act, dfg, sa, ea, parameters=parameters)
            al_tr["fitness"] = 1.0 - al_tr["cost"] / (al_empty_cost + trace_bwc_cost)
            al_tr["bwc"] = al_empty_cost + trace_bwc_cost
            align_dict[trace_act] = al_tr
            aligned_traces.append(align_dict[trace_act])

    return aligned_traces


def apply_trace(trace, dfg, sa, ea, parameters=None):
    """
    Applies the alignment algorithm provided a trace of a log, and a *connected* DFG

    Parameters
    ---------------
    trace
        Trace
    dfg
        *Connected* DFG
    sa
        Start activities
    ea
        End activities
    parameters
        Parameters of the algorithm:
        - Parameters.SYNC_COST_FUNCTION: for each activity that is in both the trace and the model, provide the
        non-negative cost of a sync move
        - Parameters.MODEL_MOVE_COST_FUNCTION: for each activity that is in the model, provide the non-negative
        cost of a model move
        - Parameters.LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost of a log move
        that is returned in the alignment to the user (but not used internally for ordering the states)
        - Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost
        of a log move that is used internally for ordering the states in the search algorithm.
        - Parameters.ACTIVITY_KEY: the attribute of the log that is the activity

    Returns
    ---------------
    ali
        Dictionary describing the alignment
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    trace_act = tuple(x[activity_key] for x in trace)

    return __apply_list_activities(trace_act, dfg, sa, ea, parameters=parameters)


def apply_from_variants_list_dfg_string(var_list, dfg_serialization, parameters=None):
    if parameters is None:
        parameters = {}

    from pm4py.objects.dfg.importer import importer as dfg_importer
    dfg, sa, ea = dfg_importer.deserialize(dfg_serialization, parameters=parameters)

    return apply_from_variants_list(var_list, dfg, sa, ea, parameters=parameters)


def apply_from_variants_list(var_list, dfg, sa, ea, parameters=None):
    if parameters is None:
        parameters = {}

    dictio_alignments = {}
    for varitem in var_list:
        variant = varitem[0]
        dictio_alignments[variant] = apply_from_variant(variant, dfg, sa, ea, parameters=parameters)
    return dictio_alignments


def apply_from_variant(variant, dfg, sa, ea, parameters=None):
    if parameters is None:
        parameters = {}

    trace = variants_util.variant_to_trace(variant, parameters=parameters)
    return apply_trace(trace, dfg, sa, ea, parameters=parameters)


def dijkstra_to_end_node(dfg, sa, ea, start_node, end_node, activities_model, sync_cost_function,
                         model_move_cost_function):
    """
    Gets the cost of the minimum path from a node to the end node

    Parameters
    ---------------
    dfg
        *Connected* DFG
    sa
        Start activities
    ea
        End activities
    start_node
        Start node of the graph (connected to all the start activities)
    end_node
        End node of the graph (connected to all the end activities)
    activities_model
        Set of the activities contained in the DFG
    sync_cost_function
        Given an activity, provides the cost when the activity is executed in a sync way
    model_move_cost_function
        Given an activity, provides the cost when the activity is executed as a move-on-model

    Returns
    -----------------
    cost_to_end_node
        Dictionary associating to each node the cost to the end node
    """
    pred = {x: set() for x in activities_model}
    pred[start_node] = set()
    pred[end_node] = set()
    for x in sa:
        pred[x].add(start_node)
    for x in ea:
        pred[end_node].add(x)
    for x in dfg:
        pred[x[1]].add(x[0])
    min_dist = {x: sys.maxsize for x in activities_model}
    min_dist[start_node] = sys.maxsize
    min_dist[end_node] = 0

    nodes_cost = {}
    nodes_cost[start_node] = 0
    nodes_cost[end_node] = 0

    for x in activities_model:
        if x in sync_cost_function:
            nodes_cost[x] = min(sync_cost_function[x], model_move_cost_function[x])
        else:
            nodes_cost[x] = model_move_cost_function[x]

    to_visit = [end_node]
    while len(to_visit) > 0:
        curr = to_visit.pop(0)
        curr_dist = min_dist[curr]
        for prev_node in pred[curr]:
            prev_dist = min_dist[prev_node]
            if prev_dist > curr_dist + nodes_cost[prev_node]:
                min_dist[prev_node] = curr_dist + nodes_cost[prev_node]
                to_visit.append(prev_node)

    # subtract the cost of the node itself to make the heuristics correct
    for el in min_dist:
        min_dist[el] -= nodes_cost[el]

    return min_dist


def __apply_list_activities(trace, dfg, sa, ea, parameters=None):
    """
    Apply the alignments provided the *connected* DFG and a list of activities of the trace

    Parameters
    --------------
    trace
        List of activities of the trace
    dfg
        *Connected* DFG
    sa
        Start activities
    ea
        End activities
    parameters
        Parameters of the algorithm:
        - Parameters.SYNC_COST_FUNCTION: for each activity that is in both the trace and the model, provide the
        non-negative cost of a sync move
        - Parameters.MODEL_MOVE_COST_FUNCTION: for each activity that is in the model, provide the non-negative
        cost of a model move
        - Parameters.LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost of a log move
        that is returned in the alignment to the user (but not used internally for ordering the states)
        - Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION: for each activity that is in the trace, provide the cost
        of a log move that is used internally for ordering the states in the search algorithm.

    Returns
    --------------
    ali
        Dictionary describing the alignment
    """
    # if parameters is None:
    #     parameters = {}

    suffix = parameters['suffix']

    if 'debug' in parameters:
        my_debug  = parameters['debug']
    else:
        my_debug = False

    activities_trace = set(x for x in trace)
    activities_model = set(x[0] for x in dfg).union(set(x[1] for x in dfg)).union(set(x for x in sa)).union(
        set(x for x in ea))
    activities_one_instantiation = activities_trace.union(activities_model)
    activities_one_instantiation = {x: x for x in activities_one_instantiation}
    activities_both_in_trace_and_model = intersectActivities(activities_model, 
                                                             activities_trace, 
                                                             suffix)
    # activities_both_in_trace_and_model = activities_trace.intersection(activities_model)
    
    #TODO adapt to repeated activities
    sync_cost_function = exec_utils.get_param_value(Parameters.SYNC_COST_FUNCTION, parameters,
                                                    {x: align_utils.STD_SYNC_COST for x in activities_both_in_trace_and_model})
    model_move_cost_function = exec_utils.get_param_value(Parameters.MODEL_MOVE_COST_FUNCTION, parameters,
                                                          {x: align_utils.STD_MODEL_LOG_MOVE_COST for x in activities_model})
    # model_move_cost_function['-'] = 0
    # for each activity that is in the trace, provides the cost of a log move
    # that is returned in the alignment to the user (but not used internally for ordering the states)
    log_move_cost_function = exec_utils.get_param_value(Parameters.LOG_MOVE_COST_FUNCTION, parameters,
                                                        {x: align_utils.STD_MODEL_LOG_MOVE_COST for x in activities_trace})
    # for each activity that is in the trace, provide the cost of a log move that is used internally for
    # ordering the states in the search algorithm.
    internal_log_move_cost_function = exec_utils.get_param_value(Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION, parameters,
                                                                 None)
    if internal_log_move_cost_function is None:
        internal_log_move_cost_function = {}
        for x in activities_trace:
            #TODO adapt to repeated activities
            if containsActivity(x, activities_model, suffix):
            # if x in activities_model:
                internal_log_move_cost_function[x] = log_move_cost_function[x]
            else:
                internal_log_move_cost_function[x] = 0

    start_node = str(uuid.uuid4())
    end_node = str(uuid.uuid4())


    outgoing_nodes = {activities_one_instantiation[x]: set() for x in activities_model}
    outgoing_nodes[start_node] = set()
    for x in dfg:
        outgoing_nodes[x[0]].add(activities_one_instantiation[x[1]])
    for x in sa:
        outgoing_nodes[start_node].add(activities_one_instantiation[x])
    for x in ea:
        outgoing_nodes[activities_one_instantiation[x]].add(end_node)
    
    
    if 'accepts_empty' in parameters and parameters['accepts_empty']:
        outgoing_nodes[start_node].add(end_node)

    # closed set: the set associates each activity of the model with the index of the trace that lastly visited that.
    # this is a very efficient closed set since it cuts double-visiting the same activity for state visited later
    # that have a lower index of a trace.
    closed = {activities_one_instantiation[x]: -1 for x in activities_model}
    closed[start_node] = -1
    closed[end_node] = -1

    f = 0
    h = 0
    g = f + h

    num_visited = 0
    num_closed = 0

    open_set = [(g, 0, num_visited, -h, f, f, start_node, False, False, None)]
    heapq.heapify(open_set)

    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)

        # my_align = __return_alignment(curr, trace, num_closed)
        # if my_debug:
        # temp = deepcopy(curr)
        # fake_end_state = (g, curr[POSITION_LOG] - 1, num_visited, -h, f, f, end_node,
        #                         False, False, temp)
        # my_align = __return_alignment(fake_end_state, trace, num_closed)
        
        # with open('temp/test_align.txt', 'a') as f:
        #     f.write(str(my_align['alignment']) + '\n')
        #     f.write('cost: ' + str(my_align['cost']) + '\n')
        #     f.write('pos log: ' + str(curr[POSITION_LOG]) + '\n')
        #     f.write('pos model:' + str(curr[POSITION_MODEL]) + '\n')
        #     f.write('closed[' + str(curr[POSITION_MODEL]) + '] = ' + \
        #         str(closed[curr[POSITION_MODEL]]) + '\n')
        #     f.write('is model move: ' + str(temp[POSITION_IS_MM]) + '\n')
        #     f.write('is log move: ' + str(temp[POSITION_IS_LM]) + '\n')
        #     f.write('\n')


        # if str(my_align['alignment']) == "[('Start', 'Start!#@#!1'), ('Start', 'Start!#@#!2'), ('watchingtv', '>>'), ('watchingtv', '>>'), ('>>', 'washing!#@#!1'), ('>>', 'washing!#@#!2'), ('>>', 'toilet!#@#!1'), ('>>', 'washing!#@#!3')]":
        #     if curr[POSITION_MODEL] == 'washing!#@#!4' and curr[POSITION_IS_MM]:
        #         print()

        # if str(my_align['alignment']) == "[('Start', 'Start!#@#!1'), ('Start', 'Start!#@#!2'), ('>>', 'washing!#@#!1'), ('>>', 'washing!#@#!2'), ('>>', 'toilet!#@#!1'), ('>>', 'washing!#@#!3'), ('>>', 'toilet!#@#!2'), ('>>', 'washing!#@#!4')]":
        #     if curr[POSITION_MODEL] == 'washing!#@#!4' and curr[POSITION_IS_LM]:
        #         print()

        num_visited = num_visited + 1

        if -curr[POSITION_LOG] <= closed[curr[POSITION_MODEL]]:
            # closed
            continue

        closed[curr[POSITION_MODEL]] = -curr[POSITION_LOG]
        not_end_trace = -curr[POSITION_LOG] < len(trace)

        if curr[POSITION_MODEL] == end_node and not not_end_trace:
            # return __return_alignment(curr, trace, num_closed)
            return __custom_return_alignment(curr, trace, num_closed, end_node)

        num_closed = num_closed + 1

        trace_curr = trace[-curr[POSITION_LOG]] if not_end_trace else None

        if not curr[POSITION_MODEL] == end_node:
            # I can do some sync / move on model
            list_act = outgoing_nodes[curr[POSITION_MODEL]]

            #TODO adapt to repeated activities
            act_not_in_model = not_end_trace and not containsActivity(trace_curr, 
                                                                      activities_model, 
                                                                      suffix)
            # act_not_in_model = not_end_trace and trace_curr not in activities_model

            # avoid scheduling model moves if the activity of the log is not in the model,
            # in that case schedule directly the model move
            if not act_not_in_model: 
                #TODO adapt to repeated activities
                can_sync = not_end_trace and containsActivity(trace_curr,
                                                              list_act,
                                                              suffix)
                # can_sync = not_end_trace and trace_curr in list_act

                if can_sync:
                    new_act = getActivityModel(trace_curr, list_act, suffix)
                    f = curr[POSITION_F] + sync_cost_function[trace_curr]
                    real_f = curr[POSITION_REAL_F] + sync_cost_function[trace_curr]
                    h = 0

                    g = f + h

                    new_state = (g, curr[POSITION_LOG] - 1, num_visited, -h, real_f, f, new_act,
                                 False, False, curr)
                    # new_state = (g, curr[POSITION_LOG] - 1, num_visited, -h, real_f, f, activities_one_instantiation[trace_curr],
                    #              False, False, curr)
                    if -new_state[POSITION_LOG] > closed[new_state[POSITION_MODEL]]:
                        heapq.heappush(open_set, new_state)

                for act in list_act:
                    if act == end_node:
                        new_state = (
                            curr[POSITION_G], curr[POSITION_LOG], num_visited, curr[POSITION_H], curr[POSITION_REAL_F], curr[POSITION_F],
                            end_node, False, False, curr)
                        if -new_state[POSITION_LOG] > closed[new_state[POSITION_MODEL]]:
                            heapq.heappush(open_set, new_state)


                    elif not not_end_trace or not containsActivity(trace_curr,
                                                                   list_act,
                                                                   suffix):
                    # elif not not_end_trace or not trace_curr in list_act: #TODO
                        f = curr[POSITION_F] + model_move_cost_function[act]
                        real_f = curr[POSITION_REAL_F] + model_move_cost_function[act]
                        h = 0

                        g = f + h

                        new_state = (
                            g, curr[POSITION_LOG], num_visited, -h, real_f, f, activities_one_instantiation[act], False,
                            True,
                            curr)
                        if -new_state[POSITION_LOG] > closed[new_state[POSITION_MODEL]]:
                            heapq.heappush(open_set, new_state)

        if not_end_trace and not curr[POSITION_IS_MM]:
        # if not_end_trace:
            # I can do some move-on-log
            f = curr[POSITION_F] + internal_log_move_cost_function[trace_curr]
            real_f = curr[POSITION_REAL_F] + log_move_cost_function[trace_curr]
            h = 0

            g = f + h

            new_state = (g, curr[POSITION_LOG] - 1, num_visited, -h, real_f, f, curr[POSITION_MODEL], True, False, curr)
            
            if -new_state[POSITION_LOG] > closed[new_state[POSITION_MODEL]]:
                heapq.heappush(open_set, new_state)


def __return_alignment(state, trace, closed):
    """
    Returns the computed alignment in an intellegible way

    Parameters
    --------------
    state
        Current state
    trace
        List of activities
    closed
        Number of closed states

    Returns
    --------------
    ali
        Dictionary describing the alignment
    """
    cost = state[POSITION_REAL_F]
    internal_cost = state[POSITION_F]
    visited = state[POSITION_NUM_VISITED]
    moves = []
    while state[POSITION_PREV] is not None:
        state = state[POSITION_PREV]
        is_lm = state[POSITION_IS_LM]
        is_mm = state[POSITION_IS_MM]
        pl = -state[POSITION_LOG] - 1
        if pl >= 0:
            lm = ">>"
            mm = ">>"
            if not is_lm:
                mm = state[POSITION_MODEL]
            if not is_mm:
                lm = trace[pl]
            moves.append((lm, mm))
    moves.reverse()

    return {Outputs.ALIGNMENT.value: moves, Outputs.COST.value: cost, Outputs.VISITED.value: visited,
            Outputs.CLOSED.value: closed, Outputs.INTERNAL_COST.value: internal_cost}


def __custom_return_alignment(state, trace, closed, end_node):
    """
    Returns the computed alignment in an intellegible way

    Parameters
    --------------
    state
        Current state
    trace
        List of activities
    closed
        Number of closed states

    Returns
    --------------
    ali
        Dictionary describing the alignment
    """

    # if state[POSITION_MODEL] != end_node:
    state = (state[0], 
            state[1], 
            state[2],
            state[3],
            state[4],
            state[5], 
            state[6],
            False, 
            False, 
            state)

    cost = state[POSITION_REAL_F]
    internal_cost = state[POSITION_F]
    visited = state[POSITION_NUM_VISITED]
    moves = []
    while state[POSITION_PREV] is not None:
        state = state[POSITION_PREV]
        is_lm = state[POSITION_IS_LM]
        is_mm = state[POSITION_IS_MM]
        pl = -state[POSITION_LOG] - 1
        if pl >= 0:
            lm = ">>"
            mm = ">>"
            if not is_lm:
                mm = state[POSITION_MODEL]
            if not is_mm:
                lm = trace[pl]

            if state[POSITION_MODEL] != end_node:
                moves.append((lm, mm))
            elif is_lm:
                moves.append((lm, mm))
    moves.reverse()

    return {Outputs.ALIGNMENT.value: moves, Outputs.COST.value: cost, Outputs.VISITED.value: visited,
            Outputs.CLOSED.value: closed, Outputs.INTERNAL_COST.value: internal_cost}


def removeSuffix(act, suffix):
    idx = act.rfind(suffix)

    if idx == -1:
        return act
    else:
        return act[:act.rfind(suffix)]


def containsActivity(act, l, suffix):
    s = set()

    for a in l:
        s.add(removeSuffix(a, suffix))
    
    return act in s


def getActivityModel(act, l, suffix):
    for a in l:
        if act == removeSuffix(a, suffix):
            return a
    
    return None


def intersectActivities(act_model, act_trace, suffix):
    s = set()

    for a in act_model:
        s.add(removeSuffix(a, suffix))
    
    return s.intersection(act_trace)