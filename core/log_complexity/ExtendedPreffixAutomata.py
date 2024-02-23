import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.objects.transition_system import constants
from pm4py.visualization.transition_system import visualizer as ts_visualizer


class ExtendedPreffixAutomata:

    root = None
    states = None
    transitions = None
    activities = None
    partitions = None
    map_state_to_events = None
    ts = None


    def __init__(self, log):
        df_log = pm4py.convert_to_dataframe(log)
        df_log = df_log.sort_values(by='time:timestamp',ascending=True)
        log_plain = df_log[['concept:name','case:concept:name']].\
                    values.tolist()

        ts = TransitionSystem(name='ts')
        root = TransitionSystem.State(name='root')
        ts.states.add(root)
        count_state = 0

        pred = {}
        states = set()
        transitions = set()
        activities = set()
        partitions = {}
        map_state_to_events = {}

        curr_partition = 1

        for e in log_plain:
            act = e[0]
            id = e[1]
            pred_e = self.__get_predecessor(pred, root, id)
            curr_state = None

            if act in pred_e.data[constants.MAP_OUTGOING]:
                curr_state = pred_e.data[constants.MAP_OUTGOING]\
                             [act].to_state
            else:
                if pred_e.data[constants.MAP_OUTGOING]:
                    curr_partition = max(partitions.values()) + 1
                else:
                    if pred_e == root:
                        curr_partition = 1
                    else:
                        curr_partition = partitions[pred_e]
                
                (new_state, transition) = self.__update_trans_syst(act, 
                                                                   pred_e, 
                                                                   count_state, 
                                                                   ts)
                states.add(new_state)
                transitions.add(transition)
                activities.add(act)
                partitions[new_state] = curr_partition
                curr_state = new_state

                count_state += 1

            unique_event = act + '_' + str(id)
            curr_state.data[constants.ALL_INCOMING].append(unique_event)

            if curr_state not in map_state_to_events:
                map_state_to_events[curr_state] = []
            
            map_state_to_events[curr_state] += unique_event
            self.__update_predecessor(pred, id, curr_state)


        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        # for s in states:
        #     print(s.name + ':' + str(s.data[constants.ALL_INCOMING]))

        # print('partitions: ' + str(partitions))

        self.root = root
        self.states = states
        self.transitions = transitions
        self.activities = activities
        self.partitions = partitions
        self.map_state_to_events = map_state_to_events
        self.ts = ts


    def __get_predecessor(self, pred, root, id):
        if id in pred:
            return pred[id]
        else:
            return root
    
    def __update_predecessor(self, pred, id, state):
        pred[id] = state

    
    def __update_trans_syst(self, act, from_state, count_state, ts):
        to_state = TransitionSystem.State(name='s' + str(count_state))
        transition = TransitionSystem.Transition(name=act,
                                                 from_state=from_state,
                                                 to_state=to_state)
        from_state.outgoing_data(transition)
        to_state.incoming_data(transition)

        ts.states.add(to_state)
        ts.transitions.add(transition)


        return (to_state, transition)
        

    # def create_extended_automata(self, log):
    #     df_log = pm4py.convert_to_dataframe(log)
    #     df_log = df_log.sort_values(by='time:timestamp',ascending=True)
    #     log_plain = df_log[['concept:name','case:concept:name']].\
    #                 values.tolist()

    #     ts = TransitionSystem(name='ts')
    #     root = TransitionSystem.State(name='root')
    #     ts.states.add(root)
    #     count_state = 0

    #     pred = {}
    #     states = set()
    #     transitions = set()
    #     activities = set()
    #     partitions = {}
    #     map_state_to_events = {}

    #     curr_partition = 1

    #     for e in log_plain:
    #         act = e[0]
    #         id = e[1]
    #         pred_e = self.__get_predecessor(pred, root, id)
    #         curr_state = None

    #         if act in pred_e.data[constants.MAP_OUTGOING]:
    #             curr_state = pred_e.data[constants.MAP_OUTGOING]\
    #                          [act].to_state
    #         else:
    #             if pred_e.data[constants.MAP_OUTGOING]:
    #                 curr_partition = max(partitions.values()) + 1
    #             else:
    #                 if pred_e == root:
    #                     curr_partition = 1
    #                 else:
    #                     curr_partition = partitions[pred_e]
                
    #             (new_state, transition) = self.__update_trans_syst(act, 
    #                                                                pred_e, 
    #                                                                count_state, 
    #                                                                ts)
    #             states.add(new_state)
    #             transitions.add(transition)
    #             activities.add(act)
    #             partitions[new_state] = curr_partition
    #             curr_state = new_state

    #             count_state += 1

    #         unique_event = act + '_' + str(id)
    #         curr_state.data[constants.ALL_INCOMING].append(unique_event)

    #         if curr_state not in map_state_to_events:
    #             map_state_to_events[curr_state] = []
            
    #         map_state_to_events[curr_state] += unique_event
    #         self.__update_predecessor(pred, id, curr_state)


    #     gviz = ts_visualizer.apply(ts)
    #     ts_visualizer.view(gviz)


    #     return (root, 
    #             states, 
    #             transitions, 
    #             activities, 
    #             partitions, 
    #             map_state_to_events,
    #             ts)