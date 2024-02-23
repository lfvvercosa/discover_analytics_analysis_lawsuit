import utils.creation.create_log_from_petri_net as playout_pn
import pm4py


if __name__ == '__main__':

    my_file = "models/petri_nets/tests/variants/exp3_p1.pnml"
    out_path = 'xes_files/test_variants/exp3/exp3_p1.xes'
    params = {
        'number_traces': 50,
        'age_min': 25,
        'age_max': 40,
        'age_upper': 70,
        'age_distrib': 'normal',
        'type_distrib': {'gold':0.75, 'silver':0.125, 'regular':0.125},
        'type_name': 'case:type',
        'resource_distrib': {'Richard':0.33, 'John':0.33, 'Herbert':0.34},
        'resource_name':'case:resource',
    }
    sim_log = playout_pn.playout(my_file, params)
    pm4py.write_xes(sim_log, out_path)

    my_file = "models/petri_nets/tests/variants/exp3_p1_v2.pnml"
    out_path = 'xes_files/test_variants/exp3/exp3_p1_v2.xes'
    params = {
        'number_traces': 50,
        'age_min': 25,
        'age_max': 40,
        'age_upper': 70,
        'age_distrib': 'normal',
        'type_distrib': {'gold':0.75, 'silver':0.125, 'regular':0.125},
        'type_name': 'case:type',
        'resource_distrib': {'Richard':0.33, 'John':0.33, 'Herbert':0.34},
        'resource_name':'case:resource',
    }
    sim_log = playout_pn.playout(my_file, params)
    pm4py.write_xes(sim_log, out_path)

    my_file = "models/petri_nets/tests/variants/exp3_p1_v3.pnml"
    out_path = 'xes_files/test_variants/exp3/exp3_p1_v3.xes'
    params = {
        'number_traces': 50,
        'age_min': 25,
        'age_max': 40,
        'age_upper': 70,
        'age_distrib': 'normal',
        'type_distrib': {'gold':0.75, 'silver':0.125, 'regular':0.125},
        'type_name': 'case:type',
        'resource_distrib': {'Richard':0.33, 'John':0.33, 'Herbert':0.34},
        'resource_name':'case:resource',
    }
    sim_log = playout_pn.playout(my_file, params)
    pm4py.write_xes(sim_log, out_path)

    my_file = "models/petri_nets/tests/variants/exp3_p2.pnml"
    out_path = 'xes_files/test_variants/exp3/exp3_p2.xes'
    params = {
        'number_traces': 50,
        'age_min': 25,
        'age_max': 40,
        'age_upper': 70,
        'age_distrib': 'normal',
        'type_distrib': {'gold':0.75, 'silver':0.125, 'regular':0.125},
        'type_name': 'case:type',
        'resource_distrib': {'Richard':0.33, 'John':0.33, 'Herbert':0.34},
        'resource_name':'case:resource',
    }
    sim_log = playout_pn.playout(my_file, params)
    pm4py.write_xes(sim_log, out_path)

    my_file = "models/petri_nets/tests/variants/exp3_p2_v2.pnml"
    out_path = 'xes_files/test_variants/exp3/exp3_p2_v2.xes'
    params = {
        'number_traces': 50,
        'age_min': 25,
        'age_max': 40,
        'age_upper': 70,
        'age_distrib': 'normal',
        'type_distrib': {'gold':0.75, 'silver':0.125, 'regular':0.125},
        'type_name': 'case:type',
        'resource_distrib': {'Richard':0.33, 'John':0.33, 'Herbert':0.34},
        'resource_name':'case:resource',
    }
    sim_log = playout_pn.playout(my_file, params)
    pm4py.write_xes(sim_log, out_path)