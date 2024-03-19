from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
import pm4py
from core import my_log_orchestrator

import pandas as pd
from datetime import timedelta
import core.my_loader as my_loader
import core.my_stats as my_stats
from core import my_create_features


if __name__ == "__main__": 
    base_path = 'dataset/tribunais_trabalho/'
    dfs_names =  [
        'TRT1.xes',
        'TRT2.xes',
        'TRT3.xes',
        'TRT4.xes',
        'TRT5.xes',
        'TRT6.xes',
        'TRT7.xes',
        'TRT8.xes',
        'TRT9.xes',
        'TRT10.xes',
        'TRT11.xes',
        'TRT12.xes',
        'TRT13.xes',
        'TRT14.xes',
        'TRT15.xes',
        'TRT16.xes',
        'TRT17.xes',
        'TRT18.xes',
        'TRT19.xes',
        'TRT20.xes',
        'TRT21.xes',
        'TRT22.xes',
        'TRT23.xes',
        'TRT24.xes',
    ]
    name = 'TRT'
    out_path = 'dataset/tribunais_trabalho/' + name + '.xes'
    DEBUG = True
    perc_act_start = 0.01
    n_stds_outlier = 2


    df_mov = my_log_orchestrator.merge(base_path, dfs_names)
    
    # Show most frequent start and end movements
    my_stats.frequency_start_end_movs(df_mov)

    # Show time of most severe bottlenecks for trace
    t1 = timedelta(days=0)
    t2 = timedelta(days=365)
    t3 = timedelta(days=(365*3))
    t4 = timedelta(days=(365*7))
    t5 = timedelta(days=(365*10))
    t6 = timedelta(days=(365*100))
    bins = pd.IntervalIndex.from_tuples([(t1,t2),
                                         (t2,t3),
                                         (t3,t4),
                                         (t4,t5),
                                         (t5,t6),
                                         ]
                                        )
    print(my_stats.time_bottlenecks(df_mov, 0.2, bins))

    df_mov = my_log_orchestrator.post_processing(df_mov,
                                                 perc_act_start,
                                                 n_stds_outlier)
    
    my_log_orchestrator.create_xes_file(df_mov, out_path, name)


    print('done!')

