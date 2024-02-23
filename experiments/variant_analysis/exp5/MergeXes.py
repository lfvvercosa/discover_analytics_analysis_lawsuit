import pm4py
from pm4py import convert_to_dataframe

import pandas as pd


class MergeXes:
    def merge(self, logs1, logs2):
        dfs1 = []
        dfs2 = []
        count_offset = 1

        for l in logs1:
            dfs1.append(convert_to_dataframe(l))

        for l in logs2:
            dfs2.append(convert_to_dataframe(l))

        count = 1
        for df in dfs1:
            df['case:concept:name'] += '_p1v' + str(count)
            df['case:cluster'] = 'P1V' + str(count)
            df['time:timestamp'] += pd.Timedelta(days=365*count_offset) 

            count += 1
            count_offset += 1

        count = 1
        for df in dfs2:
            df['case:concept:name'] += '_p2v' + str(count)
            df['case:cluster'] = 'P2V' + str(count)
            df['time:timestamp'] += pd.Timedelta(days=365*count_offset) 

            count += 1
            count_offset += 1

        df_all = pd.concat(dfs1 + dfs2)
        log_all = pm4py.convert_to_event_log(df_all)
        
        
        return log_all
        
        # pm4py.write_xes(log_all, out_path)
        # print('done!')