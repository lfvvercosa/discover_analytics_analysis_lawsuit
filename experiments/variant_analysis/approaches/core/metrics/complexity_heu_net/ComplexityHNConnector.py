import pm4py
from pathlib import Path
from experiments.variant_analysis.approaches.core.metrics.\
     JavaConnector import JavaConnector


class ComplexityHNConnector(JavaConnector):

    path_temp_logs = ''

    def __init__(self, 
                 jar_path='temp/complexity_hn.jar',
                 path_temp_logs='temp/complexity_hn/'):
        super(ComplexityHNConnector, self).__init__(jar_path)
        self.path_temp_logs = path_temp_logs

        self.path_temp_logs = path_temp_logs
        Path(self.path_temp_logs).mkdir(parents=True, exist_ok=True)
    

    def run_complexity_metric(self, df):

        clusters = df['cluster_label'].drop_duplicates().to_list()
        complex_hn = 0
        total_traces = 0

        for c in clusters:
            df_clus = df[df['cluster_label'] == c]
            log = pm4py.convert_to_event_log(df_clus)
            log_path = self.path_temp_logs + 'log_cluster_' + \
                       str(c) + '.xes'
            pm4py.write_xes(log, log_path)

            number_traces = len(log)
            complex = super().run_java(log_path)
            complex_hn += number_traces * complex
            total_traces += number_traces
        
        complex_hn /= total_traces


        return complex_hn
    
