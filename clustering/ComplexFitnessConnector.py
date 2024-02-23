import pm4py
from pathlib import Path
from clustering.JavaConnector import JavaConnector


class ComplexFitnessConnector(JavaConnector):

    path_temp_logs = ''

    def __init__(self, 
                 jar_path='temp/metrics/fit_and_complex.jar',
                 path_temp_logs='temp/logs_clusters/'):
        super(ComplexFitnessConnector, self).__init__(jar_path)
        self.path_temp_logs = path_temp_logs

        self.path_temp_logs = path_temp_logs
        Path(self.path_temp_logs).mkdir(parents=True, exist_ok=True)
    

    def run_fitness_and_complexity(self, df):

        clusters = df['cluster_label'].drop_duplicates().to_list()
        fitness_ics = 0
        complex_hn = 0
        total_traces = 0

        for c in clusters:
            df_clus = df[df['cluster_label'] == c]
            log = pm4py.convert_to_event_log(df_clus)
            log_path = self.path_temp_logs + 'log_cluster_' + \
                       str(c) + '.xes'
            pm4py.write_xes(log, log_path)

            number_traces = len(log)
            output = super().run_java(log_path)
            output = output.decode('ascii').split('\n')

            fitness_ics += number_traces * float(output[0])
            complex_hn += number_traces * float(output[1])
            total_traces += number_traces
        
        fitness_ics /= total_traces
        complex_hn /= total_traces


        return fitness_ics,complex_hn