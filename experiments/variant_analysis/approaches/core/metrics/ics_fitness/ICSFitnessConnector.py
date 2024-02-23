import subprocess
import pm4py
from pathlib import Path

class ICSFitnessConnector():

    jar_path = ''
    path_temp_logs = ''

    def __init__(self, 
                 jar_path='temp/actitrac/ics.jar', 
                 path_temp_logs = 'temp/ics_fitness/'):
        self.jar_path = jar_path

        self.path_temp_logs = path_temp_logs
        Path(self.path_temp_logs).mkdir(parents=True, exist_ok=True)


    def run_ics_fitness(self,
                        log_path):
        
        comp_proc = subprocess.run(['java', 
                                    '-jar', 
                                    self.jar_path,
                                    log_path,
                                   ],capture_output=True)
        
        return float(comp_proc.stdout)
    

    def run_ics_fitness_metric(self,
                               df):
        
        clusters = df['cluster_label'].drop_duplicates().to_list()
        fitness_ics = 0
        total_traces = 0

        for c in clusters:
            df_clus = df[df['cluster_label'] == c]
            log = pm4py.convert_to_event_log(df_clus)
            log_path = self.path_temp_logs + 'log_cluster_' + \
                       str(c) + '.xes'
            pm4py.write_xes(log, log_path)

            number_traces = len(log)
            ics = self.run_ics_fitness(log_path)
            fitness_ics += number_traces * ics
            total_traces += number_traces
        
        fitness_ics /= total_traces


        return fitness_ics
