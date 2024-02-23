from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector
        

import pm4py
from pm4py.objects.log.obj import EventLog

import subprocess
import pandas as pd
import copy
import os
import shutil
from sklearn.metrics import adjusted_rand_score
        
                            


class ActiTracConnector():

    jar_path = ''
    df = None

    def __init__(self, jar_path='temp/actitrac/actitrac.jar'):
        self.jar_path = jar_path

    
    def clear_dir(self, dir):
        folder = dir
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def run_actitrac(self, 
                     n_clusters, 
                     is_greedy,
                     dist_greed,
                     target_fit,
                     min_clus_size,
                     log_path, 
                     saving_path):
        

        self.clear_dir(saving_path)

        subprocess.call(['java', 
                         '-jar', 
                         self.jar_path,
                         log_path,
                         saving_path, 
                         str(n_clusters),
                         str(is_greedy),
                         str(dist_greed),
                         str(target_fit),
                         str(min_clus_size),
                       ])
        
        paths = []

        for i in range(n_clusters):
            file_int = saving_path + 'cluster_' + str(i) + '_internal.xes'
            file_ext = saving_path + 'cluster_' + str(i) + '_external.xes'

            if os.path.isfile(file_int):
                paths.append((file_int, file_ext))
        
        # df_var = self.get_df_variants(paths)
        # return self.get_metrics_values(df_var, k_markov)

        return self.get_metrics_literature(paths)

        
    def get_df_variants(self, paths):
        count = 0
        dfs = []

        for t in paths:
            log_int = pm4py.read_xes(t[0])
            log_ext = pm4py.read_xes(t[1])

            df_int = pm4py.convert_to_dataframe(log_int)
            df_ext = pm4py.convert_to_dataframe(log_ext)
            df_clus = pd.concat([df_int, df_ext])
            df_clus['cluster_label'] = count
            dfs.append(df_clus)

            count += 1
            print()

        df = pd.concat(dfs)
        log_clus = pm4py.convert_to_event_log(df)
        log_var = self.get_log_variants(log_clus)
        df_var = pm4py.convert_to_dataframe(log_var)


        return df_var
    

    def get_log_variants(self, log):
        log_var = EventLog()
        log_var._attributes = copy.copy(log._attributes)
        log_var._extensions = copy.copy(log._extensions)
        log_var._omni = copy.copy(log._omni)
        log_var._classifiers = copy.copy(log._classifiers)
        log_var._properties = copy.copy(log._properties)

        variants = pm4py.get_variants_as_tuples(log)

        for v in variants:
            log_var._list.append(variants[v][0])


        return log_var


    def get_metrics_values(self, df_var, k_markov):
        utils = Utils()
        df_temp = df_var.drop_duplicates(subset=['case:concept:name'])
        
        y_true_temp = df_temp['case:cluster'].to_list()
        y_true = utils.get_ground_truth2(y_true_temp)
        y_pred = df_temp['cluster_label'].to_list()

        ARI = adjusted_rand_score(y_true, y_pred)

        fit_complex = FindFitnessComplexity()
        fit, complex = fit_complex.\
                        get_metrics_from_simulation(df_var, 
                                                    k_markov)

        
        return ARI, fit, complex


    def get_metrics_literature(self, paths):
        count = 0
        dfs = []

        for t in paths:
            log_int = pm4py.read_xes(t[0])
            log_ext = pm4py.read_xes(t[1])

            df_int = pm4py.convert_to_dataframe(log_int)
            df_ext = pm4py.convert_to_dataframe(log_ext)
            df_clus = pd.concat([df_int, df_ext])
            df_clus['cluster_label'] = count
            dfs.append(df_clus)

            count += 1

        df = pd.concat(dfs)
        self.df = df

        log_var = self.get_log_variants(pm4py.convert_to_event_log(df))
        df_var = pm4py.convert_to_dataframe(log_var)
        ARI = self.get_ARI(df_var)

        fit_complex = ComplexFitnessConnector()
        fit, complex = fit_complex.run_fitness_and_complexity(df)


        return ARI, fit, complex
    

    def get_ARI(self, df_var):
        df_temp = df_var.drop_duplicates(subset=['case:concept:name'])
        utils = Utils()
        
        y_true_temp = df_temp['case:cluster'].to_list()
        y_true = utils.get_ground_truth2(y_true_temp)
        y_pred = df_temp['cluster_label'].to_list()

        return adjusted_rand_score(y_true, y_pred)




