from clustering.ComplexFitnessConnector import ComplexFitnessConnector

import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer

import subprocess
import pandas as pd
import copy
import os
import shutil
import re

from pathlib import Path


class ActiTracConnector():

    jar_path_train = ''
    jar_path_valid = ''
    df = None

    dist_greed = None
    target_fit = None
    min_clus_size = None
    is_greedy = None
    heu_miner_threshold = None
    heu_miner_long_dist = None
    heu_miner_rel_best_thrs = None
    heu_miner_and_thrs = None
    include_external = None
    max_java_heap_actitrac = None

    def __init__(self, 
                 is_greedy,
                 dist_greed,
                 target_fit,
                 min_clus_size,
                 heu_miner_threshold,
                 heu_miner_long_dist,
                 heu_miner_rel_best_thrs,
                 heu_miner_and_thrs,
                 include_external,
                 jar_path_train='temp/actitrac/actitrac.jar',
                 jar_path_valid='temp/actitrac/actitrac_valid.jar',
                 max_java_heap_actitrac='10g'):
        
        self.dist_greed = dist_greed
        self.target_fit = target_fit
        self.min_clus_size = min_clus_size
        self.is_greedy = is_greedy
        self.heu_miner_threshold = heu_miner_threshold
        self.heu_miner_long_dist = heu_miner_long_dist
        self.heu_miner_rel_best_thrs = heu_miner_rel_best_thrs
        self.heu_miner_and_thrs = heu_miner_and_thrs
        self.include_external = include_external

        self.jar_path_train = jar_path_train
        self.jar_path_valid = jar_path_valid
        self.max_java_heap_actitrac = max_java_heap_actitrac

    
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


    def remove_file_if_exists(self, file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File '{file_path}' removed successfully.")
            except OSError as e:
                print(f"Error removing file: {e}")
        else:
            print(f"File '{file_path}' does not exist.")

    def run(self, 
            n_clusters, 
            log_path, 
            saving_path,
            is_return_clusters=False):

        Path(saving_path).mkdir(parents=True, exist_ok=True)
        self.clear_dir(saving_path)

        subprocess.call(['java', 
                         '-Xms'+self.max_java_heap_actitrac, 
                         '-Xmx'+self.max_java_heap_actitrac, 
                         '-jar',
                         self.jar_path_train,
                         log_path,
                         saving_path, 
                         str(n_clusters),
                         str(self.is_greedy),
                         str(self.dist_greed),
                         str(self.target_fit),
                         str(self.min_clus_size),
                         str(self.heu_miner_threshold),
                         str(self.heu_miner_long_dist),
                         str(self.heu_miner_rel_best_thrs),
                         str(self.heu_miner_and_thrs),
                       ])
        
        paths = []

        for i in range(n_clusters):
            file_int = saving_path + 'cluster_' + str(i) + '_internal.xes'
            file_ext = saving_path + 'cluster_' + str(i) + '_external.xes'

            if os.path.isfile(file_int):
                paths.append((file_int, file_ext))
        

        if not is_return_clusters:
            return self.get_metrics_literature(paths)
        else:
            return self.get_clusters(paths)


    def validate(
                 self, 
                 log_valid_path = 'temp/actitrac/valid/log_valid.xes',
                 saving_path='temp/actitrac/valid/results/',
                 log_path_cluster = 'temp/actitrac/train/',
                 act_col='concept:name'
                ):
        
        Path(saving_path).mkdir(parents=True, exist_ok=True)
        self.clear_dir(saving_path)

        subprocess.call(['java', 
                         '-Xms50g', 
                         '-Xmx60g',  
                         '-jar', 
                         self.jar_path_valid,
                         log_valid_path,
                         log_path_cluster,
                         saving_path, 
                         str(self.is_greedy),
                         str(self.heu_miner_threshold),
                         str(self.heu_miner_long_dist),
                         str(self.heu_miner_rel_best_thrs),
                         str(self.heu_miner_and_thrs),
                         str(self.include_external),
                       ])
        
        directory = Path(saving_path)
        xes_paths = [saving_path + f.name for f in directory.iterdir() if f.is_file()]
        df_clus_valid = self.get_clusters_valid(xes_paths)

        self.clear_dir(directory)

        return df_clus_valid

        
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


    def get_metrics_literature(self, paths):
        count = 0
        dfs = []

        for t in paths:
            log_int = xes_importer.apply(t[0], 
                            variant=xes_importer.Variants.LINE_BY_LINE)
            log_ext = xes_importer.apply(t[1], 
                            variant=xes_importer.Variants.LINE_BY_LINE)

            df_int = pm4py.convert_to_dataframe(log_int)
            df_ext = pm4py.convert_to_dataframe(log_ext)
            df_clus = pd.concat([df_int, df_ext])
            df_clus['cluster_label'] = count
            dfs.append(df_clus)

            count += 1

        df = pd.concat(dfs)
        self.df = df

        # Convert 'time:timestamp' column to datetime
        df['time:timestamp'] = df['time:timestamp'].astype(str).str[:-6]
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'],
                                              format= '%Y-%m-%d %H:%M:%S')

        # log_var = self.get_log_variants(pm4py.convert_to_event_log(df))
        # df_var = pm4py.convert_to_dataframe(log_var)

        fit_complex = ComplexFitnessConnector()
        fit, complex = fit_complex.run_fitness_and_complexity(df)


        return fit, complex
    

    def parse_cluster_number(self, name):
        pattern = r"cluster_\d+"
        
        return re.findall(pattern, name)[0][len('cluster_'):]


    def get_clusters(self, paths):
        dfs = []

        for t in paths:
            label = self.parse_cluster_number(t[0])
            log_int = xes_importer.apply(t[0], 
                            variant=xes_importer.Variants.LINE_BY_LINE)
            log_ext = xes_importer.apply(t[1], 
                            variant=xes_importer.Variants.LINE_BY_LINE)

            df_int = pm4py.convert_to_dataframe(log_int)
            df_ext = pm4py.convert_to_dataframe(log_ext)
            df_clus = pd.concat([df_int, df_ext])
            df_clus['cluster_label'] = label
            dfs.append(df_clus)


        df = pd.concat(dfs)
        self.df = df

        df = df[['case:concept:name','cluster_label']]
        # df = df.rename(columns={'cluster_label':'case:lawsuit:cluster_act'})
    

        return df.drop_duplicates(subset='case:concept:name')
    

    def get_clusters_valid(self, paths):
        dfs = []

        for t in paths:
            label = self.parse_cluster_number(t)
            log_ext = xes_importer.apply(t, 
                            variant=xes_importer.Variants.LINE_BY_LINE)

            df_clus = pm4py.convert_to_dataframe(log_ext)
            df_clus['cluster_label'] = label
            dfs.append(df_clus)


        df = pd.concat(dfs)
        self.df = df

        df = df[['case:concept:name','cluster_label']]
        # df = df.rename(columns={'cluster_label':'case:lawsuit:cluster_act'})
    

        return df.drop_duplicates(subset='case:concept:name')
    

    def save_log(self, log, path):
        df_log = pm4py.convert_to_dataframe(log)
        df_log["lifecycle:transition"] =  "complete"

        log = pm4py.convert_to_event_log(df_log)
        log._attributes['concept:name'] = 'Log'

        directory = os.path.dirname(path)
        Path(directory).mkdir(parents=True, exist_ok=True)
        pm4py.write_xes(log, path)

