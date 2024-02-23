from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector

from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from pm4py.objects.log.obj import EventLog
import pm4py
import copy


class Agglom:

    def run(self, df, params_agglom, num_clusters):
        agglomClust = CustomAgglomClust()
        fit_complex = ComplexFitnessConnector()

        dend = agglomClust.agglom_fit(df, params_agglom)

        if dend:
            Z = agglomClust.gen_Z(dend)

            if len(dend) < num_clusters:
                labels = list(range(num_clusters))
            else:
                # labels = self.cut_n_clusters(Z, num_clusters)
                labels = hierarchy.fcluster(Z=Z, 
                                            t=num_clusters, 
                                            criterion='maxclust')
                labels = [l-1 for l in labels]

            ARI = self.get_ARI(df, labels)
            df = self.get_new_labels(df, labels)

            fit_complex = ComplexFitnessConnector()
            fit, complex = fit_complex.run_fitness_and_complexity(df)


            return ARI, fit, complex


    
    def get_new_labels(self, df, labels):
        df_map = self.map_labels(labels)
        df_temp = df.merge(df_map, on='cluster_label', how='left')
        df_temp = df_temp.drop(columns='cluster_label')
        df_temp = df_temp.rename(columns={'new_cluster_label':'cluster_label'})
        
        
        return df_temp


    def get_ARI(self, df, labels):
        df_map = self.map_labels(labels)
        df_temp = df.merge(df_map, on='cluster_label', how='left')
        log_temp = pm4py.convert_to_event_log(df_temp)

        log_var_temp = self.get_log_variants(log_temp)
        df_var_temp = pm4py.convert_to_dataframe(log_var_temp)
        df_var_temp = df_var_temp.drop_duplicates(subset='case:concept:name')
        utils = Utils()

        y_true_temp = df_var_temp['case:cluster'].to_list()
        y_true = utils.get_ground_truth2(y_true_temp)
        y_pred = df_var_temp['new_cluster_label'].to_list()


        return adjusted_rand_score(y_true, y_pred)

    
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
    

    def map_labels(self, labels):
        count = 0
        dict_map = {
            'cluster_label':[],
            'new_cluster_label':[],
        }

        for i in range(len(labels)):
            dict_map['cluster_label'].append(count)
            dict_map['new_cluster_label'].append(labels[i])

            count += 1

        
        return pd.DataFrame.from_dict(dict_map)

