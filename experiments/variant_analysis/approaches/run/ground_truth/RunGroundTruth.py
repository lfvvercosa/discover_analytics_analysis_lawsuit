from pm4py import convert_to_dataframe

from xes_files.creation.LogCreator import LogCreator
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector
        

class RunGroundTruth:

    def run(self, log):
        try:
            log_creator = LogCreator()
            # log = log_creator.remove_single_activity_traces(log)
            # fit_complex = FindFitnessComplexity()
            fit_complex = ComplexFitnessConnector()

            df = convert_to_dataframe(log)
            split_join = SplitJoinDF(df)
            traces = split_join.split_df()
            ids = split_join.ids
            ids_clus = [l[0] for l in ids]

            # df_variants = df[df['case:concept:name'].isin(ids_clus)]
            # df_variants['cluster_label'] = df_variants['case:cluster'].str[-1].astype(int)
            df['cluster_label'] = df['case:cluster'].str[-1].astype(int)

            # fit, complx = fit_complex.get_metrics_from_simulation(df_variants, k_markov)
            fit, complx = fit_complex.run_fitness_and_complexity(df)

            return fit, complx
        
        except Exception as e:
            print(e)

