import pandas as pd
import pm4py


class SplitJoinDF:
    CASE_ID = 'case:concept:name'
    ACT_LABEL = 'concept:name'

    df = None
    ids = None
    traces = None


    def __init__(self, df):
        self.df = df


    def split_df(self):      
        df_agg = self.df.groupby(self.CASE_ID, as_index=False).agg({self.ACT_LABEL:list})
        df_agg[self.ACT_LABEL] = df_agg[self.ACT_LABEL].str.join('')
        df_agg = df_agg.groupby(self.ACT_LABEL, as_index=False).agg({self.CASE_ID:list})

        self.traces = df_agg[self.ACT_LABEL].to_list()
        self.ids = df_agg[self.CASE_ID].to_list()


        return self.traces
    

    def join_df(self, cluster_labels):
        
        if 'cluster_label' in self.df.columns:
            self.df = self.df.drop(columns='cluster_label')

        id_label = []

        for item in zip(cluster_labels, self.ids):
            for id in item[1]:
                id_label.append((id, item[0]))

        df_id_label = pd.DataFrame.from_dict({'id_label':id_label})
        df_id_label[self.CASE_ID] = df_id_label['id_label'].str[0]
        df_id_label['cluster_label'] = df_id_label['id_label'].str[1]
        df_id_label = df_id_label.drop(columns='id_label')
        self.df = self.df.merge(df_id_label, on=self.CASE_ID, how='left')


        return self.df
    

    def join_df_uniques(self, cluster_labels):
        id_label = []


        for item in zip(cluster_labels, self.ids):
            id_label.append((item[1][0], item[0]))
        
        df_id_label = pd.DataFrame.from_dict({'id_label':id_label})
        df_id_label[self.CASE_ID] = df_id_label['id_label'].str[0]
        df_id_label['cluster_label'] = df_id_label['id_label'].str[1]
        df_id_label = df_id_label.drop(columns='id_label')

        df_temp = self.df.drop(columns=['cluster_label'], errors='ignore')
        df_variants = df_temp.merge(df_id_label, on=self.CASE_ID, how='inner')


        return df_variants
    

    # def obtain_df_uniques(self):

    

    def split_df_clusters(self, df_clust):
        log_clus = []
        clust = df_clust['cluster_label'].drop_duplicates().to_list()

        for c in clust:
            df = df_clust[df_clust['cluster_label'] == c]
            log_clus.append(pm4py.convert_to_event_log(df))
        
        
        return log_clus