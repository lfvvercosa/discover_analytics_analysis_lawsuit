from pm4py import convert_to_dataframe
import pandas as pd
from sklearn.cluster import KMeans

from clustering.SplitJoinDF import SplitJoinDF
from clustering.boa import n_gram

from core import my_utils
from core import my_filter


class Kmeans():

    DEBUG = True

    log = None
    n_clusters = None
    ngram = None
    model = None
    min_max_perc = None
    min_cols = None
    max_cols = None
    kms_algorithm = None


    def __init__(self, log, n_clusters, ngram, min_max_perc, kms_algorithm):
        self.log = log
        self.n_clusters = n_clusters
        self.ngram = ngram
        self.min_max_perc = min_max_perc
        self.kms_algorithm = kms_algorithm


    def run(self,
            cashed_dfs):
        
        df = convert_to_dataframe(self.log)
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
        df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

        df_gram_norm = cashed_dfs[self.ngram][self.min_max_perc]

        if df_gram_norm is None:
            return None

        if df_gram_norm.empty:
            print('#### Not valid simulation ####')
            return None

        model = KMeans(n_clusters=self.n_clusters, 
                       init='k-means++',
                       algorithm=self.kms_algorithm)
        model.fit(df_gram_norm)
        self.model = model        
        cluster_labels = list(model.labels_)

        if len(set(cluster_labels)) < self.n_clusters:
            print('#### Not valid simulation ####')
            return None

        df_ret = split_join.join_df(cluster_labels)
        df_ret = df_ret[['case:concept:name','cluster_label']]
        df_ret = df_ret.drop_duplicates('case:concept:name')


        return df_ret[['case:concept:name','cluster_label']]
    

    def validate(self, log_valid, df_ref):
        df = convert_to_dataframe(log_valid)
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        
        df_gram_valid = self.create_n_gram(log_valid, self.ngram)
        df_gram_valid = self.equalize_and_normalize_dfs(
                                                        df_ref, 
                                                        df_gram_valid, 
                                                       )
        cluster_labels = self.model.predict(df_gram_valid)
        df_ret = split_join.join_df(cluster_labels)
        df_ret = df_ret[['case:concept:name','cluster_label']]
        df_ret = df_ret.drop_duplicates('case:concept:name')


        return df_ret


    def selectColsByFreq(self, min_perc, max_perc, df):
        df_work = df.copy()
        
        # print('nulls in df_work: ' + str(df_work.isnull().sum().sum()))
        
        all_cols = list(df.columns)

        stack = df_work.stack()
        stack[stack != 0] = 1
        df_work = stack.unstack()

        # df_work[df_work != 0] = 1
        
        total = len(df_work.index)
        cols_freq = df_work.sum()
        sel_cols = cols_freq


        if min_perc is not None:
            min_limit = int(min_perc * total)
            sel_cols = sel_cols[(cols_freq >= min_limit)]
    
    
        if max_perc is not None:
            max_limit = int(max_perc * total)
            sel_cols = sel_cols[(cols_freq <= max_limit)]
   
        sel_cols = sel_cols.index.tolist()
        rem_cols = [c for c in all_cols if c not in sel_cols]


        return rem_cols


    def rename_columns(self, df_gram):
        map_name = {n:str(n) for n in df_gram.columns}
        df_gram = df_gram.rename(columns=map_name)


        return df_gram


    def cashe_df_grams(self, log, ngram):
        cashed_dfs = {}
        # print('cashing df-grams...')

        for n in ngram:
            # print('ngram: ' + str(n))
            cashed_dfs[n] = {}
            for min_max in ngram[n]:
                # print('min_max: ' + str(min_max))
                cashed_dfs[n][min_max] = {}
                # convert to df
                df = convert_to_dataframe(log)

                # create n-gram
                split_join = SplitJoinDF(df)
                split_join.split_df()
                ids = split_join.ids
                ids_clus = [l[0] for l in ids]
                df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                df_gram = n_gram.create_n_gram(df_clus, 
                                               'case:concept:name', 
                                               'concept:name', 
                                               n)
                df_gram = self.rename_columns(df_gram)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                rem_cols = self.selectColsByFreq(min_max[0], min_max[1], df_gram)
                df_gram = df_gram.drop(columns=rem_cols)

                ren_cols = {c:str(c) for c in df_gram.columns}
                df_gram = df_gram.rename(columns=ren_cols)

                # Winsorize
                min_perc = 0.03
                max_perc = 0.03
                df_gram = my_filter.winsorize_df(df_gram, min_perc, max_perc)

                # normalize n-gram
                for c in df_gram.columns:
                    if len(df_gram[c].drop_duplicates().to_list()) == 1:
                        df_gram = df_gram.drop(columns=c)

                df_gram_norm = df_gram.copy()

                if self.DEBUG:
                    df_gram_norm.to_csv('temp/df_gram_norm.csv',sep='\t')

                max_cols = df_gram_norm.max()
                min_cols = df_gram_norm.min()

                df_gram_norm = (df_gram_norm - min_cols)/ \
                                (max_cols - min_cols)
                
                # for c in df_gram.columns:
                    # df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/ \
                                        # (df_gram_norm[c].max() - df_gram_norm[c].min())
                
                df_gram_norm = df_gram_norm.round(4)

                cashed_dfs[n][min_max] = df_gram_norm

        self.min_cols = min_cols
        self.max_cols = max_cols


        return cashed_dfs
    

    def create_n_gram(self, log, n):
        # convert to df
        df = convert_to_dataframe(log)

        # create n-gram
        split_join = SplitJoinDF(df)
        split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
        df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

        df_gram = n_gram.create_n_gram(df_clus, 
                                        'case:concept:name', 
                                        'concept:name', 
                                        n)
        df_gram = self.rename_columns(df_gram)
        df_gram.index = pd.Categorical(df_gram.index, ids_clus)
        df_gram = df_gram.sort_index()


        return df_gram
    

    def equalize_and_normalize_dfs(self, df_ref, df):
        only_in_df_ref = [c for c in df_ref.columns if c not in df.columns]
        only_in_df = [c for c in df.columns if c not in df_ref.columns]

        for c in only_in_df_ref:
            df[c] = 0
        
        df = df.drop(columns=only_in_df)

        df_norm = (df - self.min_cols) / (self.max_cols - self.min_cols)


        return df_norm