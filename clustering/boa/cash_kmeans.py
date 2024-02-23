import pandas as pd
from pm4py import convert_to_dataframe

from clustering.SplitJoinDF import SplitJoinDF
from clustering.boa import n_gram

DEBUG = True


def selectColsByFreq(min_perc, max_perc, df):
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


def rename_columns(df_gram):
        map_name = {n:str(n) for n in df_gram.columns}
        df_gram = df_gram.rename(columns=map_name)


        return df_gram


def cashe_df_grams(log, ngram):
        cashed_dfs = {}
        print('cashing df-grams...')

        for n in ngram:
            print('ngram: ' + str(n))
            cashed_dfs[n] = {}
            for min_max in ngram[n]:
                print('min_max: ' + str(min_max))
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
                df_gram = rename_columns(df_gram)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                rem_cols = selectColsByFreq(min_max[0], min_max[1], df_gram)
                df_gram = df_gram.drop(columns=rem_cols)

                ren_cols = {c:str(c) for c in df_gram.columns}
                df_gram = df_gram.rename(columns=ren_cols)

                # normalize n-gram
                for c in df_gram.columns:
                    if len(df_gram[c].drop_duplicates().to_list()) == 1:
                        df_gram = df_gram.drop(columns=c)

                df_gram_norm = df_gram.copy()

                if DEBUG:
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


        return cashed_dfs, max_cols, min_cols