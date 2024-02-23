import utils.read_and_write.s3_handle as s3_handle



class PreProcessClust:


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

        # print('### Removed Acts by TF-IDF:')
        # print(rem_cols)
        # print()


        return rem_cols