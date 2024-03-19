import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from datetime import timedelta


class DatasetSplit():

    # Show time of most severe bottlenecks for trace
    t1 = 0
    t2 = 365
    t3 = 365*3
    t4 = 365*7
    t5 = 365*10
    t6 = 365*100

    bins = pd.IntervalIndex.from_tuples([(t1,t2),
                                         (t2,t3),
                                         (t3,t4),
                                         (t4,t5),
                                         (t5,t6),
                                         ]
                                        )
    # labels = ['very fast','fast','medium','slow','very slow']
    labels = [0,1,2,3,4]


    def gen_categories(self, df, bins, labels, target, name):
        if bins is None:
            bins = self.bins

        if labels is None:
            labels = self.labels

        df[name] = pd.cut(df[target], bins)
        df[name] = df[name].cat.rename_categories(labels)


        return df


    def strat_train_test_split(self, df, bins, labels, target, test_size, seed, name):
        df = self.gen_categories(df, bins, labels, target, name)

        y_cat = df[['cat']].to_numpy()

        cols = list(df.columns)
        cols.remove(target)
        cols.remove('cat')
        cols.append(target)
        df = df[cols]

        X = df[cols].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                 y_cat, 
                                                 test_size=test_size,
                                                 stratify=y_cat,
                                                 shuffle=True, 
                                                 random_state=seed)
        
        
        return X_train, X_test, y_train, y_test

    
    def train_test_split(self, df, target, test_size, seed):
        
        cols = list(df.columns)
        cols.remove(target)
        
        X = df[cols].to_numpy()
        y = df[[target]].to_numpy().ravel()

        X_train, X_test, y_train, y_test = train_test_split(
                                                 X, 
                                                 y, 
                                                 test_size=test_size,
                                                 shuffle=True, 
                                                 random_state=seed
                                           )
        

        return X_train, X_test, y_train, y_test


    def strat_kfold(self, X_train, y_train, k, seed):
        
        skf_gen = StratifiedKFold(k,shuffle=True,random_state=seed).split(X_train, y_train)


        return skf_gen
    

    def kfold(self, X_train, y_train, k, seed):

        skf_gen = KFold(k,shuffle=True,random_state=seed).split(X_train, y_train)


        return skf_gen
