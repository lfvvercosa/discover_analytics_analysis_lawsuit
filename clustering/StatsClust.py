import pandas as pd

from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments


def has_outlier_clus(d):
    t = d['count']

    for e in t:
        if e[1] == -1:
            return True
        
    
    return False


class StatsClust:
    
    def get_distrib_df(self, d, n):
        t = d['count']
        distrib = {}
        has_outlier = has_outlier_clus(d)

        if has_outlier:
            offset = 1
            index_list = list(range(-1,n - 1))
        else:
            offset = 0
            index_list = list(range(0,n))

        try:
            for k in t:
                if k[0] not in distrib:
                    distrib[k[0]] = [0] * n
                
                distrib[k[0]][k[1] + offset] = t[k]
            
            distrib['index'] = index_list
            df_distrib = pd.DataFrame.from_dict(distrib)
            df_distrib = df_distrib.set_index('index')
        except Exception as e:
            print(e)
            raise Exception(e)


        return df_distrib


    def get_distrib(self, df, df_ids):
        df_work = df.drop_duplicates('case:concept:name')
        df_work = df_work.merge(df_ids, how='right', on='case:concept:name') 
        df_groups = df_work.groupby(['case:cluster','cluster_label']).\
            agg(count=('case:concept:name','count'))
        n = len(df_work['cluster_label'].drop_duplicates())


        return self.get_distrib_df(df_groups.to_dict(), n)


    def get_variants_by_cluster(self, variants, labels):
        d = {}

        for item in zip(variants, labels):
            if item[1] not in d:
                d[item[1]] = []
            
            d[item[1]].append(item[0])

        
        return d
    

    def get_ground_truth_by_cluster(self, dict_var, traces, y_true):
        gd = {}
        dict_gd = {}

        for i in range(len(traces)):
            gd[traces[i]] = y_true[i]

        for k in dict_var:
            dict_gd[k] = {}

            for t in dict_var[k]:
                truth = 'G' + str(gd[t])

                if truth not in dict_gd[k]:
                    dict_gd[k][truth] = []
                
                dict_gd[k][truth].append(t)

        
        return dict_gd
    

    def get_clusters_edit_dist(self, logs):
        fit = []

        for id1 in range(0, len(logs) - 1):
            for id2 in range(id1 + 1, len(logs)):
                align = logs_alignments.apply(logs[id1], logs[id2])
                fit += [e['fitness'] for e in align]

                # print('ids: ' + str(id1) + ', ' + str(id2) + ', fit: ' + \
                #       str(sum(temp)/len(temp)) + ', size: ' + str(len(temp)))

        avg_fit = sum(fit)/len(fit)

        if avg_fit < 0:
            print()

        return round(sum(fit)/len(fit),4)
    

    def get_distrib2(self, y_true, y_pred):
        dict_distrib = {}

        total_clusters = len(set(y_pred))
        y_true_norm = [y[-2:] for y in y_true]
     
        for y_t, y_p in zip(y_true_norm, y_pred):
            if y_t not in dict_distrib:
                dict_distrib[y_t] = [0]*total_clusters
            
            dict_distrib[y_t][y_p] += 1
        

        return pd.DataFrame.from_dict(dict_distrib)