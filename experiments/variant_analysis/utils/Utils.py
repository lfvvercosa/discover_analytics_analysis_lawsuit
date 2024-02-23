

class Utils:

    def get_ground_truth(self, ids_clus):
        gd = [int(x[-3] + x[-1]) for x in ids_clus]

        return gd
    
    def get_ground_truth2(self, ids_clus):
        return [int(x[-1]) for x in ids_clus]
    

    def get_agglom_labels_by_trace(self, traces, dict_var, labels):
        trace_cluster = self.get_cluster_id(traces, dict_var)

        # print('trace_cluster:')
        # print(trace_cluster)

        return self.get_agglom_id(trace_cluster, labels)


    def get_cluster_id(self, traces, dict_var):
        map_trace_cluster = {}
        trace_cluster = []

        for k in dict_var:
            for e in dict_var[k]:
                map_trace_cluster[e] = k
        

        for e in traces:
            trace_cluster.append(map_trace_cluster[e])


        return trace_cluster
    

    def get_agglom_id(self, trace_cluster, labels):
        agglom_id = []

        for t in trace_cluster:
            agglom_id.append(labels[t])

        
        return agglom_id