from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2
import numpy as np


class MarkovMeasures:
    Gm = None
    k = -1
    edges_freq = []

    def __init__(self, log, k):
        self.Gm = create_mk_abstraction_log_2(log=log, k=k)
        self.k = k

        for e in self.Gm.edges:
            self.edges_freq.append(self.Gm[e[0]][e[1]]['weight'])

    
    def get_fitness_gaussian(self, n):
        mean = np.mean(self.edges_freq)
        std = np.std(self.edges_freq)
        thres = max(mean - n*std, 1)

        edges_freq_thres = [x for x in self.edges_freq if x > (mean - thres)]
        fitness = len(edges_freq_thres)/len(self.edges_freq)


        return fitness
    

    def get_fitness_mean(self, p):
        mean = np.mean(self.edges_freq)

        edges_freq_thres = [x for x in self.edges_freq if x > (p*mean)]
        fitness = len(edges_freq_thres)/len(self.edges_freq)


        return fitness
    

    def get_fitness_mean2(self, n):
        data = self.reject_outliers(self.edges_freq, 3)
        mean = np.mean(data)

        edges_freq_thres = [x for x in self.edges_freq if x > (n*mean)]
        fitness = len(edges_freq_thres)/len(self.edges_freq)


        return fitness
    

    def reject_outliers(self, distrib, m=2):
        data = np.array(distrib)


        return list(data[abs(data - np.median(data)) <= m * np.std(data)])


    def get_network_complexity(self):
        total_nodes = len(self.Gm.nodes)
        total_edges = len(self.Gm.edges)

        return (total_edges**2)/total_nodes
    

    def get_percent_click(self):
        total_nodes = len(self.Gm.nodes)
        total_edges = len(self.Gm.edges)
        total_possible_edges = (total_nodes * (total_nodes - 1))

        return total_edges/total_possible_edges
    

    def get_edges_freq_distrib(self):


        return self.edges_freq