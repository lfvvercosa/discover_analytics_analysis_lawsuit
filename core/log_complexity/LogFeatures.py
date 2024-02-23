from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.transition_system import constants
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py import get_variants_as_tuples
from pm4py import convert_to_dataframe

from core.log_complexity.ExtendedPreffixAutomata import ExtendedPreffixAutomata
from clustering.SplitJoinDF import SplitJoinDF

import math
from Levenshtein import distance as levenshtein_distance
from leven import levenshtein 
from collections import Counter


class LogFeatures:

    epa = None
    log = None


    def __init__(self, log):
        self.epa = ExtendedPreffixAutomata(log)
        self.log = log


    def number_events(self):
        events = 0
        
        for t in self.log:
            events += len(t._list)
        
        return events


    def number_events_types(self):
        event_types = {}

        for t in self.log:
            for e in t._list:
                evt = e._dict['concept:name']
                if evt not in event_types:
                    event_types[evt] = True
        
        return len(event_types.keys())


    def number_sequences(self):
        return len(self.log)


    def number_unique_seqs(self):
        seqs = case_statistics.get_variant_statistics(self.log)
        n = len(seqs)
        
        return n


    def percent_unique_seqs(self):
        seqs = case_statistics.get_variant_statistics(self.log)
        n = len(seqs)

        perc_unique = n / len(self.log)
        perc_unique = round(perc_unique, 4)

        return perc_unique


    def avg_sequence_length(self):
        n_events = self.number_events()
        avg = n_events / len(self.log)

        return round(avg, 4)
    

    def min_sequence_length(self):
        min_seq = float('inf')

        for t in self.log:
            seq_size = len(t._list)

            if seq_size < min_seq:
                min_seq = seq_size
           
        
        return min_seq
    

    def max_sequence_length(self):
        max_seq = float('-inf')

        for t in self.log:
            seq_size = len(t._list)

            if seq_size > max_seq:
                max_seq = seq_size
           
        
        return max_seq


    def log_edit_distance(self):
        print('### calc average edit distance...')
        new_log = self.convert_log_to_string(self.log, frequency=True)
        edit_dist = 0
        sum_freq = 0

        for v1 in new_log:
            sum_freq += v1[1]

            for v2 in new_log:
                edit_dist += \
                    levenshtein_distance(v1[0], v2[0])*v1[1]*v2[1]

        edit_dist /= sum_freq*(sum_freq-1)

        return round(edit_dist,4)


    def log_mean_edit_distance(self):
        dist = 0
        count = 0

        df = convert_to_dataframe(self.log)
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()

        for i in range(len(traces)-1):
            for j in range(i+1,len(traces)):
                dist += levenshtein(traces[i], traces[j])
                count += 1

        if len(traces) == 1:
            div = 1
        else:
            div = (len(traces) * (len(traces) - 1))/2
            
        dist /= div


        return round(dist,3)


    def norm_leven(self, s1, s2):
        bigger = max(len(s1), len(s2))
        
        
        return 1 - (bigger - levenshtein(s1, s2))/bigger


    def log_mean_edit_distance_norm(self):
        norm_dist = 0

        df = convert_to_dataframe(self.log)
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()

        for i in range(len(traces)-1):
            for j in range(i+1,len(traces)):
                norm_dist += self.norm_leven(traces[i], traces[j])

        if len(traces) == 1:
            div = 1
        else:
            div = (len(traces) * (len(traces) - 1))/2
            
        norm_dist /= div


        return round(norm_dist,3)


    def convert_log_to_string(self, frequency=False):
        variants = variants_filter.get_variants(self.log)
        mapped_log = []
        letter = ord('a')
        d = {}

        for v in variants:
            l = variants[v][0]._list
            l = [a['concept:name'] for a in l]
            freq = len(variants[v])
            c, letter, d = self.convert(l, letter, d)
            s = ''.join(c)

            if frequency:
                mapped_log.append((s, freq))
            else:
                for i in range(freq):
                    mapped_log.append(s)
        

        return mapped_log


    def convert(self, l, letter, d):
        c = []

        for m in l:
            if not m in d:
                d[m] = chr(letter)
                letter += 1

            c.append(d[m])
                
            
        return (c, letter, d)


    def avg_dist_act(self):
        var = get_variants_as_tuples(self.log)
        avg = 0

        for v in var:
            avg += len(var[v]) * len(set(v))

        avg /= len(self.log)


        return round(avg,2)


    def get_overlap(self, t1,t2):
        set_t1 = set(t1)
        set_t2 = set(t2)

        overlap = len(set_t1.intersection(set_t2)) / \
                len(set_t1.union(set_t2))

        return overlap


    def avg_non_overlap_traces(self):
        var = get_variants_as_tuples(self.log)
        avg = 0
        size_log = len(self.log)

        try:


            for v in var:
                for v2 in var:
                    avg += len(var[v]) * len(var[v2]) * self.get_overlap(v,v2)
            
            avg /= (size_log * size_log)
            avg = 1 - avg

        except Exception as e:
            print(e)

        return round(avg, 2)


    def avg_trace_size(self):
        var = get_variants_as_tuples(self.log)
        avg = 0


        for v in var:
            avg += len(var[v]) * len(v)
        
        avg /= len(self.log)


        return round(avg, 2)
    

    def variant_entropy(self):
        total_states = len(self.epa.states)
        sum_entropies = 0
        partitions_freq = self.epa.partitions.values()
        counter = Counter(partitions_freq)
        total_partitions = max(partitions_freq)

        for i in range(1, total_partitions + 1):
            sum_entropies += counter[i] * math.log(counter[i])
        
        Ev = total_states * math.log(total_states) - \
             sum_entropies
        
        Ev_norm = Ev/(total_states * math.log(total_states))

        return Ev, Ev_norm
    

    def __preffix_partition(self, states, partitions):
        freq = []

        for s in states:
            freq += [partitions[s]] * len(s.data[constants.ALL_INCOMING])

        
        return freq
    

    def sequence_entropy(self):
        preffix_part_freq = self.__preffix_partition(self.epa.states, 
                                                     self.epa.partitions)
        total_preffix = len(preffix_part_freq)
        counter = Counter(preffix_part_freq)
        total_partitions = max(preffix_part_freq)
        sum_entropies = 0

        for i in range(1, total_partitions + 1):
              sum_entropies += counter[i] * math.log(counter[i])

        Es = total_preffix * math.log(total_preffix) - \
             sum_entropies
        
        Es_norm = Es/(total_preffix * math.log(total_preffix))


        return Es, Es_norm 


# if __name__ == '__main__':
#     path_log = 'xes_files/test/test_markov2.xes'
#     log = xes_importer.apply(path_log)

#     print(self.avg_sequence_length(log))

