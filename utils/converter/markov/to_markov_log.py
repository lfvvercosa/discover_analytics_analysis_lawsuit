from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils.creation.create_log_csv import create_log_csv
from utils.creation.create_xes_from_csv import xes_from_csv

from utils.converter.markov.create_markov_log_2 import get_sub_trace


def create_activities(l, k):
    acts = []
    size_v = len(l)

    if size_v <= k:
        acts.append(str(l))
    else:    
        s = 0
        e = k

        for i in range(size_v - k + 1):
            acts.append(str(l[s + i : k + i]))
    
    acts = acts

    return acts
    

def to_markov_log(log, k):
    variants = variants_filter.get_variants(log)
    markov_log = []

    for v in variants:
        l = variants[v][0]._list
        l = [a['concept:name'] for a in l]
        freq = len(variants[v])

        markov_log.append((create_activities(l, k), freq))

    return markov_log


def to_markov_log_2(log, k):
    variants = variants_filter.get_variants(log)
    markov_log = []

    for v in variants:
        new_variant = []
        l = variants[v][0]._list
        l = [act['concept:name'] for act in l]
        len_var = len(l)
        freq = len(variants[v])

        start = -(k-1)
        end = 1
        sw = get_sub_trace(l, start, end)
        new_variant.append(str(sw))
        
        for i in range(len_var + k - 2):
            sx = sw
            start += 1
            end += 1
            sw = get_sub_trace(l, start, end)
            new_variant.append(str(sw))

        markov_log.append((new_variant.copy(), freq))

    return markov_log


def get_markov_log(log, k):
    markov_log = to_markov_log_2(log, k)

    create_log_csv('temp.csv', '\t', markov_log)
    xes_from_csv('temp.csv', '\t', 'temp.xes')
    
    log_markov = xes_importer.apply('temp.xes')

    return log_markov


if __name__ == '__main__':
    # log_path = 'xes_files/test/test_markov.xes'
    log_path = 'xes_files/3/' + \
               '3a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.xes'
    log = xes_importer.apply(log_path)

    ml = to_markov_log(log, k=3)

    filepath = 'test.csv'
    create_log_csv(filepath, '\t', ml)
    xes_from_csv(filepath, '\t', 'test.xes')




    print('test')