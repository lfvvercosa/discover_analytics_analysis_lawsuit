import random
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils.global_var import DEBUG


def most_frequent_filter_variants(log, percent):
    variants = variants_filter.get_variants(log)
    variants_count = case_statistics.get_variant_statistics(log)
    most_frequent_keys = [v['variant'] for v in variants_count]
    n = (1 - percent) * len(most_frequent_keys)
    c = 0

    while c < n:
        c += 1
        k = most_frequent_keys.pop(0)
        del(variants[k])

    return variants_filter.apply(log, variants) 


def random_filter_variants(log, percent, seed=None):
    variants = variants_filter.get_variants(log)
    random_keys = list(variants.keys())
    
    if seed:
        random.seed(seed)
    
    random.shuffle(random_keys)
    n = (1 - percent) * len(random_keys)
    c = 0

    while c < n:
        c += 1
        k = random_keys.pop(0)
        del(variants[k])

    return variants_filter.apply(log, variants) 


def most_frequent_and_random_filter(log, percent_freq, percent_rand):
    log_filtered = variants_filter.\
            filter_log_variants_percentage(
                    log, 
                    percentage=percent_freq)
    
    return random_filter_variants(log_filtered, percent_rand)


def filter_log(f, log, freq, rand):
    if f in complex_list:
        filt_freq = 0.6
        filt_rand = 0.6
    elif f in simple_list:
        filt_freq = 0.9
        filt_rand = 0.9
    else:
        filt_freq = freq[random.randrange(0, len(freq))]
        filt_rand = rand[random.randrange(0, len(rand))]
    
    return (most_frequent_and_random_filter(log,
                                           filt_freq,
                                           filt_rand),
            filt_freq,
            filt_rand
           )


if __name__ == '__main__':

    complex_list = [
        'BPI_Challenge_2012.xes.gz',
        'BPI_Challenge_2019_log_IEEE.xes.gz',
        'BPI_Challenge_2014_Inspection.xes.gz',
        '4_a_VARA_CIVEL_DE_ARACAJU_-_TJSE.xes',
        'BPI Challenge 2017.xes.gz',
        'BPI_Challenge_2011_Hospital_log.xes.gz',
    ]

    simple_list = [
        'BPI_Challenge_2014_Reference alignment.xes.gz',
        'activitylog_uci_detailed_weekends.xes.gz',
        'BPI_Challenge_2014_Control summary.xes.gz',
        'BPI_Challenge_2014_Department control parcels.xes.gz',
    ]

    base_path = 'xes_files/' 
    log_name = '3a_VARA_DE_EXECUCOES_FISCAIS_DA_COMARCA_DE_FORTALEZA_-_TJCE.xes'

    freq = [0.4, 0.5, 0.6]
    rand = [0.4, 0.5, 0.6]

    log = xes_importer.apply(base_path + log_name)

    if DEBUG:
        print('total traces model before: ' + str(len(log)))

    filt_log, filt_freq, filt_rand = filter_log(log_name, log, freq, rand)

    if DEBUG:
        print('current event log: ' + str(log_name))
        print('filter freq: ' + str(filt_freq))
        print('filter rand: ' + str(filt_rand))
        print('total traces model after: ' + str(len(filt_log)))