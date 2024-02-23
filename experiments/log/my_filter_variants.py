import random
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter


def random_filter_variants(log, percent):
    variants = variants_filter.get_variants(log)
    random_keys = list(variants.keys())
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



if __name__ == '__main__':

    # file_path = 'simulations/flower_model/source_files/logs/' + \
    #         'airline_log.xes'
    file_path = 'xes_files/real_processes/set_for_simulations/4/' + \
                'Hospital Billing - Event Log.xes.gz'
    log = xes_importer.apply(file_path)
    filtered_log = random_filter_variants(log, 0.8)

    print('test')