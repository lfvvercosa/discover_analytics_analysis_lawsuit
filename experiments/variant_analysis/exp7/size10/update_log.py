import pm4py


if __name__ == '__main__':
    log_size = 'size10/'
    log_complexity = [
       'low_complexity',
       'medium_complexity',
       'high_complexity', 
    ]
    log_total = 10
    metrics = {
        'ARI':{},
        'Fitness':{},
        'Complexity':{},
    }

    for log_complex in log_complexity:
        for i in range(log_total):
            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'
            log = pm4py.read_xes(log_path)
            log._attributes['concept:name'] = log_complex + ' - 10 Act - ' + str(i)
            pm4py.write_xes(log, log_path)

print('done!')
