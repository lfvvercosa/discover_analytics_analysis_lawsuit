from utils.creation.create_log_csv import create_log_csv
from utils.creation.create_xes_from_csv import xes_from_csv
from pm4py.objects.log.importer.xes import importer as xes_importer


if __name__ == '__main__':
    filepath = 'xes_files/test/test_markov2.csv'
    separator="\t"
    output_path = 'xes_files/test/test_markov2.xes'
    
    desirable_traces = [
        (['aa','bb','d,d'],5),
        (['a,a','cc','d,d'],1),
        (['bb','e"e','dd'],3),
        (['bb','cc','dd'],2),
        (['bb','dd','dd','aa'],2),
        (['aa','bb'],2)
    ]
    # desirable_traces = [
    #     (['a'],1),
    #     (['a','b','d','e'],1),
    #     (['a','b','c','d'],1),
    #     (['a','c','b','e'],1),
    #     (['a','f','g','h'],1),
    #     (['a','b','i','b','c','d'],1),
    # ]

    create_log_csv(filepath, separator, desirable_traces)
    xes_from_csv(filepath, separator, output_path)

    log = xes_importer.apply(output_path)

    print(log)
