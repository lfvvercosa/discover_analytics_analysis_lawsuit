import unittest
from experiments.log_filtering import random_filter_variants
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def printLogVariants(log):
    variants = variants_filter.get_variants(log)

    for v in variants:
        print(v)


def printNumberLogVariants(log):
    variants = variants_filter.get_variants(log)

    print(len(variants))


def getNumberLogVariants(log):
    variants = variants_filter.get_variants(log)

    return len(variants)


class TestRandomFilter(unittest.TestCase):

    def test1(self):
        log_path = 'xes_files/test/test_markov.xes'
        log = xes_importer.apply(log_path)

        log_filt = random_filter_variants(log, percent=0.9)

        # printLogVariants(log_filt)
    

    def test2(self):
        log_path = 'xes_files/test/test_markov2.xes'
        out_path = 'xes_files/test/test_markov2_filt.xes'
        log = xes_importer.apply(log_path)

        print('### before filtering ###')
        printLogVariants(log)


        log_filt = random_filter_variants(log, percent=0.4)

        print('### after filtering ###')
        printLogVariants(log_filt)

        # xes_exporter.apply(log_filt, out_path)
    

    def test3(self):
        log_path = 'xes_files/1/Production_Data.xes.gz'
        out_path = 'xes_files/test/Production_Data_Filt.xes.gz'
        log = xes_importer.apply(log_path)

        print('### before filtering ###')
        printNumberLogVariants(log)


        log_filt = random_filter_variants(log, percent=0.4)

        print('### after filtering ###')
        printNumberLogVariants(log_filt)

        xes_exporter.apply(log_filt, out_path)




if __name__ == '__main__':
    unittest.main()



