import unittest

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics

from features.LogFeatures import convert_log_to_string

class TestConvertLog(unittest.TestCase):

    # def test1(self):
    #     log_path = 'xes_files/test/test_log_mapping.xes'
    #     log = xes_importer.apply(log_path)
    #     new_log = convert_log_to_letters(log)

    
    def test2(self):
        log_path = 'xes_files/2/ElectronicInvoicingENG.xes.gz'
        log = xes_importer.apply(log_path)
        new_log = convert_log_to_string(log)

        self.assertTrue('abbccddffgghhiie' in new_log)


if __name__ == '__main__':
    unittest.main()
