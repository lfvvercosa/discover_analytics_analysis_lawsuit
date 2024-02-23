from pm4py.objects.log.importer.xes import importer as xes_importer


log_path = 'xes_files/1/activitylog_uci_detailed_labour.xes.gz'
log = xes_importer.apply(log_path)

print(log)