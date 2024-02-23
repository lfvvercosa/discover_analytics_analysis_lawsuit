from experiments.variant_analysis.approaches.core.metrics.\
     ics_fitness.ICSFitnessConnector import ICSFitnessConnector


jar_path = 'temp/actitrac/ics.jar'
log_path = 'xes_files/variant_analysis/exp7/size10/low_complexity/0/tree0.xes'

ics_connec = ICSFitnessConnector(jar_path)

print(ics_connec.run_ics_fitness(log_path))