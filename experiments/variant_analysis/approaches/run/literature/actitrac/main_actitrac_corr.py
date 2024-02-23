from pathlib import Path
from experiments.variant_analysis.approaches.run.\
     literature.actitrac.ActiTraCConnector import ActiTracConnector

from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd
import matplotlib.pyplot as plt


def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    jar_path = 'temp/actitrac/actitrac.jar'
    actitrac = ActiTracConnector(jar_path)

    number_of_clusters = [1,2,3,4,5,6,7,8,9,10]
    target_fit = 0.9
    is_greedy = True
    dist_greed = 0.25
    min_clus_size = 0.1
    log_size = 'size10/'
    log_complex = 'high_complexity/'
    i = 3
    name_approach = 'ActiTraC_ics'

    log_path = 'xes_files/variant_analysis/exp7/'
    log_path += log_size + log_complex  + str(i) + '/log.xes'

    saving_path = 'temp/actitrac/'
    saving_path += log_size + log_complex + str(i) + '/'

    create_path_dir(saving_path)
    fit_corr = []
    ARI_corr = []
    complx_corr = []


    for n_clusters in number_of_clusters:
        ARI, fit, complx = actitrac.run_actitrac(n_clusters,
                                                is_greedy,
                                                dist_greed,
                                                target_fit,
                                                min_clus_size,
                                                log_path,
                                                saving_path)

        print('## FITNESS: ' + str(fit))

        fit_corr.append(fit)
        ARI_corr.append(ARI)
        complx_corr.append(complx)

    # y = list(range(10))
    plt.plot(number_of_clusters, fit_corr)
    plt.show()

    print()