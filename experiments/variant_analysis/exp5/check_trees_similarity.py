from os import listdir
from os.path import isfile, join
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
     


def cross_align_similar(log1, log2):
    align_log1_2 = logs_alignments.apply(log1, log2)
    align_log2_1 = logs_alignments.apply(log2, log1)

    fit_log1_2 = [e['fitness'] for e in align_log1_2]
    fit_log2_1 = [e['fitness'] for e in align_log2_1]
    
    fitness = fit_log1_2 + fit_log2_1
    
    
    return sum(fitness)/len(fitness)


if __name__ == "__main__":
    all_logs = []
    simil = []
    mypath = 'xes_files/test_variants/exp5/'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    for i in range(len(onlyfiles)):
        log_path = mypath + onlyfiles[i]
        log = xes_importer.apply(log_path, 
                                 variant=xes_importer.Variants.LINE_BY_LINE)

        all_logs.append(log)

    for i in range(len(all_logs) - 1):
        for j in range(i + 1, len(all_logs)):
            if i % 5 == 0:
                if j % 50 == 0:
                    print('calc cross simil for %d, %d...',i,j)
            cross_simil = cross_align_similar(all_logs[i], all_logs[j])
            simil.append((i,j,round(cross_simil,4)))

    simil.sort(key=lambda t: t[2], reverse=True)


    with open('temp/simil.txt','w') as f:
        for s in simil:
            f.write(str(s) + '\n')

    print('done!')