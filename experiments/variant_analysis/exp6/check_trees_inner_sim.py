import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from pm4py import convert_to_dataframe
from leven import levenshtein       
from os.path import isfile, join, listdir


def mean_distance(traces):
    dist = 0

    for i in range(len(traces)-1):
        for j in range(i+1,len(traces)):
            dist += levenshtein(traces[i], traces[j])

    dist /= (len(traces) * (len(traces) - 1))/2


    return round(dist,3)


if __name__ == "__main__":
    path = 'xes_files/test_variants/exp6/process1/'
    all_files = [f for f in listdir(path) if isfile(join(path, f))]

    # for tree in all_files


    log = xes_importer.apply(path)
    df = convert_to_dataframe(log)

    split_join = SplitJoinDF(df)
    traces = split_join.split_df()

    print()
    print('mean distance: ' + str(mean_distance(traces)))
    print()

    print('done!')



    
