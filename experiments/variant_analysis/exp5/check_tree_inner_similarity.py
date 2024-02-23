import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from pm4py import convert_to_dataframe
from leven import levenshtein       


def mean_distance(traces):
    dist = 0

    for i in range(len(traces)-1):
        for j in range(i+1,len(traces)):
            dist += levenshtein(traces[i], traces[j])

    dist /= (len(traces) * (len(traces) - 1))/2


    return round(dist,3)



if __name__ == "__main__":
    path = 'xes_files/test_variants/exp5/exp5_1.xes'
    log = xes_importer.apply(path)
    df = convert_to_dataframe(log)

    split_join = SplitJoinDF(df)
    traces = split_join.split_df()

    print()
    print('mean distance: ' + str(mean_distance(traces)))
    print()

    print('done!')



    
