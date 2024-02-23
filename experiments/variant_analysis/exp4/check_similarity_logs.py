from experiments.variant_analysis.exp5.creation.\
     similar_tree.SimilarTree import SimilarTree
from pm4py.objects.log.importer.xes import importer as xes_importer
        


if __name__ == "__main__":
    similar_tree = SimilarTree()

    log1_path = 'xes_files/test_variants/exp4/exp4_p1_v1.xes'
    log2_path = 'xes_files/test_variants/exp4/exp4_p1_v2.xes'
    log3_path = 'xes_files/test_variants/exp4/exp4_p1_v3.xes'
    log4_path = 'xes_files/test_variants/exp4/exp4_p2_v1.xes'
    log5_path = 'xes_files/test_variants/exp4/exp4_p2_v2.xes'

    log1 = xes_importer.apply(log1_path, 
                              variant = xes_importer.Variants.LINE_BY_LINE)
    log2 = xes_importer.apply(log2_path, 
                              variant = xes_importer.Variants.LINE_BY_LINE)
    log3 = xes_importer.apply(log3_path, 
                              variant = xes_importer.Variants.LINE_BY_LINE)
    log4 = xes_importer.apply(log4_path, 
                              variant = xes_importer.Variants.LINE_BY_LINE)
    log5 = xes_importer.apply(log5_path, 
                              variant = xes_importer.Variants.LINE_BY_LINE)

    trees_simil = similar_tree.cross_align_similarity(log4, log5)
    print(trees_simil)