from pm4py.objects.log.importer.xes import importer as xes_importer

from experiments.variant_analysis.approaches.core.context_aware_clust.\
     ContextAwareClust import ContextAwareClustering


log_path = 'xes_files/test_variants/exp2/p1_v2v5.xes'
log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

context_aware_clust = ContextAwareClustering(log)

# print(context_aware_clust.context)

# print(context_aware_clust.levenshtein_context('AZBF','AZC'))
# print("The function was called " + \
#       str(context_aware_clust.leven_cont_deb.calls) + " times!")

print(context_aware_clust.levenshtein_context('ABDCEF','ACBEDF'))
print("The function was called " + \
      str(context_aware_clust.levenshtein_context_worker.calls) + " times!")

print(context_aware_clust.levenshtein_context('ABDCEF','ABDF'))
print("The function was called " + \
      str(context_aware_clust.levenshtein_context_worker.calls) + " times!")

print(context_aware_clust.levenshtein_context('ABDBDBDF','ABDF'))
print("The function was called " + \
      str(context_aware_clust.levenshtein_context_worker.calls) + " times!")

print(context_aware_clust.levenshtein_context('ABDBDBDF','ACBEDF'))
print("The function was called " + \
      str(context_aware_clust.levenshtein_context_worker.calls) + " times!")