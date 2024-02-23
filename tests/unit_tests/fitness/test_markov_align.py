from pm4py.objects.log.importer.xes import importer as xes_importer
from experiments.models.get_markov import get_markov_model
from features.fitness_feat import alignments_markov_log
from utils.converter.markov.to_markov_log import get_markov_log
from experiments.models.get_markov import get_markov_log, \
                                          get_markov_model


if __name__ == '__main__':
    k_markov = 2

    path_log = 'xes_files/1/4a_VARA_DE_SUCESSOES_E_REGISTROS_PUBLICOS_DA_CAPITAL_-_TJPE.xes'
    log = xes_importer.apply(path_log)
    # log_markov = get_markov_log(log, k_markov)

    print('')    

    fil = '4a_VARA_DE_SUCESSOES_E_REGISTROS_PUBLICOS_DA_CAPITAL_-_TJPE.xes'
    alg = 'ETM'

    Gm_model = get_markov_model(fil, alg, k_markov)
    
    v = alignments_markov_log(Gm_model, log, k_markov)

    # log = xes_importer.apply('test.xes')
    # f = '3a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.xes'
    # G_model = get_markov_model(f=f, alg='ETM', k=3)

    print('avg fitness alignment: ' + str(v))