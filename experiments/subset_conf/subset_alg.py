import pm4py
import pandas as pd
import random
import time
from pm4py.algo.filtering.log.variants import variants_filter
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
import glob
import csv
import sys
import signal

def signal_handler(signum, frame):
    raise Exception("Timed out!")

#filtros 50, 25, 10
def filter_variants_frequency(log, percent):
    variants = variants_filter.get_variants(log)
    variants_count = case_statistics.get_variant_statistics(log)
    most_frequent_keys = [v['variant'] for v in variants_count]
    n = (1 - percent) * len(most_frequent_keys)
    c = 0

    while c < n:
        c += 1
        k = most_frequent_keys.pop()

        if len(variants) > 1:
            del(variants[k])
        else:
            break
    
    return variants_filter.apply(log, variants) 


def random_filter_variants(log, percent):
    variants = variants_filter.get_variants(log)
    random_keys = list(variants.keys())
    random.shuffle(random_keys)
    n = (1 - percent) * len(random_keys)
    c = 0

    while c < n:
        c += 1
        k = random_keys.pop(0)

        if len(variants) > 1:
            del(variants[k])
        else:
            break

    return variants_filter.apply(log, variants) 


def alinhamentos(algoritmo):

    signal.signal(signal.SIGALRM, signal_handler)
    timeout = 7200
    ETM = []
    HEU_MINER = []
    IMd = []
    IMf = []
    petris=[]
    Testes = []
    

    if(algoritmo =='ETM'):
        for arquivo in glob.glob(r'petri_nets/ETM/*.pnml'):
            ETM.append(arquivo)
        petris = ETM

    elif(algoritmo=='HEU_MINER'):
        for arquivo1 in glob.glob(r'petri_nets/HEU_MINER/*.pnml'):
            HEU_MINER.append(arquivo1)
        petris = HEU_MINER

    elif(algoritmo == 'IMd'):
        for arquivo2 in glob.glob(r'petri_nets/IMd/*.pnml'):
            IMd.append(arquivo2)
        petris = IMd

    elif(algoritmo == 'IMf'):
        for arquivo3 in glob.glob(r'petri_nets/IMf/*.pnml'):
            IMf.append(arquivo3)
        petris = IMf
        
    elif(algoritmo == 'Exemplo'):
        for arquivo4 in glob.glob(r'petri_nets/Exemplo/*.pnml'):
            Testes.append(arquivo4)
        petris = Testes
    else:
        print('Algoritmo invalido')
        return 0
    print('esta é as petri nets')
    print(petris) 
    print('  ') 

    for petri in petris:
        net, im, fm = pnml_importer.apply(petri)
        q = 1
        s = petri.split("/")[-1]
        arq = s[:-5]
        log = None
        print(arq)
        while q < 6:
            arq1 = 'xes_files/' + str(q) + '/' + arq + '.xes'
            arq2 = 'xes_files/' + str(q) + '/' + arq + '.xes.gz'

            if os.path.isfile(arq1):
                log = xes_importer.apply(arq1)
                print('log encontrado xes')
                break
            elif os.path.isfile(arq2):
                log = xes_importer.apply(arq2)
                print('log encontrado gz')
                break
            else:
                q += 1
        
        #filtros
        log_freq_50 = filter_variants_frequency(log, 0.5)
        log_rand_50 = random_filter_variants(log, 0.5)
        log_freq_25 = filter_variants_frequency(log, 0.25)
        log_rand_25 = random_filter_variants(log, 0.25)
        #log_freq_10 = filter_variants_frequency(log, 0.1)
        #log_rand_10 = random_filter_variants(log, 0.1)

        '''
        #alinhamento_fitness_log_inteiro
        tempo_inicial = time.time()
        fitness = replay_fitness_evaluator.apply(log, net, im, fm, 
                                            variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
        tempo_final = time.time()
        tempo_res_total_fit = tempo_final - tempo_inicial
        print(f'Fitness_Log_inteiro = {fitness["log_fitness"]}')
        print(f'TEMPO_FINAL_FITNESS = {tempo_res_total_fit}')

        #alinhamento_precision_log_inteiro
        tempo_inicial = time.time()
        precision = precision_evaluator.apply(log, net, im, fm, 
                                        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        tempo_final = time.time()
        tempo_res_total_prec = tempo_final - tempo_inicial
        print(f'Precision_Log_inteiro = {precision}')
        print(f'TEMPO_FINAL_PRECISION = {tempo_res_total_prec}')
        '''

       #alinhamento_fitness_log_50_freq
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            fitness_freq_50 = replay_fitness_evaluator.apply(log_freq_50, net, im, fm, 
                                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
            tempo_final = time.time()
            tempo_50_freq_fit = tempo_final - tempo_inicial
            print(f'Fitness_Log_50_freq = {fitness_freq_50}')
            print(f'TEMPO_50_FREQ_FIT = {tempo_50_freq_fit}')
        except Exception:
            print("Timed out!")
            fitness_freq_50 = 'Timeout'
            tempo_50_freq_fit = 'Timeout'

        #alinhamento_precision_log_50_freq
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            precision_freq_50 = precision_evaluator.apply(log_freq_50, net, im, fm, 
                                        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            tempo_final = time.time()
            tempo_50_freq_prec = tempo_final - tempo_inicial
            print(f'Precision_Log_50_freq  = {precision_freq_50}')
            print(f'TEMPO_50_FREQ_PREC = {tempo_50_freq_prec}')
        
        except Exception:
            print("Timed out!")
            precision_freq_50 = 'Timeout'
            tempo_50_freq_prec = 'Timeout'

        
        
        #alinhamento_fitness_log_50_rand
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            fitness_rand_50 = replay_fitness_evaluator.apply(log_rand_50, net, im, fm, 
                                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
            tempo_final = time.time()
            tempo_50_rand_fit = tempo_final - tempo_inicial
            print(f'Fitness_Log_50_rand= {fitness_rand_50}')
            print(f'TEMPO_50_RAND_FIT = {tempo_50_rand_fit}')
        except Exception:
            print("Timed out!")
            fitness_rand_50 = 'Timeout'
            tempo_50_rand_fit = 'Timeout'


        #alinhamento_precision_log_50_rand
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            precision_rand_50 = precision_evaluator.apply(log_rand_50, net, im, fm, 
                                        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            tempo_final = time.time()
            tempo_50_rand_prec = tempo_final - tempo_inicial
            print(f'Precision_Log_50_rand= {precision_rand_50}')
            print(f'TEMPO_50_RAND_PREC= {tempo_50_rand_prec}')
        except Exception:
            print("Timed out!")
            precision_rand_50 = 'Timeout'
            tempo_50_rand_prec = 'Timeout'
        

        #alinhamento_fitness_log_25_freq
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            fitness_freq_25 = replay_fitness_evaluator.apply(log_freq_25, net, im, fm, 
                                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
            tempo_final = time.time()
            tempo_25_freq_fit = tempo_final - tempo_inicial
            print(f'Fitness_log_25_freq = {fitness_freq_25}')
            print(f'TEMPO_25_FREQ_FIT = {tempo_25_freq_fit}')
        except Exception:
            print("Timed out!")
            fitness_freq_25 = 'Timeout'
            tempo_25_freq_fit = 'Timeout'
        
        #alinhamento_precision_log_25_freq
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            precision_freq_25 = precision_evaluator.apply(log_freq_25, net, im, fm, 
                                        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            tempo_final = time.time()
            tempo_25_freq_prec = tempo_final - tempo_inicial
            print(f'Precision_Log_25_freq  = {precision_freq_25}')
            print(f'TEMPO_25_FREQ_PREC = {tempo_25_freq_prec}')
        except Exception:
            print("Timed out!")
            precision_freq_25 = 'Timeout'
            tempo_25_freq_prec = 'Timeout'

        #alinhamento_fitness_log_25_rand
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            fitness_rand_25 = replay_fitness_evaluator.apply(log_rand_25, net, im, fm, 
                                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
            tempo_final = time.time()
            tempo_25_rand_fit = tempo_final - tempo_inicial
            print(f'Fitness_log_25_rand = {fitness_rand_25}')
            print(f'TEMPO_25_RAND_FIT = {tempo_25_rand_fit}')
        except Exception:
            print("Timed out!")
            fitness_rand_25 = 'Timeout'
            tempo_25_rand_fit = 'Timeout'


        #alinhamento_precision_log_25_rand
        signal.alarm(timeout)
        try:
            tempo_inicial = time.time()
            precision_rand_25 = precision_evaluator.apply(log_rand_25, net, im, fm, 
                                        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            tempo_final = time.time()
            tempo_25_rand_prec = tempo_final - tempo_inicial
            print(f'Precision_Log_25_rand = {precision_rand_25}')
            print(f'TEMPO_25_RAND_PREC = {tempo_25_rand_prec}')
        except Exception:
            print("Timed out!")
            precision_rand_25 = 'Timeout'
            tempo_25_rand_prec = 'Timeout'


        #ADICIONAR NO ARQUIVO CSV OS VALORES
        documento = None
        if(algoritmo =='ETM'):
            documento = open('experiments/subset_selection_conformance/ETM.csv', 'a')
        elif(algoritmo == 'HEU_MINER'):
            documento = open('experiments/subset_selection_conformance/HEU_MINER.csv', 'a')
        elif(algoritmo == 'IMd'):
            documento = open('experiments/subset_selection_conformance/IMd.csv', 'a')
        elif(algoritmo == 'IMf'):
            documento = open('experiments/subset_selection_conformance/IMf.csv', 'a')
        elif(algoritmo == 'Exemplo'):
            documento = open('experiments/subset_selection_conformance/Exemplo.csv', 'a')
        else:
            break

        escrever = csv.writer(documento)
        escrever.writerow((arq,fitness_freq_50,tempo_50_freq_fit,
                                precision_freq_50,tempo_50_freq_prec,fitness_rand_50,
                                tempo_50_rand_fit,precision_rand_50,tempo_50_rand_prec,
                                fitness_freq_25,tempo_25_freq_fit,precision_freq_25,
                                tempo_25_freq_prec,fitness_rand_25,tempo_25_rand_fit,
                                precision_rand_25,tempo_25_rand_prec))
        documento.close()

        print('    ')
           

if __name__ == '__main__':

    print(sys.argv[1])
    alinhamentos(sys.argv[1])
    #alinhamentos('IMd')

    

     
    