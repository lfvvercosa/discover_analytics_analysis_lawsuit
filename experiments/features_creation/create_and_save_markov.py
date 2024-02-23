import os
import pickle
import multiprocessing
import time
import sys
import json
import boto3
from os import listdir
from os.path import isfile, join, exists
from pathlib import Path
from utils.converter import tran_sys_to_nx_graph
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_first_n_paths_from_vertex_pair, is_path_possible
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import \
    convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.markov.markov_utils import are_markov_paths_possible_2
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.global_var import DEBUG
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import reachability_graph


def remove_suffix(f, alg):
    if '.xes.gz' in f:
        f = f.replace('.xes.gz', '')
    if '.xes' in f:
        f = f.replace('.xes', '')
        
    return f


def write_to_s3(bucket, filename, file_content):
    s3 = boto3.resource('s3')
    object = s3.Object(bucket, 
                       filename)
    object.put(Body=file_content)


def read_from_s3(bucket, filename):
    try:
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, 
                        filename)
        
        return eval(object.get()['Body'].read())
    except:
        return None


def create_markov_model_full(net, im, k, f, ret_dict):
    
    print('### transition system...')

    start = time.time()
    ts = reachability_graph.construct_reachability_graph(net, im)
    end = time.time()
    ts_time = round(end - start, 4)

    # DEBUG
    G = tran_sys_to_nx_graph.convert(ts)
    n = 500
    paths_G = find_first_n_paths_from_vertex_pair(G, 
                                                  v1=None, 
                                                  v2=None,
                                                  n=n)
    
    print('### NFA to DFA...')

    start = time.time()
    Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
    end = time.time()
    gd_time = round(end - start, 4)

    print('### DFA Reduction...')

    start = time.time()
    Gr = reduceDFA(Gd, include_empty_state=False)
    end = time.time()
    gr_time = round(end - start, 4)

    print('### Markov k = ' + str(k) + '...')

    start = time.time()
    Gm = create_mk_abstraction_dfa_2(Gr, k=k)
    end = time.time()
    gm_time = round(end - start, 4)

    total_time = round(ts_time + gd_time + gr_time + gm_time,4)

    if not are_markov_paths_possible_2(Gm, paths_G, k):
        ret_dict['Exception'] = 'path not possible for log: ' + f
    else:
        ret_dict['Markov'] = Gm
        ret_dict['tran_sys_time'] = ts_time
        ret_dict['dfa_time'] = gd_time
        ret_dict['reduc_time'] = gr_time
        ret_dict['markov_time'] = gm_time
        ret_dict['total_time'] = total_time


def create_func_timeout(func, args, timeout):
    manager = multiprocessing.Manager()
    ret_dict = manager.dict()
    args.append(ret_dict)

    p = multiprocessing.Process(
                                target=func,
                                args=args
                               )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
    
        if DEBUG:
            print('### Timeout...')
            ret_dict['Timeout'] = timeout

    p.join()

    return ret_dict


def get_percent_complete(myalgs, log_folders, alg, fol):
    total = 10 * len(log_folders) * len(myalgs)
    current = 10 * (log_folders.index(fol) + 1) * \
              (myalgs.index(alg) + 1)
    
    return str(round(current/total, 4))


if __name__ == '__main__':
    number_skipped = 0
    paths_not_possible = []
    base_path = 'xes_files/'

    instance_id_aws = list(sys.argv)[1]
    # instance_id_aws = -1

    folders = eval(list(sys.argv)[2])
    # folders = ['1/', '2/', '3/', '4/', '5/']

    myalgs = eval(list(sys.argv)[3])
    # myalgs = ['IMf', 'IMd']

    markov_k = eval(list(sys.argv)[4])
    # markov_k = 2
    
    s3_bucket = 'luiz-doutorado-projetos2'
    s3_dir_res = 'testes_markov/resultados_markov/k_'

    petri_nets_path = [
        ('petri_nets/IMf/', 'IMf'),
        ('petri_nets/HEU_MINER/', 'HEU_MINER'),
        ('petri_nets/IMd/', 'IMd'),
        ('petri_nets/ETM/', 'ETM'),
    ]
    petri_nets_path = [p for p in petri_nets_path if p[1] in myalgs]

    res = 'Markov'
    results = []

    output_path = 'experiments/features_creation/markov/' + \
                  'from_dfa_reduced/new_k_' + str(markov_k) + '/'
    timeout_base_path = 'experiments/features_creation/markov/' + \
                        'from_dfa_reduced/new_k_' + str(markov_k) + '_timeout/'
    
    one_minute = 60
    timeout = 120 * one_minute

    for alg in petri_nets_path:
        for fol in folders:
            
            filename = s3_dir_res + str(markov_k) + '/' + alg[1] + '/' + \
                           'res' + fol[:-1] + '.json'
            temp = read_from_s3(s3_bucket, filename)

            if temp:
                results = temp

            if DEBUG:
                print('results is: ')
                print(results)
                print()
                print('### Current folder: ' + str(fol))
                print('### Current algorithm: ' + str(alg[1]))
                print('### Markov k = ' + str(markov_k))
                print('### ec2 instance: ' + str(instance_id_aws))
                print('### complete: ' + \
                    get_percent_complete(myalgs, folders, alg[1], fol))

            my_path = base_path + fol
            files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

            for f in files:

                if DEBUG:
                    print('### Current file: ' + str(f))

                saving_path = output_path + alg[1] + '/'
                timeout_path = timeout_base_path + alg[1] + '/'

                Path(saving_path).mkdir(parents=True, exist_ok=True)
                Path(timeout_path).mkdir(parents=True, exist_ok=True)

                saving_path += f + '.txt'
                timeout_path += f + '.txt'

                if exists(saving_path):
                    if DEBUG:
                        print('### Skipping: file already exists!')

                    continue
                
                if exists(timeout_path):
                    if DEBUG:
                        print('### Skipping: timeout file!')

                    continue

                try:
                    net, im, fm = \
                        pnml_importer.apply(os.path.join(alg[0],
                                        remove_suffix(f,alg[1]) + \
                                        '.pnml')
                                        )
                except Exception as e:
                    print(e)
                    number_skipped += 1
                    continue

                # path_gd = 'experiments/features_creation/dfa_reduced/' + \
                # alg[1] + '/' + f + '.txt'
                args = [net, im, markov_k, f]
                ret_dict = create_func_timeout(create_markov_model_full, 
                                               args, 
                                               timeout)

                if ret_dict:
                    if res in ret_dict:
                        Gm = ret_dict[res]
                        results.append(
                            {
                                'event_log':f,
                                'algorithm':alg[1],
                                'tran_sys_time':ret_dict['tran_sys_time'],
                                'dfa_time':ret_dict['dfa_time'],
                                'reduc_time':ret_dict['reduc_time'],
                                'markov_time':ret_dict['markov_time'],
                                'total_time':ret_dict['total_time'],
                            }
                        )

                    else:
                        if 'Exception' in ret_dict:
                            raise Exception(ret_dict['Exception'])

                        if 'Timeout' in ret_dict:
                            pickle.dump(ret_dict, open(timeout_path, 'wb'))

                        number_skipped += 1
                        continue
                else:
                    number_skipped += 1
                    continue

                if DEBUG:
                    print('### Saving ' + res + ' model...')

                pickle.dump(Gm, open(saving_path, 'wb'))

                
                file_content = json.dumps(results)
                write_to_s3(s3_bucket, filename, file_content)

                print('saved!')
    
    print('skipped: ' + str(number_skipped))
    print('done!')

    # ec2 = boto3.resource('ec2')
    # instance = ec2.Instance(instance_id_aws)
    # print(instance.terminate())

            

