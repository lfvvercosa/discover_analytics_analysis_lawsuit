import networkx as nx
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
import math
import glob
import csv

#verificar se o estado e se os estados de saida existem
def is_existing_state(G, curr_state, a):
    out_edges = list(G.out_edges(curr_state))
    for edge in out_edges:
        if edge[1] == curr_state + a:
            return True
    
    return False

#adicionar estado
def create_node(G, n):
    G.add_node(n)
    nx.set_node_attributes(G, {n:0}, 'freq')
    nx.set_node_attributes(G, {n:0}, 'seq')


#verificar se é o ultimo estado
def is_last_activity(state, activ):
    return len(state) == len(activ)



def build_automato(log):
    variants_count = case_statistics.get_variant_statistics(log)

    G = nx.DiGraph()
    G.add_node('')
    nx.set_node_attributes(G, {'':0}, 'freq')

    ##print(variants_count)
    
    for vc in variants_count:
        activ = vc['variant'].split(',')
        count = vc['count']
        prefix = ''
        curr_state = ''
    
        for a in activ:
            prefix += a
             
            if not is_existing_state(G, curr_state, a):
                create_node(G, prefix)
                G.add_edge(curr_state, prefix)
                nx.set_edge_attributes(G, {(curr_state, prefix):a}, 'act')

            G.nodes[curr_state]['freq'] += count
            G.nodes[prefix]['seq']+=1
            
            if is_last_activity(prefix, activ):
                G.nodes[prefix]['freq'] += count
            
            curr_state = prefix

    return G



def dicionario_parti(graph):
    
    leafs = [x for x in graph.nodes() if graph.out_degree(x)==0 and graph.in_degree(x)==1]
    list_leafs = sorted(leafs, key=len, reverse = True)

    nodes = list(graph.nodes)[1:]

    part = {}

    contador_part = 1

    for p in list_leafs:
        if(p in nodes):
            part['P' + str(contador_part)]= [p]
            nodes.remove(p)
            action = graph.edges[list(graph.in_edges(p))[0]]['act']
            string = p[:-len(action)]

            while(string!=''):
                if(string in nodes):
                    part['P' + str(contador_part)].append(string)
                    nodes.remove(string)
                    action = graph.edges[list(graph.in_edges(string))[0]]['act']
                    string = string[:-len(action)]

                else:
                    break
        contador_part+=1

    return part    

def somatorio_variant_entropy(dicti):
    soma = 0
    for n in dicti:
        soma += len(dicti[n]) * math.log(len(dicti[n]))

    return soma

def variant_entropy(dicti,graph):
    S = len(list(graph.nodes))-1
    return S * math.log(S) - somatorio_variant_entropy(dicti)

def variant_entropy_norm(dicti,graph):
    S = len(list(graph.nodes))-1
    return (variant_entropy(dicti,graph))/(S * math.log(S))

def seq_s_sequence_entropy(graph):
    l = list(graph.nodes)[1:]
    sum = 0
    for x in l:
        sum += graph.nodes[x]['seq']
    return sum

def sum_sequence_entropy(dicti,graph):
    soma = 0
    for n in dicti:
        for i in dicti[n]:
            soma += graph.nodes[i]['seq'] * math.log(graph.nodes[i]['seq'])

    return soma

def sequence_entropy(dicti,graph):
    return (seq_s_sequence_entropy(graph) * math.log(seq_s_sequence_entropy(graph))) - sum_sequence_entropy(dicti,graph)

def sequence_entropy_norm(dicti,graph):
    return (sequence_entropy(dicti,graph))/(seq_s_sequence_entropy(graph) * math.log(seq_s_sequence_entropy(graph)))




def tabela_entropy():
    
    logs = []
    pasta = 1
    
    while(pasta<6):
    
        for arquivo in glob.glob(r'xes_files/'+ str(pasta)+'/*'):
            logs.append(arquivo)
        pasta+=1
    
    for reg in logs:
        name = reg.split("/")[-1]
        
        log = xes_importer.apply(reg)
        G = build_automato(log)
        part = dicionario_parti(G)
        var_entropy = variant_entropy(part,G)
        var_entropy_norm = variant_entropy_norm(part,G)
        seq_entropy=sequence_entropy(part,G)
        seq_entropy_norm=sequence_entropy_norm(part,G)

        documento = open('experiments/Automato prefix/Entropy_tabel.csv', 'a')
        escrever = csv.writer(documento)
        escrever.writerow((name, var_entropy, var_entropy_norm, seq_entropy, seq_entropy_norm))
        
        documento.close()
    
    
if __name__ == '__main__':


    tabela_entropy()

    '''
    log_path = 'xes_files/5/BPI_Challenge_2013_incidents.xes.gz'

    log = xes_importer.apply(log_path)

    G = build_automato(log)
    
    dicti = {value : G.edges[value]['act'] for value in list(G.edges)}
    pos = nx.spring_layout(G)
    plt.figure(figsize = (10,10))
    nx.draw(G,pos, with_labels=1)
    nx.draw_networkx_edge_labels(
        G,pos,
        edge_labels = dicti,
        font_color='red'
    )

    plt.show
    
    part = dicionario_parti(G)

    v = variant_entropy(part,G)
    vn = variant_entropy_norm(part,G)
    se=sequence_entropy(part,G)
    sen=sequence_entropy_norm(part,G)

    print(v)
    print(vn)
    print(se)
    print(sen)

'''

