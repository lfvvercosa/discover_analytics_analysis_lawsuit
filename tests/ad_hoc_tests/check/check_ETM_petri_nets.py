import os
from os import listdir
from os.path import isfile, join


def remove_suffix(f):
    if '.xes.gz' in f:
        f = f.replace('.xes.gz', '')
    if '.xes' in f:
        f = f.replace('.xes', '')
        
    return f


event_log_dir = 'xes_files/real_processes/set_for_simulations/'
folder = ['1/', '2/', '3/', '4/', '5/']

petri_net_dir = '/home/vercosa/git/graph_classification/simul_qual_metr/petri_nets/ETM/'
count = 0
my_pn = [f for f in listdir(petri_net_dir) if isfile(join(petri_net_dir, f))]

for fol in folder:
    my_path = event_log_dir + fol
    myfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    for log in myfiles:
        if os.path.exists(petri_net_dir + remove_suffix(log) + '.pnml'):
            count += 1
        else:
            print('### non-fitting:')
            for pn in my_pn:
                if log[:6] == pn[:6]:
                    print(log)

    
print('total pn found: ' + str(count))
        
