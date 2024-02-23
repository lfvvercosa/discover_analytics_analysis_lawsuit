import pickle


my_file = 'dataset_creation/features_creation/markov/k_2/IMf/' + \
          'Production_Data.xes.gz.txt'

Gm = pickle.load(open(my_file, 'rb'))

print('test')