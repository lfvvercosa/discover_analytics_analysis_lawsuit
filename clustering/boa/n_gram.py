from bisect import bisect_left
import pandas as pd
from nltk.util import ngrams
pd.options.mode.chained_assignment = None  # default='warn'

def binary_search(a, x, lo=0, hi=None):
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi)                  # find insertion position
    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end


def create_binary_array(mov_code, act_list, size):
    bin_arr = [0] * size
    pos = binary_search(act_list, mov_code, 0, size)

    if pos != -1:
        bin_arr[pos] = 1

    return bin_arr


def create_1_gram(df_mov, id, act):
    # print('### creating 1-gram...')

    act_list = df_mov.drop_duplicates(subset=[act])[act].to_list()
    act_list.sort()
    size = len(act_list)

    # print('act_list:')
    # print(act_list)

    df_gram = df_mov[[id, act]]

    df_gram['n_gram'] = df_gram.apply(lambda df: create_binary_array(
                            df[act],
                            act_list,
                            size
                            ), axis=1
                        )

    df_gram[act_list] = pd.DataFrame(df_gram['n_gram'].tolist(), 
                                    index=df_gram.index)

    df_gram = df_gram[[id] + act_list]
    df_gram = df_gram.groupby([id], as_index=True).sum()


    return df_gram


def get_all_feats(df, id, act, n):
    df_traces = df.groupby(id).agg(trace=(act,list))
    df_traces = df_traces[df_traces['trace'].map(len) >= n]
    df_traces['n_gram'] = \
        df_traces['trace'].apply(lambda x: list(ngrams(x, n=n)))
    df_traces = df_traces.drop(columns=['trace'])

    flatten = [item for sublist in df_traces['n_gram'] for item in sublist]

    df_traces = df_traces.explode('n_gram')

    return df_traces,list(set(flatten))


def create_n_gram(df, id, act, n):
    if n == 1:
        return create_1_gram(df, id, act)

    # print('### creating ' + str(n) + '-gram...')
    df_gram, act_list = get_all_feats(df, id, act, n)
    act_list.sort()
    size = len(act_list)

    df_gram['result'] = df_gram.apply(lambda df_gram: create_binary_array(
                            df_gram['n_gram'],
                            act_list,
                            size
                            ), axis=1
                        )

    df_gram[act_list] = pd.DataFrame(df_gram['result'].tolist(), 
                                     index=df_gram.index)

    df_gram = df_gram.groupby(id).sum()


    return df_gram