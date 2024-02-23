import math

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.clustering.create_n_gram import create_n_gram


LEFT = 0    
RIGHT = 1


def insert_scores_left_given_right(freq_3gram, probs):
    ctx = symbol_contexts(freq_3gram)
    I_score = {}

    count_left_given_right, norm_left_given_right, \
    norm_count_left = neighbor(ctx, RIGHT)
        
    for a in norm_count_left:
        I_score[a] = {}
        for b in norm_count_left[a]:
            if norm_count_left[a][b] != 0:
                div = norm_count_left[a][b]/(probs[a]*probs[b])
                I_score[a][b] = round(math.log2(div),4)
    

    return I_score


def insert_scores_right_given_left(freq_3gram, probs):
    ctx = symbol_contexts(freq_3gram)
    I_score = {}

    count_right_given_left, norm_right_given_left, \
    norm_count_right = neighbor(ctx, LEFT)
        
    for a in norm_count_right:
        I_score[a] = {}
        for b in norm_count_right[a]:
            if norm_count_right[a][b] != 0:
                div = norm_count_right[a][b]/(probs[a]*probs[b])
                I_score[a][b] = round(math.log2(div),4)
    

    return I_score


def right_given_left(ctx):
    return neighbor(ctx, 0)


def left_given_right(ctx):
    return neighbor(ctx, 1)


def neighbor(ctx, i):
    count_neighbor_i = {}
    norm_i = {}
    norm_count_neighbor_i = {}
    
    for a in ctx:
        count_neighbor_i[a] = {}
        norm_i[a] = 0
        for gram_3 in ctx[a]:
            if gram_3[i] not in count_neighbor_i[a]:
                count_neighbor_i[a][gram_3[i]] = 0

            count_neighbor_i[a][gram_3[i]] += ctx[a][gram_3]
            norm_i[a] += ctx[a][gram_3]        
    
    for a in count_neighbor_i:
        norm_count_neighbor_i[a] = {}
        for b in count_neighbor_i[a]:
            norm_count_neighbor_i[a][b] = count_neighbor_i[a][b]/ \
                                          norm_i[a]


    return count_neighbor_i, norm_i, norm_count_neighbor_i


# def prob_occur(norm):
#     probs = {}
#     total = sum(norm.values()) 

#     for a in norm:
#         probs[a] = round(norm[a]/total,4)

    
#     return probs


def subst_scores(freq_3gram):
    ctx = symbol_contexts(freq_3gram)
    co_occur = {}
    S_score = {}
    norm_co_occur = 0

    for a in ctx:
        co_occur[a] = {}
        for b in ctx:
            set_a = set(ctx[a].keys())
            set_b = set(ctx[b].keys())
            ctx_a_b = set_a.intersection(set_b)
            co_occur[a][b] = co_occurrence(ctx, ctx_a_b, a, b)
            norm_co_occur += co_occur[a][b]       
    
    M = get_M(ctx, co_occur, norm_co_occur)
    probs = prob_symbols(M)

    count = 0

    for a in probs:
        count += probs[a]

    Expec = get_expec_val(probs)

    for a in M:
        S_score[a] = {}

        for b in M:
            if M[a][b] != 0:
                S_score[a][b] = round(math.log2(M[a][b]/Expec[a][b]),2)
            
    
    return S_score


def symbol_contexts(freq_3gram):
    ctx = {}

    for gram in freq_3gram:
        s = gram[1]

        if s not in ctx:
            ctx[s] = {}

        ctx[s][(gram[0],gram[2])] = freq_3gram[gram]

    
    return ctx


def co_occurrence(ctx, ctx_a_b, a, b):
    co_occur_a_b = 0

    for c in ctx_a_b:
        if a == b:
            co_occur_a_b += (ctx[a][c] * (ctx[a][c] - 1))/2
        else:
            co_occur_a_b += ctx[a][c] * ctx[b][c]


    return co_occur_a_b


def get_M(ctx, co_occur, norm_co_occur):
    M ={}

    for a in ctx:
        M[a] = {}
        for b in ctx:
            M[a][b] = round(co_occur[a][b]/norm_co_occur,4)


    return M

def prob_symbols(M):
    probs = {}

    for a in M:
        probs[a] = 0
        for b in M:
            probs[a] += M[a][b]
        
    
    return probs


def get_expec_val(probs):
    expec_val = {}

    for a in probs:
        expec_val[a] = {}

        for b in probs:
            if a == b:
                expec_val[a][b] = probs[a]**2
            else:
                expec_val[a][b] = 2*probs[a]*probs[b]

    
    return expec_val


def get_symbol_probab(df):
    df_temp = df.groupby('concept:name').agg(count=('time:timestamp','count'))
    total_occur = df_temp.sum().to_dict()['count']
    occur = (df_temp/total_occur).to_dict()['count']


    return occur



log_path = 'xes_files/test_variants/exp2/p1_v4_mini.xes'
log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
df = convert_to_dataframe(log)

df_gram = create_n_gram(df, 'case:concept:name', 'concept:name', n=3)
freq_3gram = df_gram.sum().to_dict()
probs_symbols = get_symbol_probab(df)

print(subst_scores(freq_3gram))
# insert_scores_right_given_left(freq_3gram, probs_symbols)
# insert_scores_left_given_right(freq_3gram, probs_symbols)


print(df)



