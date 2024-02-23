import math

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.clustering.create_n_gram_pad import create_n_gram_pad


class SimilarityScore:
    freq_3gram = None
    probs = None
    Subst_score = None
    Indel_score_left_given_right = None
    Indel_score_right_given_left = None

    LEFT = 0
    RIGHT = 1


    def __init__(self, log):
        df = convert_to_dataframe(log)
        df_gram = create_n_gram_pad(df, 'case:concept:name', 'concept:name', n=3)
        df_gram = self.filter_undesirable_pad(df_gram)

        self.freq_3gram = df_gram.sum().to_dict()
        self.probs = self.get_symbol_probab(df)
        self.similarity_scores()


    def get_symbol_probab(self, df):
        df_temp = df.groupby('concept:name').agg(count=('time:timestamp','count'))
        total_occur = df_temp.sum().to_dict()['count']
        occur = (df_temp/total_occur).to_dict()['count']


        return occur
    

    def filter_undesirable_pad(self, df_gram):
        cols = df_gram.columns
        cols = [c for c in cols if c[1] != '-']

        df_gram = df_gram[cols]


        return df_gram


    def get_similarity_dict(self):
        context = {}

        context['substitution'] = self.Subst_score
        context['ins_right_given_left'] = self.Indel_score_right_given_left
        context['ins_left_given_right'] = self.Indel_score_left_given_right
        context['del_right_given_left'] = context['ins_right_given_left']
        context['del_left_given_right'] = context['ins_left_given_right']
        
        context = self.fill_missing(context)

        return context
    

    def fill_missing(self, context):
        # min_val = self.context_smallest_val(context)
        min_val = 0

        all_symbols = self.probs.keys()

        for k in context:
            for s in context[k]:
                for t in all_symbols:
                    if t not in context[k][s]:
                        context[k][s][t] = min_val


        return context


    def context_smallest_val(self, context):
        min_val = float('inf')
        
        for k in context:
            for s in context[k]:
                vals = context[k][s].values()
                if vals:
                    min_s = min(context[k][s].values())
                    if min_s < min_val:
                        min_val = min_s


        return min_val 

    def similarity_scores(self):
        self.Subst_score = self.subst_scores(self.freq_3gram)
        self.Indel_score_left_given_right = \
            self.insert_scores_left_given_right(self.freq_3gram, self.probs)
        self.Indel_score_right_given_left = \
            self.insert_scores_right_given_left(self.freq_3gram, self.probs)
        
    
    def insert_scores_left_given_right(self, freq_3gram, probs):
        ctx = self.symbol_contexts(freq_3gram)
        I_score = {}

        count_left_given_right, norm_left_given_right, \
        norm_count_left = self.neighbor(ctx, self.RIGHT)
            
        for a in norm_count_left:
            I_score[a] = {}
            for b in norm_count_left[a]:
                if norm_count_left[a][b] != 0:
                    div = norm_count_left[a][b]/(probs[a]*probs[b])
                    I_score[a][b] = round(math.log2(div),4)
        

        return I_score


    def insert_scores_right_given_left(self, freq_3gram, probs):
        ctx = self.symbol_contexts(freq_3gram)
        I_score = {}

        count_right_given_left, norm_right_given_left, \
        norm_count_right = self.neighbor(ctx, self.LEFT)
            
        for a in norm_count_right:
            I_score[a] = {}
            for b in norm_count_right[a]:
                if norm_count_right[a][b] != 0:
                    div = norm_count_right[a][b]/(probs[a]*probs[b])
                    I_score[a][b] = round(math.log2(div),4)
        

        return I_score


    def right_given_left(self, ctx):
        return self.neighbor(ctx, 0)


    def left_given_right(self, ctx):
        return self.neighbor(ctx, 1)


    def neighbor(self, ctx, i):
        count_neighbor_i = {}
        norm_i = {}
        norm_count_neighbor_i = {}
        
        ctx = self.remove_padding(ctx, i)

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


    def remove_padding(self, ctx, i):
        ctx_filtered = {}

        for s in ctx:
            ctx_filtered[s] = {}
            for t in ctx[s]:
                if t[i] != '-':
                    ctx_filtered[s][t] = ctx[s][t]
        

        return ctx_filtered
    
    
    def subst_scores(self, freq_3gram):
        ctx = self.symbol_contexts(freq_3gram)
        co_occur = {}
        S_score = {}
        norm_co_occur = 0

        for a in ctx:
            co_occur[a] = {}
            for b in ctx:
                set_a = set(ctx[a].keys())
                set_b = set(ctx[b].keys())
                ctx_a_b = set_a.intersection(set_b)
                co_occur[a][b] = self.co_occurrence(ctx, ctx_a_b, a, b)
                norm_co_occur += co_occur[a][b]       
        
        M = self.get_M(ctx, co_occur, norm_co_occur)
        probs = self.prob_symbols(M)

        count = 0

        for a in probs:
            count += probs[a]

        Expec = self.get_expec_val(probs)

        for a in M:
            S_score[a] = {}

            for b in M:
                if M[a][b] != 0:
                    S_score[a][b] = round(math.log2(M[a][b]/Expec[a][b]),2)
                
        
        return S_score


    def symbol_contexts(self, freq_3gram):
        ctx = {}

        for gram in freq_3gram:
            s = gram[1]

            if s not in ctx:
                ctx[s] = {}

            ctx[s][(gram[0],gram[2])] = freq_3gram[gram]

        
        return ctx


    def co_occurrence(self, ctx, ctx_a_b, a, b):
        co_occur_a_b = 0

        for c in ctx_a_b:
            if a == b:
                co_occur_a_b += (ctx[a][c] * (ctx[a][c] - 1))/2
            else:
                co_occur_a_b += ctx[a][c] * ctx[b][c]


        return co_occur_a_b


    def get_M(self, ctx, co_occur, norm_co_occur):
        M ={}

        for a in ctx:
            M[a] = {}
            for b in ctx:
                M[a][b] = round(co_occur[a][b]/norm_co_occur,4)


        return M

    def prob_symbols(self, M):
        probs = {}

        for a in M:
            probs[a] = 0
            for b in M:
                probs[a] += M[a][b]
            
        
        return probs


    def get_expec_val(self, probs):
        expec_val = {}

        for a in probs:
            expec_val[a] = {}

            for b in probs:
                if a == b:
                    expec_val[a][b] = probs[a]**2
                else:
                    expec_val[a][b] = 2*probs[a]*probs[b]

        
        return expec_val


    def get_symbol_probab(self, df):
        df_temp = df.groupby('concept:name').agg(count=('time:timestamp','count'))
        total_occur = df_temp.sum().to_dict()['count']
        occur = (df_temp/total_occur).to_dict()['count']


        return occur


 
