from experiments.variant_analysis.approaches.core.context_aware_clust.\
     SimilarityScore import SimilarityScore
import pickle
from functools import wraps


def call_counter(func):
    @wraps(func)
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__

    return helper


def memoize(func):
    cache = {}
    @wraps(func)
    def wrap(*args,**kwargs):
        key = pickle.dumps((args, kwargs))
        if key not in cache:
            # print('Running func with ', args)
            cache[key] = func(*args, **kwargs)
        # else:
            # print('result in cache')
        return cache[key]
    return wrap


class ContextAwareClustering:
    similScore = None
    context = None


    def __init__(self, log):
        self.similScore = SimilarityScore(log)
        self.context = self.similScore.get_similarity_dict()


    def levenshtein_context(self, s, t):
        sim = self.levenshtein_context_worker(s, t, '')
        dist = 0

        sim = round(sim, 4)
        dist = round((len(s) + len(t))/sim,4)


        return dist
    

    def __levenshtein_context_debug(self, s, t):
        deb = self.leven_cont_deb(s, t, '')
        sim = 0
        dist = 0

        for d in deb:
            sim += d[2]

        sim = round(sim, 4)
        dist = round((len(s) + len(t))/sim,4)


        return deb, sim, dist


    @call_counter
    @memoize    
    def levenshtein_context_worker(self, s, t, right_s):
    
        if s == '' and t == '':
            return 0
        
        if len(s) > 1:
            left_s = s[-2]
        else:
            left_s = ''

        if s == '':
            curr_s = ''
        else:
            curr_s = s[-1]

        if t == '':
            curr_t = ''
        else:
            curr_t = t[-1]
        
        simil_del = self.get_simil('deletion', curr_s, None, left_s, right_s)
        simil_ins = self.get_simil('insertion', curr_t, None, curr_s, right_s)
        simil_sub = self.get_simil('substitution', curr_s, curr_t, None, None)

        if s == '':
            return self.levenshtein_context_worker(s, t[:-1], curr_t) + simil_ins
        
        if t == '':
            return self.levenshtein_context_worker(s[:-1], t, curr_s) + simil_del

        a = self.levenshtein_context_worker(s[:-1], t, curr_s) + simil_del
        b = self.levenshtein_context_worker(s, t[:-1], curr_t) + simil_ins
        c = self.levenshtein_context_worker(s[:-1], t[:-1], curr_t) + simil_sub

        # res = max([a,b,c])

        if a >= b and a >= c:
            return a
        elif b >= a and b >= c:
            return b
        else:
            return c

        # return res


    @call_counter
    def leven_cont_deb(self, s, t, right_s):
   
        if s == '' and t == '':
            return [('-', '-', 0)]
        
        if len(s) > 1:
            left_s = s[-2]
        else:
            left_s = ''

        if s == '':
            curr_s = ''
        else:
            curr_s = s[-1]

        if t == '':
            curr_t = ''
        else:
            curr_t = t[-1]
        
        simil_del = self.get_simil('deletion', curr_s, None, left_s, right_s)
        simil_ins = self.get_simil('insertion', curr_t, None, curr_s, right_s)
        simil_sub = self.get_simil('substitution', curr_s, curr_t, None, None)

        if s == '':
            return self.leven_cont_deb(s, t[:-1], curr_t) + [('I',curr_t, simil_ins)]
        
        if t == '':
            return self.leven_cont_deb(s[:-1], t, curr_s) + [('D',curr_s, simil_del)]

        a = self.leven_cont_deb(s[:-1], t, curr_s) + [('D',curr_s, simil_del)]
        b = self.leven_cont_deb(s, t[:-1], curr_t) + [('I',curr_t, simil_ins)]
        c = self.leven_cont_deb(s[:-1], t[:-1], curr_t) + [('S',curr_s, curr_t, simil_sub)]

        simil_a = sum([x[-1] for x in a])
        simil_b = sum([x[-1] for x in b])
        simil_c = sum([x[-1] for x in c])

        if simil_a >= simil_b and simil_a >= simil_c:
            return a
        elif simil_b >= simil_a and simil_b >= simil_c:
            return b
        else:
            return c
    
        # if simil_a <= simil_b and simil_a <= simil_c:
        #     return a
        # elif simil_b <= simil_a and simil_b <= simil_c:
        #     return b
        # else:
        #     return c
    

    def get_simil(self, op, act, act2, left_act, right_act):
        if act != '':
            if op == 'substitution':
                if act2 == '':
                    return None
                else:
                    try:
                        return self.context['substitution'][act][act2]
                    except:
                        print()
        
            if op == 'insertion':
                if left_act != '':
                    return self.context['ins_right_given_left'][act][left_act]
                else:
                    return self.context['ins_left_given_right'][act][right_act]
            
            if op == 'deletion':
                if left_act != '':
                    return self.context['del_right_given_left'][act][left_act]
                else:
                    return self.context['del_left_given_right'][act][right_act]


            raise Exception('invalid "op" for similarity')
        
        return None



    
