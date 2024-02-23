

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__

    return helper

def memoize(func):
    mem = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return memoizer

@call_counter
@memoize    
def levenshtein(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    
    res = min([levenshtein(s[:-1], t) + 1, # deletion
               levenshtein(s, t[:-1]) + 1, # insertion
               levenshtein(s[:-1], t[:-1]) + cost] # substitution
             )

    return res


def levenshtein_context(s, t):
    return leven_cont_deb(s, t, '')

def levenshtein_context2(s, t):
    return levenshtein_context_worker(s, t, '')

@call_counter
@memoize    
def levenshtein_context_worker(s, t, right_s):
   
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
    
    simil_del = get_simil('deletion', curr_s, None, left_s, right_s, context)
    simil_ins = get_simil('insertion', curr_t, None, curr_s, right_s, context)
    simil_sub = get_simil('substitution', curr_s, curr_t, None, None, context)

    if s == '':
        return levenshtein_context_worker(s, t[:-1], curr_t) + simil_ins
    
    if t == '':
        return levenshtein_context_worker(s[:-1], t, curr_s) + simil_del

    a = levenshtein_context_worker(s[:-1], t, curr_s) + simil_del
    b = levenshtein_context_worker(s, t[:-1], curr_t) + simil_ins
    c = levenshtein_context_worker(s[:-1], t[:-1], curr_t) + simil_sub

    res = max([a,b,c])


    return res


def leven_cont_deb(s, t, right_s):
   
    if s == '' and t == '':
        return [('-',0)]
    
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
    
    simil_del = get_simil('deletion', curr_s, None, left_s, right_s, context)
    simil_ins = get_simil('insertion', curr_t, None, curr_s, right_s, context)
    simil_sub = get_simil('substitution', curr_s, curr_t, None, None, context)

    if s == '':
        return leven_cont_deb(s, t[:-1], curr_t) + [('I',curr_t, simil_ins)]
    
    if t == '':
        return leven_cont_deb(s[:-1], t, curr_s) + [('D',curr_s, simil_del)]

    a = leven_cont_deb(s[:-1], t, curr_s) + [('D',curr_s, simil_del)]
    b = leven_cont_deb(s, t[:-1], curr_t) + [('I',curr_t, simil_ins)]
    c = leven_cont_deb(s[:-1], t[:-1], curr_t) + [('S',curr_s, curr_t, simil_sub)]

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
    

def get_simil(op, act, act2, left_act, right_act, context):
    if act != '':
        if op == 'substitution':
            if act2 == '':
                return None
            else:
                try:
                    return context['substitution'][act][act2]
                except:
                    print()
    
        if op == 'insertion':
            if left_act != '':
                return context['ins_right_given_left'][act][left_act]
            else:
                return context['ins_left_given_right'][act][right_act]
        
        if op == 'deletion':
            if left_act != '':
                return context['del_right_given_left'][act][left_act]
            else:
                return context['del_left_given_right'][act][right_act]


        raise Exception('invalid "op" for similarity')
    
    return None


context = {
    'substitution':
        {
         'A':{'A':6, 'B':2, 'C':3},
         'B':{'A':2, 'B':6, 'C':1},
         'C':{'A':3, 'B':1, 'C':6}, 
        },
    'ins_right_given_left':
        {
         'A':{'A':1, 'B':0.5, 'C':3, '':3},
         'B':{'A':1, 'B':0.5, 'C':2, '':3},
         'C':{'A':0.5, 'B':2, 'C':1.5, '':3},
        },
    'del_right_given_left':
        {
         'A':{'A':1, 'B':0.5, 'C':3, '':3},
         'B':{'A':1, 'B':0.5, 'C':2, '':3},
         'C':{'A':0.5, 'B':2, 'C':1.5, '':3},
        },
    'ins_left_given_right':
        {
         'A':{'A':1, 'B':0.5, 'C':3, '':3},
         'B':{'A':1, 'B':0.5, 'C':2, '':3},
         'C':{'A':0.5, 'B':5, 'C':1.5, '':3},
        },
    'del_left_given_right':
        {
         'A':{'A':1, 'B':0.5, 'C':3, '':3},
         'B':{'A':1, 'B':0.5, 'C':2, '':3},
         'C':{'A':0.5, 'B':5, 'C':1.5, '':3},
        },
}

# context = {
#     'substitution':
#         {
#          'A':{'A':0, 'B':1, 'C':1},
#          'B':{'A':1, 'B':0, 'C':1},
#          'C':{'A':1, 'B':1, 'C':0}, 
#         },
#     'ins_right_given_left':
#         {
#          'A':{'A':1, 'B':1, 'C':1, '':1},
#          'B':{'A':1, 'B':1, 'C':1, '':1},
#          'C':{'A':1, 'B':1, 'C':1, '':1},
#         },
#     'del_right_given_left':
#         {
#          'A':{'A':1, 'B':1, 'C':1, '':1},
#          'B':{'A':1, 'B':1, 'C':1, '':1},
#          'C':{'A':1, 'B':1, 'C':1, '':1},
#         },
#     'ins_left_given_right':
#         {
#          'A':{'A':1, 'B':1, 'C':1, '':1},
#          'B':{'A':1, 'B':1, 'C':1, '':1},
#          'C':{'A':1, 'B':1, 'C':1, '':1},
#         },
#     'del_left_given_right':
#         {
#          'A':{'A':1, 'B':1, 'C':1, '':1},
#          'B':{'A':1, 'B':1, 'C':1, '':1},
#          'C':{'A':1, 'B':1, 'C':1, '':1},
#         },
# }

print(levenshtein_context("ABBBC", "AC")) 
print(levenshtein_context2("ABBBC", "AC")) 









# print("The function was called " + str(levenshtein_context_worker.calls) + " times!")



# print(levenshtein("Python", "Peithen"))
# print("The function was called " + str(levenshtein.calls) + " times!")

