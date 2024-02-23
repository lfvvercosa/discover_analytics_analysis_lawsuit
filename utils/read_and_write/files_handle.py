def remove_preffix(f):
    pos = f.rfind('/')

    return f[pos+1:]

def remove_suffix(f):
    if '.xes.gz' in f:
        f = f.replace('.xes.gz', '')
    if '.xes' in f:
        f = f.replace('.xes', '')
    
    return f


def remove_preffix_suffix(f):
    f = remove_suffix(f)

    return remove_preffix(f)