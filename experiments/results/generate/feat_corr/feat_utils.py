import matplotlib.pyplot as plt


def show_hist(features, feat_import, n):

    f_i = list(zip(features, feat_import))
    f_i.sort(key = lambda x : x[1], reverse=True)
    f_i = f_i[:n]
    f_i.reverse()

    print('### features:')
    print(f_i)

    plt.yticks(fontsize = 'xx-small')
    plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
    plt.show()

def show_hist_Threshold(features, feat_import, perc):
    
    feat = []
     
    f_i = list(zip(features, feat_import))
    f_i.sort(key = lambda x : x[1], reverse=True)
    limite = f_i[0][1] * perc
    for f in f_i:
        if(f[1]>limite):
            feat.append(f)
    
    feat.reverse()

    print('### features:')
    print(feat)

    plt.yticks(fontsize = 'xx-small')
    plt.barh([x[0] for x in feat],[x[1] for x in feat])
    plt.show()
    