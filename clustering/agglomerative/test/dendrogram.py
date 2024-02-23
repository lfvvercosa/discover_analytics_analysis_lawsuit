from scipy.cluster import hierarchy 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


# attachment = A
# conclusion = B
# discharge = C
# petition = D
# publication = E
# redistribution = F
# register = G
# trial = H

traces = ['GBHEC', 'GBHBEC' ,'BAHEE','GBDF','GGBDAF', 'BAHEC']

Z =  hierarchy.linkage()

t = max(Z[:,2])
ax = plt.figure().gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

a = hierarchy.dendrogram(Z, 
                     color_threshold=t*10, 
                     labels=traces, 
                     leaf_rotation=0,
                     link_color_func=lambda x: '#000000',
                     ax=ax)

# custom_lines = [Line2D([0], [0], color='#FFFFFF', lw=0.1, linewidth=0.01),
#                 Line2D([0], [0], color='#FFFFFF', lw=0.1, linewidth=0.01),
#                 Line2D([0], [0], color='#FFFFFF', lw=0.1, linewidth=0.01)]

custom_lines = [Line2D([0],[0],linewidth=0.01),
                Line2D([0],[0]),
                Line2D([0],[0])]

# ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'], loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend(['Cold', 'Medium', 'Hot'])

text = \
'A: $attachment$\n' + \
'B: $conclusion$\n' + \
'C: $discharge$\n' + \
'D: $petition$\n' + \
'E: $publication$\n' + \
'F: $redistribution$\n' + \
'G: $register$\n' + \
'H: $trial$'

ax.annotate(text,
            xy=(1.05,0.18), xycoords='axes fraction',
            textcoords='offset points',
            size=13,
            bbox=dict(fc=(1.0, 1.0, 1.0), ec='black'))

plt.yticks(fontsize=13)
plt.gcf().set_size_inches(8, 3)
plt.tight_layout()
plt.show(block=True)
# plt.savefig('images/example_agglomerative.png', dpi=600)
plt.close()

print('done!')