import pandas as pd
import matplotlib.pyplot as plot


def get_ngram(lines):
    for line in lines:
        if 'best_ngram' in line:
            return int(line.split(' ')[1][0])


def gen_bars(res):
    for t in res:
        index = []
        data = {}

        for c in res[t]:
            index.append(c)
            
            for n in res[t][c]:
                if n not in data:
                    data[n] = []
                data[n].append(res[t][c][n])

        df = pd.DataFrame(data,
                           columns=list(data.keys()), 
                           index = index)
        axes = df.plot(kind='bar', rot=0)
        axes.set_title(t)
        plot.tight_layout()
        plot.show(block=True)


base_path = 'experiments/variant_analysis/exp7/size10/'
complexity = [
    'low_complexity',
    'medium_complexity',
    'high_complexity',
]
tech = [
    '1step_ngram_kms.txt',
    '2step_kms_ngram_agglom.txt',
]

res = {}

for t in tech:
    res[t] = {}
    for c in complexity:
        res[t][c] = {1:0, 2:0}

        for i in range(10):
            path = base_path + c + '/' + str(i) + '/' + t
            file1 = open(path, 'r')
            lines = file1.readlines()

            res[t][c][get_ngram(lines)] += 1
            file1.close()
 
 
print(res)

gen_bars(res)