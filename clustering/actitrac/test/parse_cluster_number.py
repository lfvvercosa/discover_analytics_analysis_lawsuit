import re


def parse_cluster_number(name):
        pattern = r"cluster_\d+"
        
        return re.findall(pattern, name)[0][len('cluster_'):]



print(parse_cluster_number('cluster_23_valid.xes'))
