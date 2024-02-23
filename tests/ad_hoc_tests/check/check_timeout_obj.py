import networkx as nx
import multiprocessing
import time


def my_func(a, ret_dict):
    time.sleep(6)
    G = nx.DiGraph()
    G.add_edges_from([
        ('a','b'),
        ('c','d'),
    ])

    ret_dict['graph'] = G


def timeout_func(a, timeout):
    manager = multiprocessing.Manager()
    ret_dict = manager.dict()

    p = multiprocessing.Process(target=my_func,
                                    args=(
                                            a,
                                            ret_dict
                                         )
                               )

    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
    
        return -1

    p.join()

    return ret_dict['graph']


if __name__ == '__main__':
    G = timeout_func('a', 5)

    print(G.edges)