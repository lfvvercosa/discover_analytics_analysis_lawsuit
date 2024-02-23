import pm4py


if __name__ == "__main__":
    path = 'xes_files/variant_analysis/exp7/size10/low_complexity/0/tree2.ptml'
    tree = pm4py.read_ptml(path)

    pm4py.view_process_tree(tree, format='png')