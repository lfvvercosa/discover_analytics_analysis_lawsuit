from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def visualize_petri_net(my_file):
    print('### File: ' + str(my_file))
    net, im, fm = pnml_importer.apply(my_file)

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)


if __name__ == '__main__':
    my_file = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/" + \
              "event_logs/paper_event_logs/petri_net_sepsis.pnml"
    net, im, fm = pnml_importer.apply(my_file)

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    print()