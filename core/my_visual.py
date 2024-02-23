import matplotlib.pyplot as plt
import pandas as pd


def rename_features(d):
    new_d = {}
    rename = {
        'SE':'Estado Sergipe',
        'CASE:COURT:CODE':'Identificador UJ',
        'PROCESSO_DIGITAL':'Processo Digital',
        'PENDENTES_NAO_CRIMINAL_2016':'Pendentes Não-Criminais 2016',
        'TAXA_DE_CONGESTIONAMENTO_TOTAL': 'Taxa de Congestionamento',
        'MOV_ATO_ORDINATORIO_11383': 'Movimento $\it{Ato}$ $\it{Ordinatório}$',
        'MOV_PROTOCOLO_DE_PETICAO_118': 'Movimento $\it{Petição}$',
        'MOV_DECISAO_3': 'Movimento $\it{Decisão}$',
        'TOTAL_OFFICIAL': 'Total Movimentos Serventuário',
        'PENDENTES_CRIMINAL_2015': 'Pendentes Criminais 2015',

        'MOVEMENTS_COUNT':'Total Movimentos',
        'MOV_JUNTADA_67': 'Movimento $\it{Juntada}$',
        'TOTAL_MAGISTRATE': 'Total Movimentos Magistrado',
        'MOV_RECEBIMENTO_132': 'Movimento $\it{Recebimento}$',
        'MOV_CONCLUSAO_51': 'Movimento $\it{Conclusão}$',
    }

    for k in d:
        new_d[rename[k]] = d[k]


    return new_d


def gen_bar_plot(se_import, top_n):
    import_plot = se_import[:top_n]

    new_dict = rename_features(import_plot.to_dict())
    import_plot = pd.Series(new_dict)
    import_plot = import_plot.sort_values(ascending=True)

    import_plot.plot.barh()
    plt.xticks(fontsize=15, fontname='verdana')
    plt.yticks(fontsize=15, fontname='verdana')
    plt.tight_layout()
    plt.show()