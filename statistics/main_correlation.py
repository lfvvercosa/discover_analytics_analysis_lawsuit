import pandas as pd
import re


def select_most_related_of_group(df_corr, gt, reg, top_n):
    df_work = df_corr[[gt]]
    df_work = df_work.filter(regex=reg, axis=0)
    df_work['comp'] = df_work[gt].abs() 

    top_n_rows = df_work.nlargest(top_n, 'comp')


    return top_n_rows.index.to_list()


def map_sorted_by_median(median_times):
    sorted_median = dict(sorted(median_times.items(), key = lambda item: item[1], reverse=False))
    # sorted_median = median_times
    map_median_times = {}
    count = 0

    for k in sorted_median:
        map_median_times[k] = count
        count += 1


    # return map_median_times
    return sorted_median


def get_ohe_feature(df, gt, id, reg, col_name):
    regexp = re.compile(reg)
    target_cols = [c for c in df.columns if regexp.search(c)]
    median_times = {}

    df_work = df[[id] + target_cols + [gt]]
    # df_work[assu_cols] = df[assu_cols].mask(df_work[assu_cols] > 1,1)

    for c in target_cols:
        median_times[c] = df_work[df_work[c] > 0][gt].median()

    map_median_times = map_sorted_by_median(median_times)

    df_work[col_name] = df_work[target_cols].idxmax(axis=1)
    df_work[col_name] = df_work[col_name].map(map_median_times)


    return df_work[[id,col_name]]


def get_target_encode(df, gt, col):
    df_encode = df.groupby(col).agg(median=(gt,'median'))
    target_encode = df_encode.to_dict()['median']


    return df[col].map(target_encode)


def remove_cols(df):
    regexp = re.compile('EXTRAJUDICIAL')
    jud_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('SUSPENSOS')
    susp_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('RECURSOS')
    rec_cols = [c for c in df.columns if regexp.search(c)]

    rem_cols = jud_cols + susp_cols + rec_cols
    rem_cols += [
        'case:concept:name',
        'CASE:LAWSUIT:PERCENT_KEY_MOV',
        'TAXA_DE_CONGESTIONAMENTO_LIQUIDA',
        'TAXA_DE_CONGESTIONAMENTO_TOTAL',
        'ESTOQUE',
        'INDEX',
        'CASE:COURT:CODE',
        'MOV_ATO_ORDINATORIO_11383',
        'MOV_DISTRIBUICAO_26',
        'MOV_DECISAO_3',
        'MOV_AUDIENCIA_970',
        'ASSU__PARTES_E_PROCURADORES',
    ]

    df = df[[c for c in df.columns if c not in rem_cols]]
    df = df.drop_duplicates()


    return df


if __name__ == '__main__':
    DEBUG = True
    dataset_path = 'dataset/tribunais_trabalho/dataset_model_v2.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/'
    id = 'case:concept:name'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'

    df = pd.read_csv(dataset_path, sep='\t')

    df = remove_cols(df)

    df_pear = df.corr(method='pearson').round(3)
    df_spear = df.corr(method='spearman').round(3)

    cols_order = [
        'TEMPO_PROCESSUAL_TOTAL_DIAS',
        'CLUS_AGG',
        'CLUS_KME',
        'PROCESSO_DIGITAL',
        'MOV_RECEBIMENTO_132',
        'CLASSE_PROCESSUAL',
        'MOV_ENTREGA_EM_CARGA_VISTA_493',
        'MOV_PROTOCOLO_DE_PETICAO_118',
        'CLUS_ACT',
        'NUMERO_TRT',
        'MOV_RECEBIMENTO_981',
    ]

    map_names = {
        'TEMPO_PROCESSUAL_TOTAL_DIAS':'Duration',
        'CLUS_AGG':'CL_{Agg}',
        'CLUS_KME':'CL_{Kms}',
        'PROCESSO_DIGITAL':'Digital',
        'MOV_RECEBIMENTO_132':'PM_1',
        'CLASSE_PROCESSUAL':'Class',
        'MOV_ENTREGA_EM_CARGA_VISTA_493':'PM_3',
        'NUMERO_TRT':'Justice',
        'CLUS_ACT':'CL_{Act}',
        'MOV_PROTOCOLO_DE_PETICAO_118':'PM_4',
        'MOV_RECEBIMENTO_981':'PM_5',
    }

    df_pear = df_pear[cols_order]
    df_spear = df_spear[cols_order]

    df_pear = df_pear.rename(columns=map_names, index=map_names)
    df_spear = df_spear.rename(columns=map_names, index=map_names)

    df_pear.to_csv(out_path + 'pearson_corr_unified.csv', sep='\t')
    df_spear.to_csv(out_path + 'spearman_corr_unified.csv', sep='\t')


    print('done!')