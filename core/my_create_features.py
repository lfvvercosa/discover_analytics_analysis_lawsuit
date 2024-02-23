from core import my_orchestrator
from core import my_loader
from core import my_utils

import re
import unidecode
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from models.DatasetSplit import DatasetSplit

DEBUG = True


def total_distinct_subjects(subjects):
    l = eval(subjects)
    
    
    return len(set(l))


def total_subjects(subjects):
    try:
        l = eval(subjects)
    except Exception as e:
        print('type of object: ' + str(type(subjects)))
        print('value of object: ' + str(subjects))
        raise Exception(e)
    
    
    return len(l)


def extract_subject(subjects):
    l = eval(subjects)


    return l[0]


def get_subject_level(breadscrum, level):
    if type(breadscrum) == str:
        breadscrum = breadscrum.split(':')

        if len(breadscrum) > level:
            return int(breadscrum[level])
        else:
            return -1
    else:
        return -1
    

def map_subject(df, df_code_subj, level):
    temp = df_code_subj[['assuntoCodigoNacional','breadscrum']]
    temp = temp.rename(columns={'assuntoCodigoNacional':'case:lawsuit:subjects'})
    
    df = df.merge(temp, on=['case:lawsuit:subjects'], how='left')

    df['case:lawsuit:subjects'] = df.apply(lambda df: \
                                      get_subject_level(
                                        df['breadscrum'],
                                        level), axis=1)
    
    df_temp = df[df['case:lawsuit:subjects'] == -1]

    df = df[~df['case:concept:name'].isin(df_temp['case:concept:name'])]
    df = df.drop(columns=['breadscrum'])


    return df


def count_zeroes_col(df, thres):
    total = len(df.index)
    zeroes_perc = {}
    remove_cols = []

    cols = df.columns
    cols = [c for c in cols if c[-5:-2] == '-20']

    for c in cols:
        if c != 'case:court:code':
            zeroes = df[c].value_counts()[0]
            zeroes_perc[c] = round(zeroes/total,2)
            
            if zeroes_perc[c] > thres:
                remove_cols.append(c)
    

    return remove_cols


def rename_mov_cols(df, df_code_mov, ngram):
    df_temp = df_code_mov[['movimentoCodigoNacional','movimentoNome']]
    df_temp = df_temp.set_index('movimentoCodigoNacional')
    map_vals = df_temp.to_dict()['movimentoNome']
    rename_cols = {}

    for c in df.columns:
        try:
            if 'MOV_' in c:
                if ngram == 1:
                    name = map_vals[int(c[len('MOV_'):])]
                    name = name.replace(' ', '_').upper()
                    name = name.replace('/', '_').upper()
                    name = unidecode.unidecode(name)
                    name = 'MOV_' + name

                    rename_cols[c] = name
                if ngram == 2:
                    t = eval(c[len('MOV_'):])
                    name1 = apply_to_mov_name(t[0], map_vals)
                    name2 = apply_to_mov_name(t[1], map_vals)
                    name = name1 + '=>' + name2

                    rename_cols[c] = name
        except:
            pass

    df = df.rename(columns=rename_cols)


    return df


def rename_clus_col(df, col, preffix):
    df[col] = preffix + df[col].astype(str)


    return df


def rename_subject_cols(df, df_code_subj):
    df_temp = df_code_subj[['assuntoCodigoNacional','assuntoNome']]
    df_temp = df_temp.set_index('assuntoCodigoNacional')
    map_vals = df_temp.to_dict()['assuntoNome']
    rename_cols = {}

    for c in df.columns:
        try:
            if 'ASSU_' in c:
                    name = map_vals[int(c[len('ASSU_'):])]
                    name = name.replace(' ', '_').upper()
                    name = name.replace('/', '_').upper()
                    name = unidecode.unidecode(name)
                    name = 'ASSU_' + name

                    rename_cols[c] = name
        except:
            pass

    df = df.rename(columns=rename_cols)


    return df

def rename_type(df, df_code_type):
    df_temp = df_code_type[['classeProcessual','classeNome']]
    df_temp = df_temp.rename(columns={'classeProcessual':'case:lawsuit:type'})

    df = df.merge(df_temp, on='case:lawsuit:type', how='left')
    df['case:lawsuit:type_temp'] = df['classeNome'] + '_' + \
        df['case:lawsuit:type'].astype(str)
    df['case:lawsuit:type_temp'] = 'CLA_' + df['case:lawsuit:type_temp'].\
        str.replace(' ', '_').str.upper()
    df['case:lawsuit:type_temp'] = df['case:lawsuit:type_temp'].apply(remove_accents)
    df = df.drop(columns=['case:lawsuit:type','classeNome'])
    df = df.rename(columns={'case:lawsuit:type_temp':'case:lawsuit:type'})
    
    return df


def rename_subject(df, df_code_subj):
    df_temp = df_code_subj[['assuntoCodigoNacional','assuntoNome']]
    df_temp = df_temp.rename(columns={'assuntoCodigoNacional':'case:lawsuit:subjects'})

    df = df.merge(df_temp, on='case:lawsuit:subjects', how='left')
    df['case:lawsuit:subjects_temp'] = df['assuntoNome'] + '_' + \
        df['case:lawsuit:subjects'].astype(str)
    df['case:lawsuit:subjects_temp'] = 'ASSU_' + df['case:lawsuit:subjects_temp'].\
        str.replace(' ', '_').str.upper()
    df['case:lawsuit:subjects_temp'] = df['case:lawsuit:subjects_temp'].apply(remove_accents)
    df = df.drop(columns=['case:lawsuit:subjects','assuntoNome'])
    df = df.rename(columns={'case:lawsuit:subjects_temp':'case:lawsuit:subjects'})
    
    return df


def rename_classific(df):
    df['Classificação da unidade'] = 'CLAS_' + df['Classificação da unidade'].\
        str.replace('-','_').str.replace(' ','_').str.replace(';','').\
        str.replace('(','').str.replace(')','').str.replace(',','').str.upper()
    df['Classificação da unidade'] = df['Classificação da unidade'].apply(remove_accents)

    return df


def rename_to_pattern(df, col, preffix):
    df[col] = preffix + df[col].\
        str.replace('-','_').str.replace(' ','_').str.replace(';','').\
        str.replace('(','').str.replace(')','').str.replace(',','').str.upper()
    df[col] = df[col].apply(remove_accents)


    return df


def remove_accents(name):
    try:
        return unidecode.unidecode(name)
    except Exception as e:
        print()

def standard_name_cols(columns):
    std_name = {}

    for c in columns:
        name = c.replace(' ','_').replace('-','_').upper()
        name = unidecode.unidecode(name)
        std_name[c] = name

    return std_name


def apply_to_mov_name(name, map_vals):
    name = map_vals[int(name)]
    name = name.replace(' ', '_').upper()
    name = name.replace('/', '_').upper()
    name = unidecode.unidecode(name)
    name = 'MOV_' + name


    return name


def group_infrequent_categoric(df, col, thres):
    df_temp = df.groupby(col).agg(count=(col,'count'))
    total = df_temp.sum()[0]
    min_val = thres * total
    updated_vals = {}
    curr_vals = df_temp.to_dict()['count']


    for k in curr_vals:
        if curr_vals[k] > min_val:
            updated_vals[k] = k
        else:
            updated_vals[k] = 'CLA_OUTRO_' + col
    
    classe_processual = [k for k in updated_vals]
    classe_processual_mapped = [v for v in updated_vals.values()]
    
    df_map = pd.DataFrame.from_dict({
        col:classe_processual,
        col+'_MAPEADA':classe_processual_mapped,
        })
    
    df = df.merge(df_map, on=col, how='left')
    df = df.drop(columns=col)
    df = df.rename(columns={col+'_MAPEADA':col})

    return df


def map_infrequent(df, col, thres):
    df_temp = df.groupby(col).agg(count=(col,'count'))
    total = df_temp.sum()[0]
    min_val = thres * total
    updated_vals = {}
    curr_vals = df_temp.to_dict()['count']
    count = 0
    map_labels = {}

    for k in curr_vals:
        if curr_vals[k] > min_val:
            updated_vals[k] = k
        else:
            updated_vals[k] = -1
    
    update = list(updated_vals.values())
    update.sort(key=lambda x: int(x))

    for v in update:
        if v not in map_labels:
            map_labels[v] = count
            count += 1

    updated_vals = {k:map_labels[v] for k,v in updated_vals.items()}


    return updated_vals


def group_infrequent(df, col, thres, preffix):
    df_temp = df.groupby(col).agg(count=(col,'count'))
    total = df_temp.sum()[0]
    min_val = thres * total
    updated_vals = {}
    curr_vals = df_temp.to_dict()['count']


    for k in curr_vals:
        if curr_vals[k] > min_val:
            updated_vals[k] = k
        else:
            updated_vals[k] = preffix + 'OUTRO'
    
    values = [k for k in updated_vals]
    values_mapped = [v for v in updated_vals.values()]
    
    df_map = pd.DataFrame.from_dict({
        col:values,
        col+'_MAPEADA':values_mapped,
        })
    
    df = df.merge(df_map, on=col, how='left')
    df = df.drop(columns=col)
    df = df.rename(columns={col+'_MAPEADA':col})

    return df

def group_infrequent_tipo(df):
    df['TIPO'] = df['TIPO'].replace('AADJ','Desconhecido')
    df['TIPO'] = df['TIPO'].replace('-','Desconhecido')
    df['TIPO'] = df['TIPO'].replace('Desconhecido','TIPO_DESCONHECIDO')
    df['TIPO'] = df['TIPO'].replace('UJ1','TIPO_UJ1')
    df['TIPO'] = df['TIPO'].replace('UJ2','TIPO_UJ2')


    return df


def winsorize_col(s, min_perc, max_perc):
    temp = winsorize(s, (min_perc,max_perc))
        
    if temp.min() != temp.max():
        s = temp

    return s


def apply_one_hot_encoder(df, col):
    # Get one hot encoding of column
    one_hot = pd.get_dummies(df[col])
    # Drop column as it is now encoded
    df = df.drop(col, axis = 1)
    # Join the encoded df
    df = df.join(one_hot)
    
    
    return df


def apply_target_encoding(df, 
                          col, 
                          gt, 
                          test_size, 
                          random_seed,
                          metric):
    dataset_split = DatasetSplit()
    df_work = df[['case:concept:name',col,gt]]

    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
             train_test_split(df_work,
                              gt, 
                              test_size, 
                              random_seed)
    
    df_train = pd.DataFrame(X_train_, columns=[
                                              'case:concept:name', 
                                               col,
                                            #    gt
                                              ])
    df_train[gt] = pd.DataFrame(y_train_,columns=[gt])[gt]

    df_encode = df_train.groupby(col).agg(target_encode=(gt,metric))
    df_encode = df_encode.reset_index()
    df_encode = df[['case:concept:name',col]].merge(df_encode, on=col, how='left')
    df_encode = df_encode.drop(columns=col)
    df_encode = df_encode.rename(columns={'target_encode':col})

    df = df.drop(columns=col)
    df = df.merge(df_encode, on='case:concept:name', how='left')


    return df


def normalize_cols(df):
    cols_gt = ['TEMPO_PROCESSUAL_TOTAL_DIAS']
    scaler = MinMaxScaler()
    cols_norm = [c for c in df.columns if c not in cols_gt]

    df[cols_norm] = scaler.fit_transform(df[cols_norm])
    

    return df


def convert_to_list(subjects):
    return eval(subjects)


def create_features(df,
                    df_code_subj,
                    df_code_mov,
                    df_pend,
                    df_congest,
                    df_ibge,
                    df_clus,
                    n,
                    min_perc,
                    max_perc,
                    level_subject
                    ):

    if DEBUG:
        print('total lines: ' + str(len(df.index)))

    ### Create IBGE features
    # df_county  = my_orchestrator.get_ibge_features(df, df_ibge)

    ### Get time lawsuit ###
    df_time = my_utils.get_trace_time(df)

    ### Handle lawsuit subject feature ###
    if DEBUG:
        print('extract subject...')

    df['total_subjects'] = df.apply(lambda df: total_subjects(
                                        df['case:lawsuit:subjects']), axis=1
                                    )
    df['total_distinct_subjects'] = df.apply(lambda df: total_distinct_subjects(
                                                df['case:lawsuit:subjects']), axis=1
                                            )
    df['case:lawsuit:subjects'] = df.apply(lambda df: convert_to_list(
                                           df['case:lawsuit:subjects']), axis=1
                              )
    
    ### Create 1-gram for subjects
    df_gram_subjects = my_orchestrator.create_1_gram_subjects(df, 
                                                              df_code_subj, 
                                                              min_perc, 
                                                              max_perc, 
                                                              level_subject)

    ### Create n-gram ###
    df_gram = my_orchestrator.create_n_gram_movements(df, min_perc, max_perc, n)

    ### Create total movements features
    df_total_movs = df.groupby('case:concept:name').\
                       agg(movements_count=('concept:name','count'))
    df_total_movs_first_level = my_orchestrator.\
                                    get_total_movs_first_level(df, df_code_mov)
    
    ### Merge dataframes
    df_gram = df_gram.merge(df_time, on='case:concept:name', how='left')
    df_gram = df_gram.merge(df_total_movs, on='case:concept:name', how='left')
    df_gram = df_gram.merge(df_total_movs_first_level, 
                            on='case:concept:name', 
                            how='left')
    df_gram = df_gram.merge(df_gram_subjects, on='case:concept:name', how='left')
    df_gram = df_gram.merge(df_clus, on='case:concept:name', how='left')


    df_merge = df[[
        'case:concept:name',
        'case:lawsuit:type',
        'case:lawsuit:number',
        'case:court:code',
        'case:court:name',
        # 'case:court:county',
        'case:digital_lawsuit',
        'case:lawsuit:subjects',	
        'case:secrecy_level',
        'total_distinct_subjects',
        'total_subjects',
        'case:lawsuit:percent_key_mov',
    ]]

    df_merge = df_merge.drop_duplicates(subset='case:concept:name')
    df_merge = df_merge.set_index('case:concept:name')

    df_gram = df_gram.merge(df_merge, on='case:concept:name', how='left')
    df_gram = df_gram.reset_index()

    df_gram = df_gram.merge(df_pend, on='case:court:code', how='left')
    df_gram = df_gram.merge(df_congest, on='case:court:code', how='left')


    # df_gram = df_gram.merge(df_county, on='case:court:county', how='left')

    df_gram = df_gram.drop(columns=[
                                    # 'case:court:county',
                                    'UF',
                                    'case:lawsuit:subjects',
                                    ]
                          )
    df_gram = df_gram.rename(columns={'Nome_UF':'UF'})
    

    return df_gram


def process_features(df_feat, 
                     base_path, 
                     ngram,
                     trunc_min,
                     trunc_max,
                     infreq_thres, 
                     test_size,
                     random_seed,
                     metric_encode,
                     is_statistics=False):
    thres = 0.6
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'

    # Remove repeated or not useful features
    drop_cols = [c for c in df_feat.columns if 'PENDENTES-EXECUÇÃO JUDICIAL' in c or \
                 'CRIMINAL-' in c]
    drop_cols += [
        # 'case:concept:name',
        # 'index',
        'case:lawsuit:number',
        'case:secrecy_level',
        'Município',
        'Justiça',
        'Unidade Judiciária',
        'Município sede',
        'Classificação',
        'Classificação da unidade',
        'Tipo de unidade',
        'Tribunal',
        'Tipo',
    ]

    if 'index' in df_feat.columns:
        drop_cols.append('index')

    df_feat = df_feat.drop(columns=drop_cols)

    ## Fill nulls
    regexp = re.compile('^ASSU_')
    subj_cols = [c for c in df_feat.columns if regexp.search(c)]
    # mov_cols += ['Classificação da unidade']

    for c in subj_cols:
        df_feat[c] = df_feat[c].fillna(list(df_feat[c].mode())[0])

    df_feat['case:digital_lawsuit'] = df_feat['case:digital_lawsuit'].replace(-1,0)
    df_feat['case:court:code'][df_feat['case:court:code'] == 'CODIGO_INEXISTENTE'] = '0'
    
    # df_feat['Mesorregiao'] = df_feat['Mesorregiao'].fillna('Desconhecido')

    # Remove columns with high null (or zero) percentage (above 60%)    
    rem_cols = count_zeroes_col(df_feat, thres)
    df_feat = df_feat.drop(columns=rem_cols)

    print('total cols with high null percentage: ' + str(len(rem_cols)))

    # Rename columns
    df_code_mov = my_loader.load_df_movements(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_subj = my_loader.load_df_subject(base_path)
    
    df_feat = rename_mov_cols(df_feat, df_code_mov, ngram)
    df_feat = rename_type(df_feat, df_code_type)
    df_feat = rename_subject_cols(df_feat, df_code_subj)
    df_feat['case:digital_lawsuit'] = 'DIG_' + df_feat['case:digital_lawsuit'].\
                                      astype(str)

    # df_feat = rename_classific(df_feat)
    # df_feat = rename_to_pattern(df_feat, col='UF', preffix='UF_')
    # df_feat = rename_to_pattern(df_feat, col='Mesorregiao', preffix='MESO_')
    # df_feat = rename_clus_col(df_feat, 'case:lawsuit:cluster_act', 'CLUS_ACT_')

    rename = {
        # 'case:lawsuit:subjects':'ASSUNTO_PROCESSUAL',
        'case:lawsuit:type':'CLASSE_PROCESSUAL',
        'case:digital_lawsuit':'PROCESSO_DIGITAL',
        'total_distinct_subjects':'TOTAL_ASSUNTOS_DISTINTOS',
        'total_subjects':'TOTAL_ASSUNTOS',
        'total_time':'TEMPO_PROCESSUAL_TOTAL_DIAS',
        'case:court:name':'NUMERO_TRT',
    }

    df_feat = df_feat.rename(columns=rename)
    all_except_id = [c for c in df_feat.columns if c != 'case:concept:name']
    df_feat = df_feat.rename(columns=standard_name_cols(all_except_id))

    df_feat = group_infrequent_categoric(df_feat, 'CLASSE_PROCESSUAL', infreq_thres)
    
    # df_feat = group_infrequent_tipo(df_feat)
    # df_feat = group_infrequent_categoric(df_feat, 'CLASSIFICACAO_DA_UNIDADE', infreq_thres)
    # df_feat = group_infrequent(df_feat, 'UF', infreq_thres, 'UF_')
    
    # df_feat = group_infrequent_categoric(df_feat, 'ASSUNTO_PROCESSUAL', infreq_thres)
    # df_feat = group_infrequent(df_feat, 'MESORREGIAO', 0.0025, 'MESO_')
    # df_feat = group_infrequent(df_feat, 'CASE:LAWSUIT:CLUSTER_ACT', 0.0025, 'CLUS_ACT_')


    # Winsorizing outliers from numeric cols
    df_feat['CASE:COURT:CODE'] = df_feat['CASE:COURT:CODE'].astype(int)

    do_not_process_cols = [
        'case:concept:name',
        'TEMPO_PROCESSUAL_TOTAL_DIAS',
        'PROCESSO_DIGITAL',
        'NUMERO_TRT',
        'CLUS_KME',
        'CLUS_AGG',
        'CLUS_ACT',
        'CASE:COURT:CODE',
        'CLASSE_PROCESSUAL',

        # 'CLASSIFICACAO_DA_UNIDADE',
        # 'TIPO',
        # 'TRIBUNAL',
        # 'MESORREGIAO',
        # 'CASE:LAWSUIT:CLUSTER_ACT',
        # 'UF',
    ]
    # assu_cols = [c for c in df_feat.columns if 'ASSU_' in c]
    # non_numeric_cols_and_target += assu_cols
    
    numeric_cols = [c for c in df_feat.columns if c not in do_not_process_cols]
    categ_cols = ['CLASSE_PROCESSUAL',
                  'PROCESSO_DIGITAL',
                  'NUMERO_TRT',
                  'CLUS_KME',
                  'CLUS_AGG',
                  'CLUS_ACT',
                 ]

    # if DEBUG:
        # print('Winsorizing outliers for columns: ' + str(numeric_cols))

    for c in df_feat.columns:
        if c in numeric_cols:

            df_feat[c] = winsorize_col(df_feat[c], 
                                        trunc_min, 
                                        trunc_max)

    # Fill missing
    df_feat[numeric_cols] = df_feat[numeric_cols].\
                                fillna(df_feat[numeric_cols].median())
    
    if is_statistics:
        if DEBUG:
            print('Warning: generating dataset only for statistics purpose')

        return df_feat
    else:

        df_feat = apply_target_encoding(df_feat, 
                                        'CLASSE_PROCESSUAL', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )
        
        df_feat = apply_target_encoding(df_feat, 
                                        'PROCESSO_DIGITAL', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )

        df_feat = apply_target_encoding(df_feat, 
                                        'NUMERO_TRT', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )

        df_feat = apply_target_encoding(df_feat, 
                                        'CLUS_KME', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )
        
        df_feat = apply_target_encoding(df_feat, 
                                        'CLUS_AGG', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )
        
        df_feat = apply_target_encoding(df_feat, 
                                        'CLUS_ACT', 
                                        gt, 
                                        test_size, 
                                        random_seed,
                                        metric_encode
                                       )

        # Convert categoric to numeric
        # df_feat = apply_one_hot_encoder(df_feat, 'CLASSE_PROCESSUAL')
        # df_feat = apply_one_hot_encoder(df_feat, 'PROCESSO_DIGITAL')
        # df_feat = apply_one_hot_encoder(df_feat, 'NUMERO_TRT')

        # df_feat = apply_one_hot_encoder(df_feat, 'CLASSIFICACAO_DA_UNIDADE')
        # df_feat = apply_one_hot_encoder(df_feat, 'TIPO')
        # df_feat = apply_one_hot_encoder(df_feat, 'UF')
        # df_feat = convert_categoric_one_hot_encoder(df_feat, 'ASSUNTO_PROCESSUAL')
        # df_feat = apply_one_hot_encoder(df_feat, 'MESORREGIAO')
        # df_feat = apply_one_hot_encoder(df_feat, 'CASE:LAWSUIT:CLUSTER_ACT')

            
        # Normalize columns
        df_feat[numeric_cols] = normalize_cols(df_feat[numeric_cols]).round(5)
        df_feat[categ_cols] = normalize_cols(df_feat[categ_cols]).round(5)

        # Make target to last column
        cols = list(df_feat.columns)
        cols.remove(gt)
        cols.append(gt)

        df_feat = df_feat[cols]

        # Drop duplicated rows
        df_feat = df_feat.drop_duplicates()
        

        return df_feat
    

def rename_movs(df_log, df_code_mov):
    df_work = df_code_mov[['movimentoCodigoNacional','movimentoNome']]
    df_work['movimentoCodigoNacional'] = df_work['movimentoCodigoNacional'].astype(str)
    df_work = df_work.rename(columns={'movimentoCodigoNacional':'concept:name'})
    df_log = df_log.merge(df_work, on='concept:name', how='left')
    df_log = df_log.drop(columns='concept:name')
    df_log = df_log.rename(columns={'movimentoNome':'concept:name'})

    
    return df_log