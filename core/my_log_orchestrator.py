import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

import core.my_loader as my_loader
import core.my_filter as my_filter
import core.my_parser as my_parser
import core.my_utils as my_utils
import core.my_stats as my_stats


def parse_dataframes(my_justice, just_specs, base_path, court):
    df = my_loader.load_just_spec_df(just_specs, 
                                     base_path, 
                                     my_justice)
    
    if df is not None:
        df = my_parser.parse_data(df, court)

        df_assu = df[['id',
                    'processoNumero',
                    'assuntoCodigoNacional',
                    'classeProcessual']].\
                explode('assuntoCodigoNacional')
        df_assu = df_assu[~df_assu['assuntoCodigoNacional'].isna()]

        df_mov = df.explode('movimento')
        df_mov = my_parser.parse_data_mov(df_mov)


        return df, df_assu, df_mov


    return (None,None,None)


def pre_process( 
                df_assu, 
                df_mov,
                df_code_subj,
                df_code_type,
                df_code_mov,
                start_limit,
                level_magistrado,
                level_serventuario,
                level_type,
                percent_appear_mov,
                mandatory_mov,
                mandatory_type,
                mandatory_type_level,
                is_map_movement
                ):
    
    ## Removing null traces

    print('### Total traces start:')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_null_traces(df_mov)

    if df_mov.empty:
        return None

    print('### Total traces after removing null')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Removing invalid traces

    df_mov = my_filter.filter_invalid_traces(df_mov,
                                             df_assu,
                                             df_code_subj,
                                             df_code_type,
                                             df_code_mov)
    
    if df_mov.empty:
        return None

    print('### Total traces after removing invalid')
    print(my_utils.get_total_traces_df_mov(df_mov))
    
    ## Remove traces that contain starting movement after limit year
    # df_mov = my_filter.filter_traces_started_after_time(df_mov, 
    #                                                     start_limit)
    
    # print('### Total traces after limiting movement year')
    # print(my_utils.get_total_traces_df_mov(df_mov))
    
    ## Order traces
    df_mov = df_mov.sort_values(['id','movimentoDataHora'])

    ## Handle same timestamp
    df_mov = my_utils.handle_same_timestamp(df_mov)

    ## Keep only traces that contain mandatory movement
    df_mov = my_filter.filter_traces_based_on_movement(df_mov, [mandatory_mov])

    if df_mov.empty:
        return None

    print('### Total traces after "mandatory movement" filtering')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Order traces
    df_mov = df_mov.sort_values(['id','movimentoDataHora'])

    ## Keep traces where mandatory movement is close to the end
    df_mov = my_filter.filter_traces_movement_position(df_mov, 
                                                       mandatory_mov, 
                                                       percent_appear_mov)
    
    if df_mov.empty:
        return None

    print('### Total traces after mandatory movement close to the ' + \
          'end filtering')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Order traces
    df_mov = df_mov.sort_values(['id','movimentoDataHora'])

    ## Remove trailing part of trace (after last occurrence of mandatory movement)
    df_mov = my_filter.remove_trace_trail(df_mov, mandatory_mov)

    if is_map_movement:
        ## Map movements
        df_mov = my_utils.map_movements(df_mov, 
                                        df_code_mov, 
                                        level_magistrado,
                                        level_serventuario)
    

    df_mov = my_filter.remove_traces_invalid_movements(df_mov)

    print('### Total traces after removing invalid movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    if df_mov.empty:
        return None

    ## Filter by Type attribute
    df_mov = my_filter.filter_by_type(df_mov, 
                                      df_code_type, 
                                      mandatory_type_level, 
                                      mandatory_type)
    
    print('### Total traces after filtering by type')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Map Type attribute
    # df_mov = my_utils.map_type(df_mov,
    #                            df_code_type,
    #                            level_type)
    
    if df_mov.empty:
        return None


    # print('### Total traces after mapping movements')
    # print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = df_mov.sort_values(['id','movimentoDataHora'])
    df_mov = df_mov.reset_index(drop=True)

    ## Remove traces with single activity

    df_mov = my_filter.filter_single_act_traces(df_mov)

    if df_mov.empty:
        return None

    print('### Total traces after filtering single movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Remove repeated activities in sequence (autoloops)

    # df_mov = my_filter.rem_autoloop(df_mov, 
    #                                 id_col='id', 
    #                                 act_col='movimentoCodigoNacional')

    ## Order traces
    df_mov = df_mov.sort_values(['id','movimentoDataHora'])


    return df_mov


def merge(base_path, dfs_names):
    all_dfs = []
    count = 0
    
    print('### merging...')

    for name in dfs_names:
        try:
            print('loading ' + name)
            log_path = base_path + name
            log = xes_importer.apply(log_path, 
                                    variant=xes_importer.Variants.LINE_BY_LINE)
            df = pm4py.convert_to_dataframe(log)
            count += len(df.drop_duplicates('case:concept:name').index)

            all_dfs.append(df)
        except:
            continue

    print('### Total traces before merging: ' + str(count))
    df_mov = pd.concat(all_dfs)

    count = len(df_mov.drop_duplicates('case:concept:name').index)
    print('### Total traces after merging: ' + str(count))


    return df_mov


def post_processing(df_mov,
                    perc_act_start,
                    n_stds_outlier):

    # Remove traces with rare start or end activities 
    # (possibly didnt yet finish or had already started)

    df_mov = my_filter.filter_non_complete_traces(df_mov, perc_act_start)

    print('### Total traces after filtering non-complete')
    print(my_utils.get_total_traces_df_mov(df_mov, 'case:concept:name'))

    ## Remove traces with outlier duration

    df_mov = my_filter.filter_outlier_duration_trace(df_mov, n_stds_outlier)

    print('### Total traces after filtering outlier duration')
    print(my_utils.get_total_traces_df_mov(df_mov, 'case:concept:name'))

    ## Remove traces with duration equal to zero days

    df_mov = my_filter.filter_zero_days_duration(df_mov)

    print('### Total traces after filtering zero days traces')
    print(my_utils.get_total_traces_df_mov(df_mov, 'case:concept:name'))


    return df_mov


def create_xes_file(df_mov, path, name):
    case_id = 'id'
    activity = 'movimentoCodigoNacional'
    timestamp = 'movimentoDataHora'

    df_mov = df_mov.rename(columns={
        case_id:'case:concept:name',
        activity:'concept:name',
        timestamp:'time:timestamp',
        'classeProcessual':'case:lawsuit:type',
        'assuntoCodigoNacional':'case:lawsuit:subjects',
        'processoNumero':'case:lawsuit:number',
        'orgaoJulgadorCodigo':'case:court:code',
        'processoEletronico':'case:digital_lawsuit',
        'nivelSigilo':'case:secrecy_level',
        'codigoMunicipioIBGE':'case:court:county',
        'court':'case:court:name',
        'is_mov_at_the_end':'case:lawsuit:percent_key_mov',
    })

    df_mov['case:concept:name'] = df_mov['case:concept:name'].astype(str)

    log = pm4py.convert_to_event_log(df_mov)
    log._attributes['concept:name'] = name
    pm4py.write_xes(log, path)

    print('xes file "' + path +  '" was created.')