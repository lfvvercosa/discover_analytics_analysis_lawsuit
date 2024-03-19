import zipfile
import core.my_loader as my_loader
from pm4py.objects.log.importer.xes import importer as xes_importer

from core import my_log_orchestrator

# The event-log contains the procedural movements of each lawsuit ordered by its timestamp
# It will be used to the creation of n-gram later.

if __name__ == "__main__":
    base_path = 'dataset/'
    court_type = 'tribunais_trabalho'
    zip_file = court_type + 'tribunais_trabalho/justica_trabalho.zip'
    # my_justice = 'TRIBUNAIS_TRABALHO'
    output_path = 'dataset/' + court_type
    level_magistrado = 1
    level_serventuario = 2
    level_type = 1
    percent_appear_mov = 0.5
    mandatory_mov = 246
    # type is used to refer to process class
    mandatory_type = 158
    mandatory_type_level = 1
    is_map_movement = True

    # Unzip dataset 
    with zipfile.ZipFile(base_path + zip_file, 'r') as zip_ref:
        zip_ref.extractall(path=base_path + court_type + '/')

    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)

    start_limit = 2020
    n_stds_outlier = 2
    

    trt_all = [
        'TRT1',
        'TRT2',
        'TRT3',
        'TRT4',
        'TRT5',
        'TRT6',
        'TRT7',
        'TRT8',
        'TRT9',
        'TRT10',
        'TRT11',
        'TRT12',
        'TRT13',
        'TRT14',
        'TRT15',
        'TRT16',
        'TRT17',
        'TRT18',
        'TRT19',
        'TRT20',
        'TRT21',
        'TRT22',
        'TRT23',
        'TRT24',
    ]

    for trt in trt_all:
        print('processing trt: ' + trt)
        df, df_subj, df_mov = \
            my_log_orchestrator.parse_dataframes(court_type,
                                                 [trt],
                                                 base_path,
                                                 trt)

        if df_mov is not None:
            df_mov = my_log_orchestrator.pre_process(
                        df_subj,
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
                    )

            if df_mov is not None:
                saving_path = output_path + '/' + trt + '.xes'
                my_log_orchestrator.create_xes_file(df_mov, saving_path, trt.upper())

    print('done!')
