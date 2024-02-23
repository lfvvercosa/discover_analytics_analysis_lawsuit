from utils.converter.csv_to_xes import convert_csv_to_xes
from pm4py.objects.log.importer.xes import importer as xes_importer


# convert all CSV logs

# 1

# path_csv = 'xes_files/real_processes/Detail_Incident_Activity.csv'
# path_xes = 'xes_files/real_processes/Detail_Incident_Activity.xes'
# case_id_col = 'case'
# act_col = 'activity'
# timestamp_col = 'timestamp'
# csv_sep=','


# convert_csv_to_xes(path_csv,
#                    path_xes,
#                    case_id_col,
#                    act_col,
#                    timestamp_col,
#                    csv_sep)

# 2

path_csv = 'xes_files/real_processes/Detail_Incident_Activity.csv'
path_xes = 'xes_files/real_processes/BPI_Challenge_2014_' + \
    'Detail_Incident_Activity.xes'
case_id_col = 'Incident ID'
act_col = 'IncidentActivity_Type'
timestamp_col = 'DateStamp'
csv_sep=';'


convert_csv_to_xes(path_csv,
                   path_xes,
                   case_id_col,
                   act_col,
                   timestamp_col,
                   csv_sep)

# 3


