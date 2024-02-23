from utils.read_and_write import s3_handle
import sys


bucket = 'luiz-doutorado-projetos2'
filename = 'results_' + sys.argv[1] + '.txt'
fullname = 'variant_analysis/exp7/agglom_lev_w/' + filename
metrics = {
        'ARI':{},
        'Fitness':{},
        'Complexity':{},
    }
log_complexity = [sys.argv[1]]
log_total = 10


for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(log_total):
            metrics['ARI'][log_complex].append(i)
            metrics['Fitness'][log_complex].append(i)
            metrics['Complexity'][log_complex].append(i)

content = 'ARI:\n\n'
content += str(metrics['ARI']) + '\n\n'

content += 'Fitness:\n\n'
content += str(metrics['Fitness']) + '\n\n'

content += 'Complexity:\n\n'
content += str(metrics['Complexity']) + '\n\n'

s3_handle.write_to_s3(bucket = bucket, 
                      filename = fullname, 
                      file_content = content)

print('done!')