import boto3


def read_from_s3(bucket, filename):
    try:
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, 
                        filename)
        
        return object.get()['Body'].read()
    except:
        return []


s3_bucket = 'luiz-doutorado-projetos2'
s3_dir_res = 'testes_markov/resultados_markov/k_1/IMf/res2221.json'

a = read_from_s3(s3_bucket, s3_dir_res)

print('oi')
