import boto3
import utils.read_and_write.s3_handle as s3_handle


my_path = 'temp/my_test.txt'
bucket = 'luiz-doutorado-projetos2'
filename = 'variant_analysis/my_test.txt'
content = "hello darkness my old friend\nI'm talking to you again"

s3_handle.write_to_s3(bucket, filename, content)


