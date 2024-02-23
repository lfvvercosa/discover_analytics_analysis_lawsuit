import boto3
import requests
import utils.read_and_write.s3_handle as s3_handle


def save_to_s3_and_shutdown(results_path, bucket, filename):
    # Open a file: file
    file = open(results_path, mode='r')
    
    # read all lines at once
    content = file.read()
    
    # close the file
    file.close()

    s3_handle.write_to_s3(bucket = bucket, 
                          filename = filename, 
                          file_content = content)
    
    print('wrote file to s3!')

    # shutdown ec2 instance

    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    instance_id = response.text

    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)

    print('id: ' + str(instance))
    print('shutdown: ' + str(instance.terminate()))