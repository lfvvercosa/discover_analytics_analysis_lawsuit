import boto3 
import requests


response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
instance_id = response.text

ec2 = boto3.resource('ec2')
instance = ec2.Instance(instance_id)

print('id: ' + str(instance))
print('shutdown: ' + str(instance.terminate()))