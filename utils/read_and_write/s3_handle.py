import boto3


def write_to_s3(bucket, filename, file_content):
    s3 = boto3.resource('s3')
    object = s3.Object(bucket, 
                       filename)
    object.put(Body=file_content)


def read_from_s3(bucket, filename):
    try:
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, 
                        filename)
        
        return eval(object.get()['Body'].read())
    except:
        return None