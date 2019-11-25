import shutil
import os
import logging
import boto3
from botocore.exceptions import ClientError


def upload_file(file_name, bucket, object_name=None, extraArgs=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(bucket, object_name, file_name):


    s3_client = boto3.client('s3')
    try:
        response = s3_client.download_file(bucket, object_name, file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def send_dataset_to_cloud(dataset_name):

    path = os.getcwd() + '/'

    shutil.make_archive(dataset_name, 'zip', path + dataset_name)

    bucket = 'zumidatasets'

    uploaded = upload_file(dataset_name + '.zip', bucket)

    return uploaded


def download_dataset_from_cloud(dataset_zip_name, dataset_directory):

    path = os.getcwd() + '/' + dataset_directory

    os.mkdir(path)

    bucket = 'zumidatasets'

    downloaded = download_file(bucket, dataset_zip_name, dataset_zip_name)

    shutil.unpack_archive(dataset_zip_name, path, 'zip')

    return downloaded


def send_model_to_cloud(model_name):

    path = os.getcwd() + '/'

    shutil.make_archive(model_name, 'zip', path + model_name)

    bucket = 'zumimodels'

    uploaded = upload_file(model_name + '.zip', bucket)

    return uploaded


def download_model_from_cloud(model_zip_name):

    path = os.getcwd()

    bucket = 'zumimodels'

    downloaded = download_file(bucket, model_zip_name, model_zip_name)

    shutil.unpack_archive(model_zip_name, path, 'zip')

    return downloaded

'''

#path = os.getcwd()

#shutil.make_archive('data', 'zip', path + '\\trafficsigns')

#uploaded = upload_file('data.zip', 'computervisiontestjet')

#print(uploaded)


#Downloadpath = path + '\\data2.zip'

#pathlib.Path(Downloadpath).parent.mkdir(parents=True, exist_ok=True)
#'MyBucket', 'hello-remote.txt', 'hello2.txt

#downloaded = download_file('computervisiontestjet', 'data.zip', 'Downloadeddata\\data3.zip')

#shutil.unpack_archive('data3.zip', path, 'zip')
#print("Archive file unpacked successfully.")

#print(downloaded)

'''