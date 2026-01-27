import boto3
from src.configuration.aws_connection import S3client
from io import StringIO
from typing import Union,List
import os,sys
from src.exception import MyException
from src.logger import logging 
from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle


class SimpleStorageService:
    """
    A class for interacting with AWS storage services, providing methods for file management 
    data uplodas, and data retrival in S3 Bucket
    """

    def __init__(self):
        """
        Initialize the SimpleStorageService instance with s3 resource and client from the S3client class.
        """
        s3_client  = S3client()
        self.s3_resource = s3_client.s3_resourse
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self,bucket_name,s3_key) -> bool:
        """
        Checks if a specified S3 key (file path) is available in the specified bucket.
        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Key path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_object = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_object) > 0
        except Exception as e:
            raise MyException(e,sys)
        

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False)-> Union[StringIO,str]:
        """
        Reads the specified S3 object with optional decoding and formatting.

        Args:
            object_name (str): The S3 object name.
            decode (bool): Whether to decode the object content as a string.
            make_readable (bool): Whether to convert content to StringIO for DataFrame usage.

        Returns:
            Union[StringIO, str]: The content of the object, as a StringIO or decoded string.
        """
        try:
            # Read and decode the object content id decode = True
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode else object_name.get()["Body"].read()
            )

            # Convert of StringIO if make_readable = True
            conv_func = lambda: StringIO(func()) if make_readable else func()
            return conv_func()
        except Exception as e:
            raise MyException(e,sys) from e
    
    def get_backet(self, bucket_name: str)-> Bucket:
        """
        Retrieves the S3 bucket object based on the provided bucket name.

        Args:
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Bucket: S3 bucket object.
        """
        logging.info("Entered the get_bucket method of SimpleStorageService class")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket
        except Exception as e:
            raise MyException(e,sys) from e
        
    def get_file_object(self,filename:str, bucket_name: str)->Union[List[object], object]:
        """
        Retrieves the file object(s) from the specified bucket based on the filename.

        Args:
            filename (str): The name of the file to retrieve.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Union[List[object], object]: The S3 file object or list of file objects.
        """
        logging.info("Entered the get_file_object method of SimpleStorageService class")
        try:
            bucket = self.get_backet(bucket_name)
            file_object = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_object = func(file_object)
            logging.info("Exited the get_file_object method of SimpleStorageService class")
            return file_object
        except Exception as e:
            raise MyException(e,sys)
