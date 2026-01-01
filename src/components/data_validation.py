import json 
import os
import sys
import pandas as pd
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion
        :param data_validation_config: Configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file( file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    
    def validate_number_of_columns(self, dataframe: DataFrame)-> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns in the dataframe

        Output      :   Return bool value based on validation result
        On Failure  :   Write on exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns is present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    def is_columns_exist(self, dataframe: DataFrame)-> bool:
        """
        Method Name :   is_columns_exist
        Description :   This method validates the existence of a category and numerical columns

        Output      :   Return bool value based on validation result
        On Failure  :   Write on exception log and then raise an exception
        """
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            
            for column in self._schema_config["calegorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            
            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")
            
            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e,sys) 
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_validation(self)-> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation components

        Output      :   Return DataValidationArtifact object
        On Failure  :   Write on exception log and then raise an exception
        """
        try:
            validation_error_msg = "" 
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            )

            # checking the col len of dataframe for train\test df
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f"Column are missing in training dataframe"
            else:
                logging.info("All required columns are present in train dataframe")

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Column are missing in test dataframe"
            else:
                logging.info("Alll required columns are present in test dataframe")
            
            # validating col dtypes for train\test
            status = self.is_columns_exist(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in train dataframe"
            else:
                logging.info(f"All categorical/int columns present in train dataframe {status}")
            
            status = self.is_columns_exist(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe"
            else:
                logging.info(f"All categorical/int columns present in test dataframe {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Ensure the directory for validaton_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save the validation status as a json file
            validation_report = {
                "validation_status" : validation_status,
                "message" : validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)
            
            logging.info("Data validation artifact created and saved to JSON file")
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)