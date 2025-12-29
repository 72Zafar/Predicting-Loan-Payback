import os 
from datetime import datetime

# For mongodb
DATABASE_NAME = "Loan_PayBack"
COLLECTION_NAME = "loan_payback_data"
MONGODB_URL_KEY = "MONGODB_URL"

# Data Ingestion - Use the actual collection name that has data
DATA_INGESTION_COLLECTION_NAME: str = "loan_payback_data"

PIPELINE_NAME:str = ""
ARTIFACTS_DIR: str = "artifacts"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "loan_paid_back"

CURRENT_YEAR = datetime.now().year
PREPOCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "loan_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

""" 
Data ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.25

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

