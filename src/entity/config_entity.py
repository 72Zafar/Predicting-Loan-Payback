import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

class DataIngestionConfig:
    data_ingestiom_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestiom_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestiom_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestiom_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
    collection_name: str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataValidationConfig:
    data_validationConfig: str = os.path.join(training_pipeline_config.artifacts_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validationConfig, DATA_VALIDATION_REPORT_FILE_NAME)
