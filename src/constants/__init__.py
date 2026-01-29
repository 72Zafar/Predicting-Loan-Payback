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


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
""" lightgbm model parameters """

MODEL_TRAINER_EXPECTED_SCORE : float = 0.7

LGBM_OBJECTIVE: str = "binary"
LGBM_METRIC: str = "auc"
LGBM_BOSSTING_TYPE: str = "gbdt"
LGBM_N_ESTIMATORS: int = 10  # Actual values is 1000
LGBM_LEARNING_RATE: float = 0.01
LGBM_COLSAMPLE_FREQ: int = 1
LGBM_MIN_CHILD_SAMPLES: int = 20
LGBM_REAG_ALPHA: float = 0.05
LGBM_REAG_LAMBDA: float = 0.01
LGMB_RANDOM_STATE: int = 42
LGMB_N_JOBS: int = -1
LGMB_DEVICE: str = "cpu"
LGMB_VERBOSE: int = -1

""" CatBoost model parameters """
CAT_ITERATIONS: int = 15 # Actual values is 3000
CAT_DEPTHE: int = 8
CAT_LOSS_FUNCTION: str = "Logloss"
CAT_EVAL_METRIC: str = "AUC"
CAT_RANDOM_SEED: int = 42
CAT_AUTO_CLASS_WEIGHTS: str = "Balanced"
CAT_L2_LEAF_REG: int = 5
CAT_TASK_TYPE: str = "CPU"

"""" XGBoost model parameters """
XGB_OBJECTIVE: str = "binary:logistic"
XGB_EVAL_METRIC: str = "auc"
XGB_LEARNING_RATE: float = 0.01
XGB_MAX_DEPTH: int = 8
XGB_MIN_CHILD_WEIGHT: int = 3
XGB_COLSAMPLE_BYTREE: float = 0.3
XGB_SUBSAMPLE: float = 0.6
XGB_REG_ALPHA: float = 0.5
XGB_REG_LAMBDA: float = 2.0
XGB_N_ESTIMATORS: int = 10  # Actual values is 10000
XGB_RANDOM_STATE: int = 42
XGB_N_JOBS: int = -1
XGB_VERBOSE: int = -1
XGB_DEVICE: str = "cpu"
XGB_TREE_METHOD: str = "hist"

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopspro-end"
MODEL_PUSHER_S3_KEY = "model-registry"

APP_HOST = "0.0.0.0"
APP_PORT = 5000