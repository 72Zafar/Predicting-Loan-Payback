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

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPOCESSING_OBJECT_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_FILE_NAME )
    exceptted_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    """ lightgbm model parameters """
    lgbm_objective = LGBM_OBJECTIVE
    lgbm_matric = LGBM_METRIC
    lgbm_bossting_type = LGBM_BOSSTING_TYPE
    lgbm_n_estimators = LGBM_N_ESTIMATORS
    lgbm_learning_rate = LGBM_LEARNING_RATE
    lgbm_colsample_freq = LGBM_COLSAMPLE_FREQ
    lgbm_min_child_samples = LGBM_MIN_CHILD_SAMPLES
    lgbm_reag_alpha = LGBM_REAG_ALPHA
    lgbm_reag_lambda = LGBM_REAG_LAMBDA
    lgbm_random_state = LGMB_RANDOM_STATE
    lgbm_n_jobs = LGMB_N_JOBS
    lgbm_device = LGMB_DEVICE
    lgbm_verbose = LGMB_VERBOSE
    """ CatBoost model parameters """
    cat_iterations = CAT_ITERATIONS
    cat_depth = CAT_DEPTHE
    cat_loss_function = CAT_LOSS_FUNCTION
    cat_eval_metric = CAT_EVAL_METRIC
    cat_random_seed = LGMB_RANDOM_STATE
    cat_auto_class_weights = CAT_AUTO_CLASS_WEIGHTS
    cat_l2_leaf_reg = CAT_L2_LEAF_REG
    cat_task_type = CAT_TASK_TYPE
    """ XGBoost model parameters """
    xgb_n_estimators = XGB_N_ESTIMATORS
    xgb_learning_rate = XGB_LEARNING_RATE
    xgb_max_depth = XGB_MAX_DEPTH
    xgb_min_child_weight = XGB_MIN_CHILD_WEIGHT
    xgb_colsample_bytree = XGB_COLSAMPLE_BYTREE
    xgb_subsample = XGB_SUBSAMPLE
    xgb_reg_alpha = XGB_REG_ALPHA
    xgb_reg_lambda = XGB_REG_LAMBDA
    xgb_random_state = XGB_RANDOM_STATE
    xgb_n_jobs = XGB_N_JOBS
    xgb_verbose = XGB_VERBOSE
    xgb_device = XGB_DEVICE
    xgb_tree_method = XGB_TREE_METHOD 


@dataclass 
class ModelEvaluationConfing:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME