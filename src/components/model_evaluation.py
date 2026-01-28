# from src.entity.config_entity import ModelEvaluationConfing
# from src.entity.artifact_entity import ModelTrainerArtifact,DataIngestionArtifact,ModelEvaluationArtifact
# from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
# from src.utils.main_utils import read_yaml_file,load_object
# from src.entity.s3_estimator import proj1Estimator
# from src.exception import MyException
# from src.logger import logging

# from sklearn.metrics import (
#     roc_auc_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     accuracy_score
# )
# from sklearn.preprocessing import StandardScaler,  OrdinalEncoder
# from typing import Optional
# from dataclasses import dataclass
# import pandas as pd
# import numpy as np
# import sys

# @dataclass
# class EvaluateModelResponse:
#     trained_model_f1_score: float
#     best_model_f1_score: float
#     is_model_accepted: bool
#     difference: float


# class ModelEvaluation:
#     def __init__(self, model_eval_config: ModelEvaluationConfing, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact):
#         """
#         """
#         try:
#             self.model_eval_config = model_eval_config
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.model_trainer_artifact = model_trainer_artifact
#             # load schema file for drop columns and target column
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

#         except Exception as e:
#             raise MyException(e, sys)
    
#     def get_best_model(self)-> Optional[proj1Estimator]:
#         """
#         This function is used to get model form production stage.
#         Op: Return the model object if available in s3 bucket
#         """
#         try:
#             bucket_name = self.model_eval_config.bucket_name
#             model_path = self.model_eval_config.s3_model_key_path
#             proj1_Estimator = proj1Estimator(bucket_name=bucket_name, model_path=model_path)   

#             if proj1_Estimator.is_model_present(model_path=model_path):
#                 return proj1_Estimator
#             return None 
#         except Exception as e:
#             raise MyException(e, sys)
        
    
#     def remove_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
#         """
#         remove outliers from numerical columns using IQR method.
#         """
#         try:
#             numerical_columns = ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
#             for col in numerical_columns:
#                 Q1 = df[col].quantile(0.25)
#                 Q3 = df[col].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_bound = Q1 - 1.5 * IQR
#                 upper_bound = Q3 + 1.5 * IQR
#                 df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
#             return df
#         except Exception as e:
#             raise MyException(e, sys)


#     def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Create new features to enhance model performance.
#         """
#         try:
#             df = df.copy()

#             df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
#             df['affordability_ratio'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200)

#             df['risk_score'] = (
#                 df['debt_to_income_ratio'] * 0.3 +
#                 (800 - df['credit_score']) / 800 * 0.3 +
#                 df['interest_rate'] / 25 * 0.2 +
#                 (df['loan_amount'] / df['annual_income']) * 0.2
#             )
#             if 'grade_subgrade' in df.columns:
#                 df['grade'] = df['grade_subgrade'].str[0]
#                 df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)


#             employment_mapping = {
#                 'Unemployed': 0,
#                 'Student': 1,
#                  'Self-employed': 2,
#                 'Employed': 3,
#                 'Retired': 2
#             }
#             df['employment_stability'] = df['employment_status'].map(employment_mapping)

#             education_mapping = {
#                 'High School': 1,
#                 'Other': 2,
#                 'Bachelor\'s': 3,
#                 'Master\'s': 4,
#                 'PhD': 5
#             }
#             df['education_num'] = df['education_level'].map(education_mapping)
#             return df
#         except Exception as e:
#             raise MyException(e, sys)

#     def preprocess_data(self, df: pd.DataFrame)-> pd.DataFrame:
#         """
#         Preprocess the data by dropping unnecessary columns and encoding categorical features.
#         """
#         try:
#             # make copies so original dfs are not modified unexpectedly
#             df = df.copy()
#             # drop columns properly..
#             cols_to_drop = ["education_level", "employment_status","grade_subgrade"]
#             cols_to_drop = [c for c in cols_to_drop if c in df.columns]
#             if cols_to_drop:
#                 df = df.drop(columns=cols_to_drop)
            
#                 # Numerical features (kept for reference)
#                 features = [
#                     'id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
#                     'loan_amount', 'interest_rate', 'loan_paid_back',
#                     'income_to_loan_ratio', 'affordability_ratio', 'risk_score',
#                     'subgrade_num', 'employment_stability', 'education_num'
#                 ]
#                 # categorical features
#                 categorical_cols = ['gender', 'marital_status', 'loan_purpose', 'grade']

#                 # Keep only categorical columns that actually exist
#                 categorical_cols = [c for c in categorical_cols if c in df.columns]

#                 # Encode categorical columns robustly
#                 if categorical_cols:
#                     enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#                     enc.fit(df[categorical_cols])
#                     df[categorical_cols] = enc.transform(df[categorical_cols])
    
#             return df
#         except Exception as e:
#             raise MyException(e, sys)

#     def standerd_scale_data(self,df: pd.DataFrame):
#         """
#         Standardize numerical arrays using StandardScaler.

#         Fits scaler on X_train only and transforms X_test with the same scaler.
#         Returns (X_train_scaled, X_test_scaled, scaler)
#         """
#         try:
#             stander = StandardScaler()
#             df = stander.fit_transform(df)
#             return df, stander
#         except Exception as e:
#             raise MyException(e, sys) 
        
    
#     def evaluate_models(self)-> ModelEvaluationArtifact:
#         """
#         This function is used to evaluate trained model with production model and choose the best model. 
#         """
#         try:
#             test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
#             x,y = test_df.drop(TARGET_COLUMN,axis=1), test_df[TARGET_COLUMN]

#             logging.info("Test data loaded and transforming it for prediction")

#             x =  self.remove_outliers(x)
#             x = self.create_new_features(x)
#             x = self.preprocess_data(x)
#             x = self.standerd_scale_data(x)
#             logging.info("Test data transformed for prediction")
            
#             trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
#             logging.info("Trained model loaded for prediction")

#             trained_model_roc_auc = self.model_trainer_artifact.metric_artifact.test_roc_auc
#             logging.info("Trained model ROC AUC loaded for prediction")

#             best_model_roc_auc = None
#             best_model = self.get_best_model()
#             if best_model is not None:
#                 logging.info("Computiing ROC AUC for best model")
#                 y_hat_best_model = best_model.predict(x)
#                 best_model_roc_auc = roc_auc_score(y, y_hat_best_model)
#                 logging.info(f"ROC AUC for best model: {best_model_roc_auc}")

#             tmp_best_model_score = 0 if best_model_roc_auc is None else best_model_roc_auc
#             result = EvaluateModelResponse(
#                 trained_model_roc_auc=trained_model_roc_auc,
#                 best_model_roc_auc=tmp_best_model_score,
#                 is_model_accepted=trained_model_roc_auc > tmp_best_model_score,
#                 difference = trained_model_roc_auc - tmp_best_model_score)
            
#             logging.info(f"Result:{result}")
#             return result
#         except Exception as e:
#             raise MyException(e, sys)
        
#     def initiate_model(self)-> ModelEvaluationArtifact:
#         """
#         This function is used to initiate all steps of the model evaluation
#         Returns model evaluation artifact
#         """
#         try:
#             print("_____________________________________________________________________")
#             logging.info("Initialized model evaluation component.")
#             evaluate_model_response = self.evaluate_models()
#             s3_model_path = self.model_eval_config.s3_model_key_path

#             model_evaluation_artifactv = ModelEvaluationArtifact(
#                 is_model_accepted=evaluate_model_response.is_model_accepted,
#                 s3_model_path = s3_model_path,
#                 trained_model_path=self.model_trainer_artifact.trained_model_file_path,
#                 changed_accuracy=evaluate_model_response.difference,
#             )

#             logging.info(f"Model evaluation artifact: {model_evaluation_artifactv}")

#             return model_evaluation_artifactv
#         except Exception as e:
#             raise MyException(e, sys) 
        





















import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfing
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from src.entity.s3_estimator import proj1Estimator


@dataclass
class EvaluateModelResponse:
    trained_model_roc_auc: float
    best_model_roc_auc: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfing,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def _load_best_model(self) -> Optional[proj1Estimator]:
        """
        Load production model from S3 if it exists
        """
        try:
            estimator = proj1Estimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path
            )

            if estimator.is_model_present(self.model_eval_config.s3_model_key_path):
                return estimator

            return None
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_models(self) -> EvaluateModelResponse:
        """
        Evaluate newly trained model against production model
        using the SAME raw test data
        """
        try:
            logging.info("Starting model evaluation")

            # Load raw test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Load trained ensemble model
            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            logging.info("Computing ROC-AUC for trained model")

            trained_model_proba = trained_model.predict_proba(X_test)
            trained_model_roc_auc = roc_auc_score(y_test, trained_model_proba)

            # Load production model (if exists)
            best_model = self._load_best_model()
            best_model_roc_auc = 0.0

            if best_model is not None:
                logging.info("Computing ROC-AUC for production model")

                best_model_proba = best_model.predict_proba(X_test)
                best_model_roc_auc = roc_auc_score(y_test, best_model_proba)

            is_accepted = trained_model_roc_auc > best_model_roc_auc
            difference = trained_model_roc_auc - best_model_roc_auc

            result = EvaluateModelResponse(
                trained_model_roc_auc=trained_model_roc_auc,
                best_model_roc_auc=best_model_roc_auc,
                is_model_accepted=is_accepted,
                difference=difference
            )

            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrates model evaluation and returns artifact
        """
        try:
            logging.info("Initiating model evaluation component")

            evaluation_response = self.evaluate_models()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_response.difference
            )

            logging.info(
                f"Model evaluation artifact created: {model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)
