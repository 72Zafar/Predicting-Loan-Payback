import sys
import pandas as pd
import numpy as np
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
    ModelEvaluationArtifact,
    DataTransformationArtifact
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
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
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
        using the preprocessed test data
        """
        try:
            logging.info("Starting model evaluation")

            # Load preprocessed test data (already processed through the full pipeline)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)
            
            # Extract features and target
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

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
