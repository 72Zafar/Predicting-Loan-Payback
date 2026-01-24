import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from src.entity.estimator import BlendedEnsembleModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _get_models(self):
        lgb = LGBMClassifier(
            objective=self.model_trainer_config.lgbm_objective,
            n_estimators=self.model_trainer_config.lgbm_n_estimators,
            learning_rate=self.model_trainer_config.lgbm_learning_rate,
            random_state=self.model_trainer_config.lgbm_random_state,
            n_jobs=self.model_trainer_config.lgbm_n_jobs
        )

        cat = CatBoostClassifier(
            iterations=self.model_trainer_config.cat_iterations,
            depth=self.model_trainer_config.cat_depth,
            random_seed=self.model_trainer_config.cat_random_seed,
            loss_function=self.model_trainer_config.cat_loss_function,
            eval_metric=self.model_trainer_config.cat_eval_metric,
            verbose=False
        )

        xgb = XGBClassifier(
            n_estimators=self.model_trainer_config.xgb_n_estimators,
            learning_rate=self.model_trainer_config.xgb_learning_rate,
            max_depth=self.model_trainer_config.xgb_max_depth,
            subsample=self.model_trainer_config.xgb_subsample,
            colsample_bytree=self.model_trainer_config.xgb_colsample_bytree,
            random_state=self.model_trainer_config.xgb_random_state,
            n_jobs=self.model_trainer_config.xgb_n_jobs,
            eval_metric="logloss"
        )

        return lgb, cat, xgb

    def _evaluate(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)

        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
            # accuracy_score=accuracy_score(y_true, y_pred),
            roc_auc_score=roc_auc_score(y_true, y_pred_proba)
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            oof_lgb = np.zeros(len(X_train))
            oof_cat = np.zeros(len(X_train))
            oof_xgb = np.zeros(len(X_train))

            for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logging.info(f"Fold {fold + 1}")

                X_tr, X_val = X_train[trn_idx], X_train[val_idx]
                y_tr, y_val = y_train[trn_idx], y_train[val_idx]

                lgb, cat, xgb = self._get_models()

                lgb.fit(X_tr, y_tr)
                cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
                xgb.fit(X_tr, y_tr)

                oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
                oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
                oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]

            blend_weights = (0.4, 0.35, 0.25)
            w_lgb, w_cat, w_xgb = blend_weights

            oof_blend = (
                w_lgb * oof_lgb +
                w_cat * oof_cat +
                w_xgb * oof_xgb
            )

            metric_artifact = self._evaluate(y_train, oof_blend)

            # Train FINAL models on FULL data
            lgb, cat, xgb = self._get_models()
            lgb.fit(X_train, y_train)
            cat.fit(X_train, y_train, verbose=False)
            xgb.fit(X_train, y_train)

            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            final_model = BlendedEnsembleModel(
                preprocessing_object=preprocessing_obj,
                lgb_model=lgb,
                cat_model=cat,
                xgb_model=xgb,
                blend_weights=blend_weights,
                threshold=0.5
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise MyException(e, sys)
