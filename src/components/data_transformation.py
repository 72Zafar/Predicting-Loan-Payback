import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,  OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler

from src.constants import TARGET_COLUMN,SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.entity.artifact_entity import DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    
    def remove_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        remove outliers from numerical columns using IQR method.
        """
        try:
            numerical_columns = ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
            for col in numerical_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features to enhance model performance.
        """
        try:
            df = df.copy()

            df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
            df['affordability_ratio'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200)

            df['risk_score'] = (
                df['debt_to_income_ratio'] * 0.3 +
                (800 - df['credit_score']) / 800 * 0.3 +
                df['interest_rate'] / 25 * 0.2 +
                (df['loan_amount'] / df['annual_income']) * 0.2
            )
            if 'grade_subgrade' in df.columns:
                df['grade'] = df['grade_subgrade'].str[0]
                df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)


            employment_mapping = {
                'Unemployed': 0,
                'Student': 1,
                 'Self-employed': 2,
                'Employed': 3,
                'Retired': 2
            }
            df['employment_stability'] = df['employment_status'].map(employment_mapping)

            education_mapping = {
                'High School': 1,
                'Other': 2,
                'Bachelor\'s': 3,
                'Master\'s': 4,
                'PhD': 5
            }
            df['education_num'] = df['education_level'].map(education_mapping)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def preprocess_data(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Preprocess the data by dropping unnecessary columns and encoding categorical features.
        """
        try:
            # make copies so original dfs are not modified unexpectedly
            df = df.copy()
            # drop columns properly..
            cols_to_drop = ["education_level", "employment_status","grade_subgrade"]
            cols_to_drop = [c for c in cols_to_drop if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            
                # Numerical features (kept for reference)
                features = [
                    'id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
                    'loan_amount', 'interest_rate', 'loan_paid_back',
                    'income_to_loan_ratio', 'affordability_ratio', 'risk_score',
                    'subgrade_num', 'employment_stability', 'education_num'
                ]
                # categorical features
                categorical_cols = ['gender', 'marital_status', 'loan_purpose', 'grade']

                # Keep only categorical columns that actually exist
                categorical_cols = [c for c in categorical_cols if c in df.columns]

                # Encode categorical columns robustly
                if categorical_cols:
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    enc.fit(df[categorical_cols])
                    df[categorical_cols] = enc.transform(df[categorical_cols])
    
            return df
        except Exception as e:
            raise MyException(e, sys)

    def standerd_scale_data(self, X_train,X_test):
        """
        Standardize numerical arrays using StandardScaler.

        Fits scaler on X_train only and transforms X_test with the same scaler.
        Returns (X_train_scaled, X_test_scaled, scaler)
        """
        try:
            stander = StandardScaler()
            X_train_scaled = stander.fit_transform(X_train)
            X_test_scaled = stander.transform(X_test)
            return X_train_scaled, X_test_scaled, stander
        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate data transformation process.
        """
        try:
            logging.info("Reading training and testing data")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            # load train & test data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path) 
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Train & Test Data Loaded Successfully")

            # remove outliers
            logging.info("Removing outliers from training and testing data")

            train_df = self.remove_outliers(train_df)
            test_df = self.remove_outliers(test_df)
            logging.info("Outliers removed successfully")

            # create new features
            logging.info("Creating new features for training and testing data")
            train_df = self.create_new_features(train_df)
            test_df = self.create_new_features(test_df)
            logging.info("New features created successfully")

            # preprocess data
            logging.info("Preprocessing training & testing data")
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            logging.info("Preprocessing completed successfully")

            # train test split
            logging.info("Splitting input and target features")
            # extract target first, then drop target column to get input features
            train_target = train_df[TARGET_COLUMN]
            test_target = test_df[TARGET_COLUMN]

            train_df_input = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            test_df_input = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            logging.info("Input and target features split successfully")

            logging.info("Applying standard scaling to input features")
            # prepare numpy arrays for scaling
            input_features_train_arr = train_df_input.values
            input_features_test_arr = test_df_input.values

            # Standardize numerical (fit scaler on train only)
            input_features_train_arr, input_features_test_arr, scaler = self.standerd_scale_data(
                X_train=input_features_train_arr,
                X_test=input_features_test_arr
            )
            logging.info("Standard scaling applied successfully")

            # Appling under sampling to handle class imbalance
            logging.info("Applying under sampling to handle class imbalance in training data")

            # Apply under sampling
            undersampler = RandomUnderSampler(random_state=42)
            input_features_train_resampled, train_target = undersampler.fit_resample(
                input_features_train_arr, train_target
            )
            logging.info("Under sampling applied successfully")
            train_arr = np.c_[input_features_train_resampled, np.array(train_target)]
            test_arr = np.c_[input_features_test_arr, np.array(test_target)]
            logging.info("Train & Test arrays created successfully")

            transformer = {
                "scaler": scaler,
                "undersampler": undersampler,
                "feature_columns": list(train_df_input.columns)
            }

            save_object(
                self.data_transformation_config.transformed_object_file_path,transformer
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,array=test_arr
            )
            logging.info("Saving tranformation object and transformed files..")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys)