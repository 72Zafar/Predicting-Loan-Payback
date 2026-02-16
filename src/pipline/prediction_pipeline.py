import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import LoanPayBackPredictionConfig
from src.entity.s3_estimator import proj1Estimator
from pandas import DataFrame

# Encoding maps to match OrdinalEncoder (categories sorted) used in training (data_transformation.py)
GENDER_MAP = {"Female": 0, "Male": 1}
MARITAL_STATUS_MAP = {"Divorced": 0, "Married": 1, "Single": 2}
LOAN_PURPOSE_MAP = {
    "Car": 0,
    "Debt consolidation": 1,
    "Education": 2,
    "Home": 3,
    "Medical": 4,
    "Other": 5,
}
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
UNKNOWN_CATEGORY = -1  # OrdinalEncoder handle_unknown='use_encoded_value', unknown_value=-1


def _to_float(x):
    """Convert to float; return 0.0 if None or invalid."""
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _to_int(x):
    """Convert to int; return 0 if None or invalid."""
    if x is None or x == "":
        return 0
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


class LoanPayBack_Columns:
    def __init__(self,
                 annual_income,
                 debt_to_income_ratio,
                 credit_score,
                 loan_amount,
                 interest_rate,
                 gender,
                 marital_status,
                 loan_purpose,
                 income_to_loan_ratio,
                 affordability_ratio,
                 risk_score,
                 grade,
                 subgrade_num,
                 employment_stability,
                 education_num
                 ):
        """
        loan payment prediction
        Input:  all feature of the trained modle for prediction
        """
        try:
            self.annual_income = annual_income
            self.dett_to_income_ratio = debt_to_income_ratio
            self.credit_score = credit_score
            self.loan_amount = loan_amount
            self.interest_rate = interest_rate
            self.gender = gender
            self.marital_status = marital_status
            self.loan_purpose = loan_purpose
            self.income_to_loan_ratio = income_to_loan_ratio
            self.affordability_ratio = affordability_ratio
            self.risk_score = risk_score
            self.grade = grade
            self.subgrade_num = subgrade_num
            self.employment_stability = employment_stability
            self.education_num = education_num

        except Exception as e:
            print("Exception occured in LoanPayBackPrediction class constructor")
            raise MyException(e,sys)
        
    def loan_payback_input_data_frame(self) -> DataFrame:
        """Build input DataFrame with categoricals encoded and all columns numeric (to match training)."""
        logging.info("Entered loan_payback_input_data_frame method of LoanPayBackPrediction class")
        try:
            # Encode categoricals (match OrdinalEncoder used in training)
            gender_val = GENDER_MAP.get(str(self.gender).strip(), UNKNOWN_CATEGORY)
            marital_val = MARITAL_STATUS_MAP.get(str(self.marital_status).strip(), UNKNOWN_CATEGORY)
            loan_purpose_val = LOAN_PURPOSE_MAP.get(str(self.loan_purpose).strip(), UNKNOWN_CATEGORY)
            grade_val = GRADE_MAP.get(str(self.grade).strip(), UNKNOWN_CATEGORY)

            input_data = {
                "annual_income": [_to_float(self.annual_income)],
                "debt_to_income_ratio": [_to_float(self.dett_to_income_ratio)],
                "credit_score": [_to_float(self.credit_score)],
                "loan_amount": [_to_float(self.loan_amount)],
                "interest_rate": [_to_float(self.interest_rate)],
                "gender": [float(gender_val)],
                "marital_status": [float(marital_val)],
                "loan_purpose": [float(loan_purpose_val)],
                "income_to_loan_ratio": [_to_float(self.income_to_loan_ratio)],
                "affordability_ratio": [_to_float(self.affordability_ratio)],
                "risk_score": [_to_float(self.risk_score)],
                "grade": [float(grade_val)],
                "subgrade_num": [_to_int(self.subgrade_num)],
                "employment_stability": [_to_int(self.employment_stability)],
                "education_num": [_to_int(self.education_num)],
            }

            logging.info("Created input data frame for loan payback prediction")
            logging.info("Exited loan_payback_input_data_frame method of LoanPayBackPrediction class")

            return DataFrame(input_data)
        
        except Exception as e:
            raise MyException(e,sys)
        
    
class Loan_Payback_Data_Classifier:
    def __init__(self,prediction_pipeline_config:LoanPayBackPredictionConfig = LoanPayBackPredictionConfig())->None:
        """
        :param prediction_pipeline_config: Configuration for prediction pipeline
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self._cached_model = None  # Cache for the loaded model
        except Exception as e:
            raise MyException(e,sys)
    
    def load_and_cache_model(self):
        """
        Load the model from S3 and cache it in memory.
        Call this once at application startup.
        """
        try:
            logging.info("Loading model from S3 and caching in memory")
            estimator = proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            self._cached_model = estimator.load_model()
            logging.info("Model successfully cached in memory")
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, dataframe) -> str:
        """
        This is method of loan payback data classifier
        Uses cached model if available, otherwise loads from S3
        return: Prediction in string format
        """
        try:
            logging.info("Entered the predict method of Loan_Payback_Data_Classifier class")
            
            # Use cached model if available
            if self._cached_model is not None:
                logging.info("Using cached model for prediction")
                result = self._cached_model.predict_proba(dataframe)
            else:
                # Fallback: load model on demand (slower)
                logging.info("Model not cached, loading from S3")
                estimator = proj1Estimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path
                )
                result = estimator.predict_proba(dataframe)
            
            return result
        except Exception as e:
            raise MyException(e, sys)