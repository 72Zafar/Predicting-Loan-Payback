import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import LoanPayBackPredictionConfig
from src.entity.s3_estimator import proj1Estimator
from pandas import DataFrame


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
            self.annual_income = annual_income,
            self.dett_to_income_ratio = debt_to_income_ratio,
            self.credit_score = credit_score,
            self.loan_amount = loan_amount,
            self.interest_rate = interest_rate,
            self.gender = gender,
            self.marital_status = marital_status,
            self.loan_purpose = loan_purpose,
            self.income_to_loan_ratio = income_to_loan_ratio,
            self.affordability_ratio = affordability_ratio,
            self.risk_score = risk_score,
            self.grade = grade,
            self.subgrade_num = subgrade_num,
            self.employment_stability = employment_stability,
            self.education_num = education_num
        
        except Exception as e:
            print("Exception occured in LoanPayBackPrediction class constructor")
            raise MyException(e,sys)
        
    def loan_payback_input_data_frame(self)-> DataFrame:
        """ This function returns the input data frame for loan payback prediction"""
        logging.info("Entered loan_payback_input_data_frame method of LoanPayBackPrediction class")
        try:
           
            input_data = {
                "annual_income":[self.annual_income],
                "debt_to_income_ratio":[self.dett_to_income_ratio],
                "credit_score":[self.credit_score],
                "loan_amount":[self.loan_amount],
                "interest_rate":[self.interest_rate],
                "gender":[self.gender],
                "marital_status":[self.marital_status],
                "loan_purpose":[self.loan_purpose],
                "income_to_loan_ratio":[self.income_to_loan_ratio],
                "affordability_ratio":[self.affordability_ratio],
                "risk_score":[self.risk_score],
                "grade":[self.grade],
                "subgrade_num":[self.subgrade_num],
                "employment_stability":[self.employment_stability],
                "education_num":[self.education_num]
            }

            logging.info("Created input data frame for loan payback prediction")
            logging.info("Exited loan_payback_input_data_frame method of LoanPayBackPrediction class")

            return input_data
        
        except Exception as e:
            raise MyException(e,sys)
        
    
class Loan_Payback_Data_Classifier:
    def __init__(self,prediction_pipeline_config:LoanPayBackPredictionConfig = LoanPayBackPredictionConfig())->None:
        """
        :param prediction_pipeline_config: Configuration for prediction pipeline
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e,sys)
        
    def predict(self,dataframe)-> str:
        """
        This is method of loan payback data classifier
        return: Prediction in string format
        """
        try:
            logging.info("Entered the predict method of Loan_Payback_Data_Classifier class")
            model = proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e,sys)