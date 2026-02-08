from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional
from src.exception import MyException
import sys

from src.constants import APP_HOST,APP_PORT
from src.pipline.prediction_pipeline import Loan_Payback_Data_Classifier,LoanPayBack_Columns
from src.pipline.training_pipeline import TrainingPipeline

# Initialize FastAPI app
app = FastAPI()
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class DataFrom:
    """
    DataFrom class handle and pross incoming data 
    """
    def __init__(self,request:Request):
        self.request: Request = request
        self.Id:Optional[int] = None
        self.Annual_income:Optional[float] = None
        self.Debet_to_income_ratio:Optional[float] = None
        self.Credit_score:Optional[float] = None
        self.Loan_amount:Optional[float] = None
        self.Interest_rate:Optional[float] = None
        self.Gender:Optional[str] = None
        self.Marital_status:Optional[str] = None
        self.Loan_prupose:Optional[str] = None
        self.Income_to_loan_ratio:Optional[float] = None
        self.Affordability:Optional[float] = None
        self.Risk_score:Optional[float] = None
        self.Grade:Optional[str] = None
        self.Subgrade_num:Optional[int] = None
        self.Employment_stability:Optional[str] = None
        self.Education_num:Optional[str] = None
    

    async def get_loan_back_data(self):
        """
        Method to retrieve and assign from data to class attributes.
        This Method is asynchronous to handle from data fetching without blocking.
        """
        form = await self.request.form()
        self.Annual_income = form.get("Annual_income")
        self.Debet_to_income_ratio = form.get("Debet_to_income_ratio")
        self.Credit_score = form.get("Credit_score")
        self.Loan_amount = form.get("Loan_amount")
        self.Interest_rate = form.get("Interest_rate")
        self.Gender = form.get("Gender")
        self.Marital_status = form.get("Marital_status")
        self.Loan_prupose = form.get("Loan_prupose")
        self.Income_to_loan_ratio = form.get("Income_to_loan_ratio")
        self.Affordability = form.get("Affordability")
        self.Risk_score = form.get("Rist_score")
        self.Grade = form.get("Grade")  # form field is "Grate" (typo), stored as Grade for pipeline
        self.Subgrade_num = form.get("Subgrade_num")
        self.Employment_stability = form.get("Employment_stability")
        self.Education_num = form.get("Education_num")
        pass
    
    
# Route to handle form submission and make prediction
@app.post("/")
async def predictionRouteClient(request:Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """    
    try:

        form = DataFrom(request)
        await form.get_loan_back_data()
        loan_data = LoanPayBack_Columns(
            annual_income=form.Annual_income,
            debt_to_income_ratio=form.Debet_to_income_ratio,
            credit_score=form.Credit_score,
            loan_amount=form.Loan_amount,
            interest_rate=form.Interest_rate,
            gender=form.Gender,
            marital_status=form.Marital_status,
            loan_purpose=form.Loan_prupose,
            income_to_loan_ratio=form.Income_to_loan_ratio,
            affordability_ratio=form.Affordability,
            risk_score=form.Risk_score,
            grade=form.Grade,
            subgrade_num=form.Subgrade_num,
            employment_stability=form.Employment_stability,
            education_num=form.Education_num
        )

        # Convert input data to DataFrame
        loan_data_df = loan_data.loan_payback_input_data_frame()
    
        # Initialize prediction pipeline
        model_predictor = Loan_Payback_Data_Classifier()
        
        # make prediction (blended model returns 1D array of P(class=1); use class 1 if >= 0.5)
        proba = model_predictor.predict(dataframe=loan_data_df)
        prob_class1 = float(proba[0]) if hasattr(proba[0], "__float__") else float(proba[0][1])
        prediction = 1 if (prob_class1 >= 0.5) else 0

        if prediction == 1:
            return JSONResponse(
                content={"status": True, "prediction":"Loan will be paid back", "message": "Yes, the loan is likely to be paid back."}
            )
        else:
            return JSONResponse(
                content={"status": False, "prediction": "Loan will not be paid back", "message": "No, the Loan is unlikely to be paid back."}
            )


    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": None, "prediction": None, "message": str(e)}
        )

# Route to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Endpoint to serve the loan prediction form.
    """
    return templates.TemplateResponse("index.html", {"request": request})
    
if __name__ == "__main__":
    app_run(app, host=APP_HOST,port=APP_PORT)
