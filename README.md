Loan Payback Prediction – End-to-End MLOps Pipeline
Problem Statement

Banks run marketing campaigns to offer term deposits to customers.
The real business problem is:

Which customers are likely to subscribe to a term deposit so that the bank can reduce unnecessary calls and improve campaign ROI?

This project builds a full production-ready ML system to predict whether a customer will subscribe to a term deposit using historical marketing data.

The project is not only about training a model.
It focuses on building a complete, automated, and deployable ML pipeline.

Dataset

The dataset comes from a Portuguese banking marketing campaign.
Each record represents one client and their interaction history.

Target variable:

y → whether the client subscribed to a term deposit.

The raw data is stored in MongoDB Atlas instead of local files to simulate a real production data source.

What makes this project different (and not just another notebook)

Most ML projects stop at:

train → evaluate → done

This project continues to:

production-style data ingestion

schema-based validation

transformation pipelines

model registry in cloud storage

automated training and deployment

CI/CD and Docker

From notebook to production pipeline (what I actually did)
1. Notebook phase (experimentation)

I started with a Jupyter notebook for:

exploratory data analysis

feature understanding

basic preprocessing

model experimentation

This phase helped answer:

which features are useful

which columns need encoding

which columns contain missing or invalid values

Once the logic was stable, the notebook was no longer used for training.

2. Major problem I faced

The biggest mistake I initially made:

I mixed data logic, validation, and modeling inside one notebook.

This caused:

repeated preprocessing bugs

inconsistent feature ordering

silent schema changes breaking models

3. How I fixed it

I converted the entire notebook logic into a structured pipeline with strict separation of concerns.

Every step became a component.

Pipeline Architecture

The training pipeline is composed of the following components:

Data Ingestion

Reads raw customer data from MongoDB using a dedicated data access layer.

Converts key–value records into a structured DataFrame.

Stores raw and split datasets as artifacts.

Data Validation

Uses a schema file (schema.yaml) to validate:

column names

data types

presence of target column

Fails the pipeline early if the schema is violated.

This completely removed silent data bugs.

Data Transformation

Builds preprocessing pipelines for:

numerical features

categorical features

Handles:

missing values

encoding

scaling

Saves the fitted transformer as an artifact.

This ensures the same transformations are used during training and inference.

Model Trainer

Trains the estimator using the transformed data.

Evaluates model performance.

Stores the trained model artifact.

Model Evaluation

This stage compares the newly trained model with the previously deployed model stored in cloud storage.

If the new model does not improve beyond a defined threshold, it is rejected.

Model registry is implemented using Amazon Web Services S3.

Model Pusher

If the model passes evaluation:

it is pushed to the S3 model registry

it becomes the new production candidate

Prediction Pipeline

A separate inference pipeline is implemented that:

loads the latest approved model from S3

loads the preprocessing object

performs prediction on new user inputs

The model is exposed through a web API.

Why MongoDB was used

Instead of loading CSV files directly, the dataset is pushed into
MongoDB Atlas and fetched dynamically.

This simulates:

real production data ingestion

database-driven pipelines

decoupling of storage and training code

Logging and exception handling

A custom logging and exception layer was implemented and used across all components.

This allowed me to:

trace failures across the pipeline

identify faulty components during CI/CD runs

debug remote failures on the server

CI/CD and deployment

The project is fully containerized using Docker.

The pipeline is deployed on an EC2 instance and automatically built and deployed using:

GitHub Actions

AWS ECR

self-hosted runner

Every push triggers:

image build

registry push

deployment on the server

How this project is structured

Your structure actually follows a clean MLOps design.
This is the part you must show clearly in your repo:

src/
 ├── components/
 │     ├── data_ingestion.py
 │     ├── data_validation.py
 │     ├── data_transformation.py
 │     ├── model_trainer.py
 │     ├── model_evaluation.py
 │     └── model_pusher.py
 │
 ├── entity/
 │     ├── config_entity.py
 │     ├── artifact_entity.py
 │     └── s3_estimator.py
 │
 ├── configuration/
 │     ├── mongo_db_connections.py
 │     └── aws_connection.py
 │
 ├── data_access/
 │     └── proj1_data.py
 │
 ├── utils/
 │     └── main_utils.py
 │
 ├── pipeline/
 │     ├── training_pipeline.py
 │     └── prediction_pipeline.py
How notebook code maps to pipeline code

Here is the part you were struggling to explain.

Be very explicit in your README:

Notebook responsibility	Pipeline location
Data loading from DB	data_access/proj1_data.py
EDA understanding	Notebook only (not part of production)
Schema checks	components/data_validation.py
Feature engineering & encoding	components/data_transformation.py
Model fitting	components/model_trainer.py
Metric comparison	components/model_evaluation.py
Model saving	components/model_pusher.py

This is exactly how professionals explain notebook → pipeline conversion.