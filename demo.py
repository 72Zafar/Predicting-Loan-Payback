from src.logger import logging


from src.pipline.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()
training_pipeline.run_pipeline()