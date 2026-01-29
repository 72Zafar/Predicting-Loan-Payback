import sys
from src.exception import MyException
from src.logger import logging
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import proj1Estimator