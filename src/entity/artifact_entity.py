from dataclasses import dataclass
import os
from  src.constants import *
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str