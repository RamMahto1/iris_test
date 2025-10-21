from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    train_data_path: str
    test_data_path: str


class DataValidation:
    def __init__(self, train_data: str, test_data: str):
        self.config = DataValidationConfig(train_data, test_data)

    def initiate_data_validation(self):
        try:
            # Read train and test data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info("Read train and test data for validation")

            # Check missing values
            if train_df.isnull().sum().any():
                logging.warning(" Missing values found in training data")
            else:
                logging.info(" No missing values in training data")

            if test_df.isnull().sum().any():
                logging.warning(" Missing values found in testing data")
            else:
                logging.info(" No missing values in testing data")

            # Check column consistency
            if not set(train_df.columns) == set(test_df.columns):
                raise CustomException("Training and testing data have different columns", sys)
            else:
                logging.info(" Training and testing data have consistent columns")

            # Optional: check datatypes
            if not all(train_df.dtypes == test_df.dtypes):
                logging.warning(" Column data types differ between train and test sets")
            else:
                logging.info("Column data types are consistent")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)