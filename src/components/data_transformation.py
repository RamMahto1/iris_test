from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from src.utils import save_object  
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates preprocessing pipeline for numeric features.
        """
        try:
            numerical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_features)
            ])

            logging.info("Data transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms train and test CSVs: scales numeric features, encodes target column.
        Returns transformed train/test arrays, preprocessor path, and label encoder.
        """
        try:
            # Read CSVs
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully.")

            target_column_name = 'species'

            # Encode target
            le = LabelEncoder()
            y_train_arr = le.fit_transform(train_df[target_column_name])
            y_test_arr = le.transform(test_df[target_column_name])
            
            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
            logging.info("LabelEncoder saved successfully.")

            # Preprocess input features
            preprocessing_obj = self.get_data_transformer_object()
            input_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_test_df = test_df.drop(columns=[target_column_name], axis=1)

            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            # Combine features + target
            train_arr = np.c_[input_train_arr, y_train_arr]
            test_arr = np.c_[input_test_arr, y_test_arr]

            logging.info(f"Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor object saved successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path, le

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)
