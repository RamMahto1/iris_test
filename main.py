from src.logger import logging
from src.exception import CustomException
import sys

from src.components.data_loader import load_iris_data
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation

df = load_iris_data(password='root')

if df is not None:
    print("Data loaded successfully!")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.describe())
else:
    print("Failed to load data")


## step: 2 Data ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
logging.info("data ingestion has completed")

# step: 2 data validation
data_validation = DataValidation(train_data_path,test_data_path)
data_validation.initiate_data_validation()
logging.info("data validation has completed")

## step: 4 data transformation
data_transformer = DataTransformation()
train_arr, test_arr, preprocessor_path, _ = data_transformer.initiate_data_transformation(train_data_path, test_data_path)
logging.info("data transforation has completed")

# step: 5 model trainer
model_trainer = ModelTrainer()
model_trainer.initiate_model_trainer(train_arr,test_arr)
logging.info("model has trained")



# logging.info("logging has started")

# try:
#     a = 0/1
#     logging.info("zero divided by 1")
# except Exception as e:
#     raise CustomException(e,sys)