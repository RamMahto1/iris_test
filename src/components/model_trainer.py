from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.utils import save_object, evaluate_models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVR": SVR(),
                "Ridge": Ridge(max_iter=5000),
                "Lasso": Lasso(max_iter=5000)
            }
            
            # Define hyperparameters
            params = {
                "Decision Tree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
                "SVR": {"C": [0.1, 1, 10], "epsilon": [0.1, 0.2, 0.5]},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.1, 1.0, 10.0]}
            }
            
            # Evaluate models
            report, best_model_name, best_model, best_score = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )
            
            logging.info("Model evaluation completed")
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_score}")
            
            
            if isinstance(best_model, list):
                best_model = best_model[0]
                
            
            # Save only the single best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")
            
            return report, best_model_name, best_model, best_score
        
        except Exception as e:
            raise CustomException(e, sys)