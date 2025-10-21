import pickle
import os
from src.exception import CustomException
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def save_object(file_path: str, obj):
  
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load a pickle object from file
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
   
    try:
        best_score = -float('inf')
        best_model = None
        best_model_name = None
        report = []

        for model_name, model in models.items():
            logging_msg = f"Evaluating model: {model_name}"
            print(logging_msg)

            param_grid = params.get(model_name, {})
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                # Set the best params to the model
                model.set_params(**gs.best_params_)

            # Train model on full training data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            report.append({
                "Model": model_name,
                "R2_Score": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse
            })

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = model_name

        return report, best_model_name, best_model, best_score

    except Exception as e:
        raise CustomException(e, sys)
