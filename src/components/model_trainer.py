import os
import sys
from dataclasses import dataclass
import numpy as np # type: ignore
import pandas as pd # type: ignore # type : ignore

from src.utils import evaluate_model, save_object

from catboost import CatBoostRegressor # type: ignore

from sklearn.metrics import r2_score # type: ignore
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from xgboost import XGBRegressor # type: ignore

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_file_path):
        try:
            logging.info(" split trainning and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostRegressor(verbose=False) # type: ignore
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6: #threshold fro model performance
                raise CustomException(" no best model found", sys)
            
            logging.info(" best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
