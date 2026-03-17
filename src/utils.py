# functionalities that are used across the project in a common way are defined in this file. This file is used to avoid code duplication and to keep the code organized.

import os
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import dill # type: ignore
from sklearn.metrics import r2_score # type: ignore

from sklearn.model_selection import GridSearchCV # type: ignore 
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # Tune only if params exist for this model and are non-empty
            model_params = param.get(model_name, {})
            if model_params:
                gs = GridSearchCV(
                    model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )
                gs.fit(X_train, y_train)

                # Use the fitted best estimator from grid search.
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            model_train_pred = best_model.predict(X_train)
            model_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, model_train_pred)
            test_model_score = r2_score(y_test, model_test_pred)

            report[model_name] = test_model_score
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)
