# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os
from sklearn.model_selection import RandomizedSearchCV

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'Extra Trees Regressor': ExtraTreesRegressor(),
                'Lightgbm': LGBMRegressor()
            }

            # hyperparameter grids for each model
            rf_params = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }

            et_params = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']}

            lgbm_params = {
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [1, 5, 10]}


            # Create a dictionary of models and their corresponding hyperparameter grids
            model_params = {
                'Random Forest': (RandomForestRegressor(), rf_params),
                'Extra Trees Regressor': (ExtraTreesRegressor(),et_params),
                'Lightgbm': (LGBMRegressor(),lgbm_params),
            }

           


            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # Find the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            

            # Hyperparameter tuning for the best model
            if best_model_name in model_params:
                best_model, best_params = model_params[best_model_name]
                logging.info(f'Started hyper parameter tuning')
                random_search = RandomizedSearchCV(best_model, best_params, scoring='r2', cv=5)
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                best_model_score = random_search.best_score_
                print(f'Best Model after Hyperparameter Tuning, Model Name: {best_model_name}, R2 Score: {best_model_score}')
                logging.info(f'Best Model after Hyperparameter Tuning, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
