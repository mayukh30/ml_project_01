# for transforming the data and making it ready for modeling
# read code from data ingestion and do the transformation

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# column tranformer is used to create pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            # `math_score` is the training target, so only keep feature columns here.
            numerical_columns = ['reading_score','writing_score']
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #imputer means to fill the missing values with some strategy like mean, median, most_frequent etc
            #scaler is used to scale the data to a standard normal distribution with mean 0 and standard deviation 1

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            #one hot encoder is used to convert categorical variables into a format that can be provided to ML algorithms to do a better job in prediction
            logging.info("numerical columns scaling completed : {numerical_columns}")
            logging.info(" categorical columns encoding completed: {categorical_columns}")

            Preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return Preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)