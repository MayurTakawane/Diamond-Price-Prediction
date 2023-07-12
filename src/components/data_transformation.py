import os
import sys
from exception import customException
from logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts',"preprocessor.pkl")

class dataTransformtion:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()

    # creating data transformation object (preprocessor) object
    def get_data_transformation_object(self):
        
        try: 
            logging.info("Initiating get_data_transformation_object")
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            # Creating pipeline

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('OrdinalEncoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )
            
            # Combining pipeline and creating preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ('numPipeline',num_pipeline,numerical_cols),
                    ('catPipeline',cat_pipeline,categorical_cols)
                ]
            )

            logging.info("preprocessor object created successfully")
            return preprocessor
        
        except Exception as e:
            logging.info("Error occurend in get_data_transformation_object function in data transformation")
            raise customException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:

            logging.info('initiate_data_transformation')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            # independent variables
            X_train = train_df.drop(columns='price')
            X_test = test_df.drop(columns='price')

            # dependent variables
            y_train = train_df['price']
            y_test = test_df['price']

            # Transforming using preprocessor obj
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # Feature scaling   
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            # saving object
            save_object(file_path=dataTransformationConfig.preprocessor_file_path,object=preprocessing_obj)

            return(
                X_train,
                y_train,
                X_test,
                y_test,
                dataTransformationConfig.preprocessor_file_path
            )
        except Exception as e:
            logging.info("Error occurend in initiating data transformation function in data transformation")
            raise customException(e,sys)
