import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from logger import logging
from exception import customException
from components.data_transformation import dataTransformtion
from components.model_training import modelTraining

@dataclass
class dataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class dataIngestion:
    def __init__(self):
        self.dataIngestionConfig = dataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:

            df = pd.read_csv(os.path.join('notebooks/','data/gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')
            os.makedirs(os.path.dirname(dataIngestionConfig.raw_data_path),exist_ok=True)
            df.to_csv(dataIngestionConfig.raw_data_path,index=False)
            logging.info("raw csv stored successfully")

            #split data
            logging.info("Train test split")
            train,test = train_test_split(df,test_size=0.30,random_state=42)

            # creating dataframe of splited data
            train_data = pd.DataFrame(train)
            test_data = pd.DataFrame(test)

            # storing dataframe in csv format
            train_data.to_csv(dataIngestionConfig.train_data_path,index=False)
            test_data.to_csv(dataIngestionConfig.test_data_path,index=False)
            logging.info("train csv and test csv stored successfully")

            logging.info('Ingestion of Data is completed')

            return(
                self.dataIngestionConfig.train_data_path,
                self.dataIngestionConfig.test_data_path
            )
        
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise customException(e,sys)




# Trial running components

# a = dataIngestion()
# test,train = a.initiate_data_ingestion()
# print(test)
# print(train)
# a = dataTransformtion()

# X_train,y_train,X_test,y_test,preprocessor_path = a.initiate_data_transformation(test,train)
# print(X_train,y_train,X_test,y_test,preprocessor_path)

# a = modelTraining()
# a.initiate_model_training(X_train,y_train,X_test,y_test)
