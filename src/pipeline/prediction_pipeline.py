import os
import sys
import pandas as pd
from src.exception import customException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass

@dataclass
class predictionPipelineConfig:
    preprocessor_object_path = os.path.join('artifacts','preprocessor.pkl')
    model_object_path = os.path.join('artifacts','model.pkl')


class predictPipeline:
    def __init__(self):
        self.predict_pipeline_config = predictionPipelineConfig()

    def prediction(self,new_data_X_test):
        try:
            logging.info(f"New data : \n{new_data_X_test}")
            preprocessor = load_object(os.path.join('artifacts','preprocessor.pkl'))
            model = load_object(os.path.join('artifacts','model.pkl'))

            data_transform = preprocessor.transform(new_data_X_test)
            pred = model.predict(data_transform)
            return pred
        except Exception as e:
            logging.info("Error occured in predicton function in predictPipeline class in prediction_pipeline.py")
            raise customException(e,sys)
        
class customData:
    def __init__(self,carat:float,cut:str,color:str,clarity:str,depth:float,table:float,x:float,y:float,z:float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def convert_data_into_dataframe(self):
        try:
            data_dict = {
                'carat' : [self.carat],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity],
                'depth' : [self.depth],
                'table' : [self.table],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z]
            }

            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            logging.info("Error occured in converting dataframe in prediction_pipeline.py")


