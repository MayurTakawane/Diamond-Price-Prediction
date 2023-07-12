import os
import sys
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from dataclasses import dataclass
from exception import customException
from logger import logging
from utils import evalute_model,save_object

@dataclass
class modelTrainingConfig:
    model_trainer_path = os.path.join('artifacts','model.pkl')

class modelTraining:
    def __init__(self):
        self.model_trainer_config = modelTrainingConfig()

    def initiate_model_training(self,X_train,y_train,X_test,y_test):

        try:

            models = {
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet()
            }

            model_report = evalute_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # get best model score form report
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                object=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customException(e,sys)
