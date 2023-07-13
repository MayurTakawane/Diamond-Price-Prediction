import os
import sys
from exception import customException
from logger import logging
import pickle
from sklearn.metrics import r2_score

def save_object(file_path,object):
    try:
        # create directory
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        # dump object as pickle
        with open(file_path,"wb") as f:
            pickle.dump(object,f)

    except Exception as e:
        logging("Error ocurred in utils, function --> save object")
        raise customException
    
def evalute_model(X_train,y_train,X_test,y_test,models):
    
    try:

        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = test_model_score

            return report
    
    except Exception as e:
        logging.info("Error occured in evalute_model function in utils")
        raise customException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info("Error occured in load_object function in utils")
        raise customException(e,sys)