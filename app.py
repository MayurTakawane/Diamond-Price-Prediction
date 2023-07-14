from src.pipeline.prediction_pipeline import customData,predictPipeline
from src.logger import logging
from src.exception import customException
from flask import Flask,render_template,request
import numpy as np

app = Flask(__name__)

@app.route('/')
def homePage():
    logging.info("Loading home page")
    return render_template('index.html')

@app.route("/predictPricePage",methods = ["POST"])
def predict_datapoint():
    logging.info("Loading form")
    return render_template('form.html')

@app.route("/results", methods = ["POST"])
def get_results():
    logging.info("Loading reults page")
    logging.info("Gathering form data")
    data=customData(
            carat=float(request.form.get('carat')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity'),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z'))
        )

    logging.info("Converting data to dataframe")
    dataframe = data.convert_data_into_dataframe() ## wasted time for tuple error(there was no need of ',' in self.carat = carat,)
    logging.info("prediction processing")
    pred = predictPipeline()
    results = pred.prediction(dataframe)
    results = np.round(results,decimals=2)
    logging.info("prediction successful returning results")
    return render_template('results.html',results = results)

if __name__ == '__main__':
    logging.info("Starting app")
    logging.info("Prediction pipeline begins")
    app.run(host='0.0.0.0',port=5000)