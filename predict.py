from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)
CORS(app)

@app.route("/predict_covid/", methods=["GET"])

def forecaster():
    
    #receive data from front-end
    horizon = request.args.get("horizon")
    x = request.args.get("x")
    y = request.args.get("y")
   
    #prepare for prediction
    train_dataset = pd.DataFrame()
    train_dataset["ds"] = pd.to_datetime(json.loads(x))
    train_dataset["y"] = json.loads(y)

    #train model
    prophet_basic = Prophet()
    prophet_basic.fit(train_dataset) 

    # make prediction
    future = prophet_basic.make_future_dataframe(periods=int(horizon))
    forecast = prophet_basic.predict(future)

    forecast.to_csv("forecast.csv")

    return(jsonify(forecast[["ds", "trend"]].to_json(orient = "records")))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
