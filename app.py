from flask import Flask, render_template, request

import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weathersit = int(request.form['weathersit'])
    temp = float(request.form['temp'])
    atemp = float(request.form['atemp'])
    hum = float(request.form['hum'])
    windspeed = float(request.form['windspeed'])
    season = int(request.form['season'])
    yr = int(request.form['yr'])
    mnth = int(request.form['mnth'])
    hr = int(request.form['hr'])
    weekday = int(request.form['weekday'])
    holiday = int(request.form['holiday'])
    workingday = int(request.form['workingday'])
    
    dict = {'season': season, 'yr': yr, 'mnth': mnth, 'hr': hr, 'holiday': holiday,
            'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit,
            'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed}
    logging.info('Input taken successfully')
    # Converting input to dataframe
    input_df = pd.DataFrame([dict])
   
    # Preprocessing
    train_df = pd.read_csv(os.path.join('artifacts', 'train.csv'))
    logging.info('Loading train data successful')
    target_column_name = 'cnt'
    drop_columns = [target_column_name, 'instant', 'dteday', 'casual', 'registered']
    input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

    # Creating object
    scaler = StandardScaler()

    # Transforming using preprocessor object
    input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
    logging.info(input_feature_train_arr)
    input_df = scaler.transform(input_df)
    logging.info(f'transformed input{input_df}')

    # Loading saved model
    model_path = os.path.join('artifacts', 'model.pkl')
    loaded_model = pickle.load(open(model_path, 'rb'))
    y_pred = loaded_model.predict(input_df)
    logging.info("Prediction successful")
    y_pred = np.round(y_pred)
    
    return render_template('predicted_demand.html', demand=y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
