from flask import Flask, render_template, request
import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('predicted_demand.html', demand=None)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    dict = {
        'season': int(request.form['season']),
        'yr': int(request.form['yr']),
        'mnth': int(request.form['mnth']),
        'hr': int(request.form['hr']),
        'holiday': int(request.form['holiday']),
        'weekday': int(request.form['weekday']),
        'workingday': int(request.form['workingday']),
        'weathersit': int(request.form['weathersit']),
        'temp': float(request.form['temp']),
        'atemp': float(request.form['atemp']),
        'hum': float(request.form['hum']),
        'windspeed': float(request.form['windspeed'])
    }

    input_df = pd.DataFrame([dict])

    train_df = pd.read_csv(os.path.join('artifacts', 'train.csv'))

    target_column_name = 'cnt'
    drop_columns = [target_column_name, 'instant', 'dteday', 'casual', 'registered']

    input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

    scaler = StandardScaler()
    input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
    input_feature_test_arr = scaler.transform(input_df)

    model_path = os.path.join("artifacts", "model.pkl")
    loaded_model = pickle.load(open(model_path, 'rb'))
    y_preds = loaded_model.predict(input_df)

    return render_template('predicted_demand.html', demand=round(y_preds[0], 0))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
