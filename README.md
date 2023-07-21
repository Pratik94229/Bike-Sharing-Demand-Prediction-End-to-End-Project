# Bike-sharing-demand-prediction

Overview

This project aims to develop a predictive model to forecast the bike rental count based on various features such as date, weather conditions, and time of the day. The project utilizes hour datasets." The "day" dataset contains daily aggregated bike rental information, while the "hour" dataset provides hourly details.
Dataset Description
Day Dataset

## Folder structure

```
├───artifacts
├───documents
│   └───Project Report
├───logs
├───notebook
│   └───data
├───src
│   ├───components
│   ├───pipeline
├───static
├───templates
```
- `notebook/`: Jupyter notebooks for data exploration, preprocessing, and model development.
- `notebook/data/`: contains the dataset file(s).
- `src/`: Source code for the project, including preprocessing functions and model training .
- `src/pipeline/`:Source code for different implemented pipelines.
- `artifacts/`: Directory to store model and evaluation results and perform predictions.
- `static/` and `templates/` contains basic frontend framework for deployment using flask.

## The "day" dataset consists of the following columns:

    - instant: A unique identifier for each record
    - dteday: Date of the record
    - season: Season of the year (1 = spring, 2 = summer, 3 = fall, 4 = winter)
    - yr: Year (0 = 2011, 1 = 2012)
    - mnth: Month (1 to 12)
    - holiday: Binary flag indicating if it is a holiday (1 = yes, 0 = no)
    - weekday: Day of the week (0 to 6, where 0 = Sunday)
    - workingday: Binary flag indicating if it is a working day (1 = yes, 0 = no)
    - weathersit: Weather situation (1 = clear, 2 = mist/cloudy, 3 = light rain/snow, 4 = heavy rain/snow)
    - temp: Normalized temperature in Celsius
    - atemp: Normalized feeling temperature in Celsius
    - hum: Normalized humidity
    - windspeed: Normalized wind speed
    - casual: Count of casual bike rentals
    - registered: Count of registered bike rentals
    - cnt: Total count of bike rentals (casual + registered)

## Hour Dataset

The "hour" dataset contains the following columns:

    - instant: A unique identifier for each record
    - dteday: Date of the record
    - season: Season of the year (1 = spring, 2 = summer, 3 = fall, 4 = winter)
    - yr: Year (0 = 2011, 1 = 2012)
    - mnth: Month (1 to 12)
    - hr: Hour of the day (0 to 23)
    - holiday: Binary flag indicating if it is a holiday (1 = yes, 0 = no)
    - weekday: Day of the week (0 to 6, where 0 = Sunday)
    - workingday: Binary flag indicating if it is a working day (1 = yes, 0 = no)
    - weathersit: Weather situation (1 = clear, 2 = mist/cloudy, 3 = light rain/snow, 4 = heavy rain/snow)
    - temp: Normalized temperature in Celsius
    - atemp: Normalized feeling temperature in Celsius
    - hum: Normalized humidity
    - windspeed: Normalized wind speed
    - casual: Count of casual bike rentals
    - registered: Count of registered bike rentals
    - cnt: Total count of bike rentals (casual + registered)
    
To run the project locally, please ensure you have the following dependencies installed:

- Python 3.7 or higher
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook 
- ipykernel
- streamlit

Once you have the dependencies, follow these steps to set up the project:

1. Clone the repository: `git clone https://github.com/Pratik.94229/Bike-sharing-demand-prediction.git`
3. Navigate to the project directory: `cd Bike-sharing-demand-prediction`
4. Create a virtual environment (optional): `conda create -p venv python==3.8`
5. Activate the virtual environment (optional): `activate venv/`
6. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Launch VSCODE
2. Open the `Model Training.ipynb` notebook.
3. Run the cells in the notebook to execute the code and see the results.
4. Feel free to modify the code or experiment with different models and parameters.

## Results

The results of the bike sharing demand prediction are evaluated based on various metrics such as mean absolute error (MAE), root mean squared error (RMSE), and R-squared score. These metrics provide insights into the performance of the model and how well it predicts bike sharing demand.

## Model Building and Selection

To predict the bike rental count, several machine learning models were implemented and evaluated. The following algorithms were utilized:

    Linear Regression
    Random Forest
    Extra Trees Regressor
    LightGBM
    XGBoost

After training and evaluating these models, XGBoost was chosen as the final model due to its superior performance in terms of accuracy and predictive power.
Model Deployment

The selected XGBoost model was deployed using Streamlit, a Python library for building interactive web applications. The deployment allows users to input the relevant features such as date, weather conditions, and time, and obtain the predicted bike rental count as the output.

Deployment Link- https://pratik94229-bike-sharing-demand-p-srccomponentsstreamlit-4ksiv8.streamlit.app/

To use the deployed model using flask, follow these steps:

    1) Install the required dependencies by running pip install -r requirements.txt(make sure to remove commented e .).
    2) First train the model using command python src/pipeline/training_pipeline.py which creates model. 
    3) Then run the Flask application using the command python app.py
    4) Access the application in your web browser at the provided URL.
    5) Enter the required input features such as date, weather conditions, and time etc.
    6) Click the "Predict" button to obtain the predicted bike rental count.

To use the deployed model using streamlit, follow these steps:

    1) Install the required dependencies by running pip install -r requirements.txt(make sure to remove commented e .).
    2) First train the model using command python src/pipeline/training_pipeline.py which creates model. 
    3) Then run the Streamlit application using the command streamlit run src/components/streamlit.py.
    4) Access the application in your web browser at the provided URL.
    5) Enter the required input features such as date, weather conditions, and time etc.
    6) Click the "Predict" button to obtain the predicted bike rental count.

Please note that the accuracy of the predictions may vary based on the input data and model performance.
## Conclusion

This project demonstrates the application of machine learning models in predicting bike rental counts. By leveraging the power of LightGBM and utilizing various input features, the model achieves accurate predictions, allowing users to make informed decisions related to bike rental management.

The Streamlit deployment provides a convenient and interactive way to access the model's predictions, making it user-friendly and accessible to a wide range of users.

For further details, code implementation, and analysis, please refer to the code repository.
