import pandas as pd
import xgboost as xgb
import joblib
import click
import holidays
from datetime import timedelta
import numpy as np
import random
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@click.command()
@click.option('--date', prompt='Enter the date to predict up to (YYYY-MM-DD)', type=str)
def predict_inventory(date):
    """
    Predicts inventory levels for each day from the last date in the dataset up to 
    a given target date using a trained XGBoost model.

    Parameters:
    - date (str): The target date up to which the inventory predictions are to be made, in the format "YYYY-MM-DD".

    Workflow:
    1. Loading Resources: Loads the trained XGBoost model and label encoder from disk.
    2. Data Preparation: Loads the previous dataset, ensures it's sorted by date, and determines the last date.
    3. Prediction Loop:
        - For each day from the last date in the dataset to the target date:
            a. Generate relevant features for the day, including weekday, whether it's a holiday, and lag features.
            b. Predict the inventory level using the trained model.
            c. Append the prediction to the dataset to assist in generating lag features for subsequent days.
    4. Display: Outputs the predicted inventory for each date.

    Returns:
    None. Prints the predicted inventory levels for each date.
    """
    # Load the trained model and label encoder
    xgb_model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    
    # Load the previous dataset
    dataset = pd.read_csv("Inventory Management/xgboost/dataset/dataset.csv")
    dataset['Date'] = pd.to_datetime(dataset['Date'])

    # Ensure the dataset is sorted by date
    dataset = dataset.sort_values(by='Date')
    
    # Last date in the dataset
    last_date = dataset['Date'].iloc[-1]
    target_date = pd.to_datetime(date)
    
    predictions = []
    
    # Prediction Loop:
    # While the current 'last_date' is less than the specified 'target_date':
    while last_date < target_date:
        # Preprocess the data
        last_date += timedelta(days=1)
        input_data = pd.DataFrame({'Date': [last_date]})
        input_data['weekday'] = input_data['Date'].dt.day_name()
        input_data['holiday'] = input_data['Date'].apply(lambda x: x in holidays.US()).astype(int)
        input_data['weekday'] = le.transform(input_data['weekday'])
        
        # Generate lag features from the previous dataset
        for i in range(1, 8):
            input_data[f'lag_{i}'] = dataset['Inventory'].iloc[-i]
        
        # Predict using the loaded model
        prediction = xgb_model.predict(input_data.drop('Date', axis=1))
        predictions.append((last_date, prediction[0]))
        
        # Append the prediction to the dataset for subsequent lag features
        new_row = {'Date': last_date, 'Inventory': prediction[0]}
        dataset = dataset.append(new_row, ignore_index=True)

    # Display the predictions
    for date, pred in predictions:
        print(f"Predicted inventory for {date.strftime('%Y-%m-%d')}: {pred.astype(int)}")

if __name__ == "__main__":
    predict_inventory()
