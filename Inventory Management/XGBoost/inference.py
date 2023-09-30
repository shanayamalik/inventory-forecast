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

# @click.command()
# @click.option('--date', prompt='Enter the date to predict up to (YYYY-MM-DD)', type=str)
def predict_inventory(date="2023-09-22"):
    dataset = pd.read_csv("Inventory Management/xgboost/dataset/dataset.csv")
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

    while last_date < target_date:
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
