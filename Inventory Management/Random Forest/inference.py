import pandas as pd
import joblib
import click
import numpy as np
import random
import warnings
import holidays

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_holiday_weekday_columns(data):
    """
    Adds 'IsHoliday' column to the provided data.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing a 'Date' column.

    Returns:
    - data (pd.DataFrame): DataFrame with added 'IsHoliday' column.
    """
    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Add 'IsHoliday' column: 1 if the date is a holiday, 0 otherwise
    data['IsHoliday'] = data['Date'].apply(lambda x: x in holidays.US()).astype(int)
        
    return data

@click.command()
@click.option('--end_date', prompt='Enter the end date for prediction (MM-DD-YYYY)', help='The end date for which inventory should be predicted.')
def predict_inventory(end_date):
    """
    Predicts inventory up to a specified end date using a trained Random Forest model.
    
    Parameters:
    - end_date (str): The end date in 'MM-DD-YYYY' format up to which inventory prediction is needed.

    Returns:
    - None: Prints the predicted inventory for each day up to the specified end date.
    """
    # Load the trained model
    rf_model = joblib.load('rf_inventory_model.pkl')

    # Convert the user-provided date to the correct format
    end_date_formatted = pd.to_datetime(end_date, format='%m-%d-%Y')

    # Predict the inventory till the specified end date
    data = pd.read_csv('inventory forcast/random forest/scripts/dataset.csv')
    last_date = pd.to_datetime(data['Date'].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=end_date_formatted)
    future_data = pd.DataFrame({
        'Date': future_dates
    })
    future_data = add_holiday_weekday_columns(future_data)
    future_data['Day_of_Week'] = future_data['Date'].dt.dayofweek
    future_data['Month'] = future_data['Date'].dt.month
    future_data['Year'] = future_data['Date'].dt.year
    X_future = future_data.drop(columns=['Date'])
    future_inventory_predictions = rf_model.predict(X_future)

    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Inventory': future_inventory_predictions
    })

    print(future_predictions_df)

if __name__ == '__main__':
    predict_inventory()
