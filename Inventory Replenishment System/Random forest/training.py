import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    Adds 'IsHoliday' and 'Weekday' columns to the provided data.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing a 'Date' column.

    Returns:
    - data (pd.DataFrame): DataFrame with added 'IsHoliday' and 'Weekday' columns.
    """
    # Define a simple list of US holidays (month, day)
    
    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Add 'IsHoliday' column: 1 if the date is a holiday, 0 otherwise
    data['IsHoliday'] = data['Date'].apply(lambda x: x in holidays.US()).astype(int)
    
    # Add 'Weekday' column: 0 (Monday) to 6 (Sunday)
    data['Weekday'] = data['Date'].dt.weekday
    
    return data


@click.command()
@click.argument('input_csv', type=str)
def main(input_csv):
    """
    Trains a RandomForestRegressor model on provided data and saves the trained model.
    
    Parameters:
    - input_csv (str): Path to the CSV file containing inventory and sales data.
                      Expected columns: Date,Product ID,Sale Count,Inventory.

    Returns:
    - None: Saves the trained model to 'trained_forecasting_model.pkl'.
    """
    # Load data
    data = pd.read_csv(input_csv)
    data = add_holiday_weekday_columns(data)
    # Assuming the columns are named as mentioned: Date, Product ID, Sale Count, Inventory
    X = data.drop(columns=['Date', 'Product ID', 'Sale Count'])
    
    y = data['Sale Count']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, 'trained_forecasting_model.pkl')

if __name__ == '__main__':
    main()
