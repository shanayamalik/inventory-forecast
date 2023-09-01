import click
import pandas as pd
import pickle
import numpy as np
import random
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@click.command()
@click.option('--start_date', prompt='Enter start date in mm-dd-yyyy format', help='The start date for forecasting.')
@click.option('--duration', prompt='Enter duration (number of days)', type=int, help='Duration for which forecast is needed.')
def inventory_forecast(start_date, duration):
    """
    Forecasts the sales for a specified duration starting from the given start date and determines the 
    required inventory level to meet the predicted demand.
    
    Parameters:
    - start_date (str): The start date in 'mm-dd-yyyy' format from which the forecast should begin.
    - duration (int): Duration for which forecast is needed in number of days.

    Returns:
    - None: Prints the required inventory level and the forecasted sales.
    """
    # Load the saved ARIMA model
    model_filename = 'arima_model.pkl'
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load data
    data = pd.read_csv('path_to_dataset.csv', parse_dates=['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)

    # Forecast sales for the specified duration
    start_idx = data.index.get_loc(pd.Timestamp(start_date))
    forecast = model.forecast(steps=duration + start_idx)[-duration:]
    
    # Retrieve the last inventory value from the dataset
    last_inventory = data['Inventory'].iloc[-1]
    
    # Calculate how much inventory is needed to last the entire duration
    total_forecasted_sales = forecast.sum()
    required_inventory = total_forecasted_sales - last_inventory

    click.echo(f"Required Inventory: {required_inventory}")
    click.echo(f"Forecasted Sales:\n{forecast}")

if __name__ == '__main__':
    inventory_forecast()
