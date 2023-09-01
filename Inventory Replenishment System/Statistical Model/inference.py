import click
import pandas as pd
import pickle

@click.command()
@click.option('--start_date', prompt='Enter start date in mm-dd-yyyy format', help='The start date for forecasting.')
@click.option('--duration', prompt='Enter duration (number of days)', type=int, help='Duration for which forecast is needed.')
def inventory_forecast(start_date, duration):
    """
    This script forecasts the sales and determines the required inventory.
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