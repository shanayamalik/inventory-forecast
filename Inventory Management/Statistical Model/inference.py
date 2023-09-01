import pandas as pd
import joblib
import click

@click.command()
@click.option('--date', prompt='Enter the date in mm-dd-yyyy format', help='The date till which inventory prediction is needed.')
def predict_inventory(date):
    # Convert the input date to datetime format
    end_date = pd.to_datetime(date)
    
    # Load the trained model
    model = joblib.load('arima_model.pkl')
    
    # Get the number of days to forecast
    forecast_days = (end_date - model.data.dates[-1]).days
    
    # Make predictions
    forecast = model.get_forecast(steps=forecast_days)
    mean_forecast = forecast.predicted_mean
    
    # Print the predictions
    for date, prediction in zip(pd.date_range(model.data.dates[-1], periods=forecast_days, closed='right'), mean_forecast):
        print(f"Predicted inventory for {date.strftime('%Y-%m-%d')}: {prediction:.2f}")
    
if __name__ == '__main__':
    predict_inventory()
