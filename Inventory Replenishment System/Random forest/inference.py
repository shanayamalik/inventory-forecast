import pandas as pd
import joblib
import click
import numpy as np
import random
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@click.command()
@click.argument('input_csv', type=str)
@click.option('--start_date', 'S', type=str, required=True, help='Start date for replenishment calculation')
@click.option('--duration', 'D', type=int, required=True, help='Duration for which replenishment is to be calculated')
def main(input_csv, S, D):
    """
    Calculates the recommended restock amount for a given duration, starting from a specified date, 
    using a trained forecasting model.
    
    Parameters:
    - input_csv (str): Path to the CSV file containing inventory data.
    - S (str): Start date for replenishment calculation in format 'YYYY-MM-DD'.
    - D (int): Duration for which replenishment is to be calculated in days.

    Returns:
    - None: Prints the recommended restock amount.
    """
    # Load the trained model
    model = joblib.load('trained_forecasting_model.pkl')

    # Load the CSV file to get the last day's inventory
    data = pd.read_csv(input_csv)
    last_inventory = data['Inventory Level'].iloc[-1]
    
    # Prepare data for the specified range of dates (simplified for this example)
    dates = pd.date_range(start=S, periods=D)
    future_data = pd.DataFrame({'Date': dates})
    
    # Predict sales for the range
    forecasted_sales = model.predict(future_data)
    
    # Calculate total forecasted sales and determine restock amount
    total_forecasted_sales = forecasted_sales.sum()
    restock_amount = max(0, total_forecasted_sales - last_inventory)
    
    # Display the recommended restock amount
    print(f"Recommended restock amount for {D} days starting from {S}: {restock_amount}")

if __name__ == '__main__':
    main()
