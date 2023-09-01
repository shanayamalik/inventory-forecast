import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
def main(input_csv):
    """
    Trains a RandomForestRegressor model on provided data and saves the trained model.
    
    Parameters:
    - input_csv (str): Path to the CSV file containing inventory and sales data.
                      Expected columns: Date, Product, Daily Sales, Inventory Level.

    Returns:
    - None: Saves the trained model to 'trained_forecasting_model.pkl'.
    """
    # Load data
    data = pd.read_csv(input_csv)
    
    # Assuming the columns are named as mentioned: Date, Product, Daily Sales, Inventory Level
    X = data.drop(columns=['Date', 'Product', 'Daily Sales'])
    y = data['Daily Sales']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, 'trained_forecasting_model.pkl')

if __name__ == '__main__':
    main()
