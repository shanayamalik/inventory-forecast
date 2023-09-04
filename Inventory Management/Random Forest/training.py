import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
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
        
    return data

# Load and preprocess the data
# data = pd.read_csv('inventory forcast/random forest/scripts/dataset.csv')
data = pd.read_csv('dataset.csv')
data=add_holiday_weekday_columns(data)
# Convert 'Date' to datetime format and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data = data.drop(columns=['Product ID'])
# Split the data
train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_data.drop(columns=['Inventory', 'Date','Sale Count'])
y_train = train_data['Inventory']

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
X_train
# Save the trained model
joblib.dump(rf_model, 'rf_inventory_model.pkl')
