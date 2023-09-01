import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess the data
data = pd.read_csv('/path')

# Convert 'Date' to datetime format and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data = data.drop(columns=['Product ID'])

# Split the data
train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_data.drop(columns=['Inventory', 'Date'])
y_train = train_data['Inventory']

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'rf_inventory_model.pkl')
