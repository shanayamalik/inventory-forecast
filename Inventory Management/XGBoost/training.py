import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
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

# Load the dataset
dataset = pd.read_csv("Inventory Management/xgboost/dataset/dataset.csv")

# Preprocess the data
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['weekday'] = dataset['Date'].dt.day_name()
dataset['holiday'] = dataset['Date'].apply(lambda x: x in holidays.US()).astype(int)
dataset.drop('Sale Count', axis=1, inplace=True)

# Create lag features to capture historical inventory values
for i in range(1, 8):
    dataset[f'lag_{i}'] = dataset['Inventory'].shift(i)
dataset.dropna(inplace=True)

# Convert weekday names to numerical labels for model training
le = LabelEncoder()
dataset['weekday'] = le.fit_transform(dataset['weekday'])

# Split the data into features (X) and target (y)
X = dataset.drop(['Date', 'Product ID', 'Inventory'], axis=1)
y = dataset['Inventory']

# Split the dataset into training and test sets without shuffling (keeping the time series order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Save the trained model and the label encoder
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(le, "label_encoder.pkl")
