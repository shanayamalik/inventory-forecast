
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
dataset = pd.read_csv("Inventory Replenishment System/xgboost/dataset/dataset.csv")

# Preprocess the data
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['weekday'] = dataset['Date'].dt.day_name()
dataset['holiday'] = dataset['Date'].apply(lambda x: x in holidays.US()).astype(int)
dataset.drop('Inventory', axis=1, inplace=True)

for i in range(1, 8):
    dataset[f'lag_{i}'] = dataset['Sale Count'].shift(i)
dataset.dropna(inplace=True)

le = LabelEncoder()
dataset['weekday'] = le.fit_transform(dataset['weekday'])
X = dataset.drop(['Date', 'Product ID', 'Sale Count'], axis=1)
y = dataset['Sale Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model
early_stop = xgb.callback.EarlyStopping(
    rounds=2, metric_name='logloss', data_name='Validation_0', save_best=True
)
xgb_model = xgb.XGBRegressor(learning_rate=0.001)
xgb_model.fit(X_train, y_train)

# Save the trained model and the label encoder
joblib.dump(xgb_model, "xgb_model_sales.pkl")
joblib.dump(le, "label_encoder_sales.pkl")