import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import random
import warnings
import holidays
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
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

for i in range(1, 8):
    dataset[f'lag_{i}'] = dataset['Inventory'].shift(i)
dataset.dropna(inplace=True)
# print(dataset)
le = LabelEncoder()
dataset['weekday'] = le.fit_transform(dataset['weekday'])
X = dataset.drop(['Date', 'Product ID', 'Inventory'], axis=1)
y = dataset['Inventory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model
# xgb_model = xgb.XGBRegressor(colsample_bytree= 0.6, gamma= 0.4, learning_rate= 0.01,max_depth= 3, min_child_weight= 1, n_estimators= 50,objective="reg:squarederror", reg_alpha= 1, reg_lambda= 4.5, subsample=0.6)
xgb_model=xgb.XGBRegressor(n_estimators=250)
xgb_model.fit(X, y)
Ada_model = AdaBoostRegressor(n_estimators=200,learning_rate=0.01,loss='square')
Ada_model.fit(X_train, y_train)

# Save the trained model and the label encoder 
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(Ada_model, "Ada_model.pkl")

joblib.dump(le, "label_encoder.pkl")

# Predict using the custom function

# Now use the custom prediction function
 


