from sklearn.model_selection import GridSearchCV
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
import warnings
import holidays
import joblib
joblib.parallel_backend('threading')

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'n_estimators': [50, 100, 200, 500],
    'objective': ['reg:squarederror'],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 1.5, 2, 3, 4.5]
}

# Initialize the XGBRegressor
xgb_reg = xgb.XGBRegressor()
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
# dataset = pd.read_csv("dataset.csv")
dataset = pd.read_csv("Inventory Management/xgboost/dataset/dataset.csv")

# Preprocess the data
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['weekday'] = dataset['Date'].dt.day_name()
dataset['holiday'] = dataset['Date'].apply(lambda x: x in holidays.US()).astype(int)
dataset.drop('Sale Count', axis=1, inplace=True)

for i in range(1, 8):
    dataset[f'lag_{i}'] = dataset['Inventory'].shift(i)
dataset.dropna(inplace=True)

le = LabelEncoder()
dataset['weekday'] = le.fit_transform(dataset['weekday'])
X = dataset.drop(['Date', 'Product ID', 'Inventory'], axis=1)
y = dataset['Inventory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
import tqdm
from tqdm import tqdm
from sklearn.metrics import make_scorer, mean_squared_error

# Create a tqdm progress bar
total_combinations = np.prod([len(v) for v in param_grid.values()])
pbar = tqdm(total=total_combinations, position=0, leave=True)

# Custom scorer that updates tqdm progress bar
# def custom_scorer(estimator, X, y):
#     pbar.update(1)
#     y_pred = estimator.predict(X)
#     return -mean_squared_error(y, y_pred)
def custom_scorer(y_true, y_pred):
    pbar.update(1)
    return -mean_squared_error(y_true, y_pred)

# Use the custom scorer in GridSearchCV
grid_search = GridSearchCV(
    xgb_reg, 
    param_grid, 
    scoring=make_scorer(custom_scorer, greater_is_better=False), 
    cv=3,
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
pbar.close()

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model using the best parameters
best_xgb_model = grid_search.best_estimator_
