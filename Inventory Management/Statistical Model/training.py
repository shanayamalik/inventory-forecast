import pandas as pd
import itertools
from statsmodels.tsa.arima.model import ARIMA
import joblib
import numpy as np
import random
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data
data = pd.read_csv("/data")
data['Date'] = pd.to_datetime(data['Date'])

# Define the p, d and q parameters to take any value between 0 and 3
p = d = q = range(0, 4)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Store AIC values
aic_dict = {}

# Grid search for the best ARIMA parameters
for param in pdq:
    try:
        model_arima = ARIMA(data['Inventory'], order=param)
        arima_result = model_arima.fit()
        aic_dict[param] = arima_result.aic
    except:
        continue

# Get the parameters with the lowest AIC value
best_params = min(aic_dict, key=aic_dict.get)

# Fit the ARIMA model with the best parameters
best_model = ARIMA(data['Inventory'], order=best_params)
best_result = best_model.fit()

# Save the model to disk
joblib.dump(best_result, 'arima_model.pkl')
print(f"Model ARIMA{best_params} saved to arima_model.pkl")
