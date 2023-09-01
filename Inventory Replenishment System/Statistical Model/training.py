import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import pickle
from itertools import product

# Load dataset
data = pd.read_csv('path_to_dataset.csv', parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)

# Grid search for ARIMA parameters
def arima_grid_search(data, p_values, d_values, q_values):
    """
    Performs a grid search on ARIMA model parameters, selects the best model based on Mean Absolute Error,
    and saves the trained model as a pickle file.
    
    Parameters:
    - data_path (str): Path to the CSV file containing the dataset.
                      The dataset is expected to have a 'Date' column and a 'Sale Count' column.
    - p_values (list): List of p values for ARIMA.
    - d_values (list): List of d values for ARIMA.
    - q_values (list): List of q values for ARIMA.

    Returns:
    - None: Saves the trained best ARIMA model to 'arima_model.pkl'.
    """
    best_score, best_cfg = float('inf'), None
    for p, d, q in product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            mae = evaluate_arima_model(data, order)
            if mae < best_score:
                best_score, best_cfg = mae, order
        except:
            continue
    return best_cfg

def evaluate_arima_model(data, order):
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:]
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mae = mean_absolute_error(test, predictions)
    return mae

p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)
best_order = arima_grid_search(data['Sale Count'], p_values, d_values, q_values)

best_model = ARIMA(data['Sale Count'], order=best_order)
best_model_fit = best_model.fit()
with open('arima_model.pkl', 'wb') as model_file:
    pickle.dump(best_model_fit, model_file)
