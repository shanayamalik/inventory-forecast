# ARIMA Inventory Forecasting

This repository contains scripts to train an ARIMA model for inventory forecasting and make predictions based on the trained model.

## Files

1. `training.py`: This script performs a grid search to identify the best ARIMA parameters based on the AIC criterion, trains the model with the best parameters, and saves the trained model as `arima_model.pkl`.
2. `inference.py`: This script loads the trained ARIMA model and predicts inventory levels up to a user-specified date.

## Requirements

- Python 3.x
- pandas
- statsmodels
- joblib
- click

## Usage

### Training the Model

To train the ARIMA model and save it:

```
python training.py
```

### Making Predictions

To predict inventory levels up to a specific date:

```
python inference.py --date mm-dd-yyyy
```

## License

MIT License. Feel free to use, modify, and distribute as you see fit.
