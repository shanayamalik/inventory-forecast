
# ARIMA Sales Forecasting and Inventory Management

This project provides scripts to train an ARIMA model for sales forecasting and to infer from it to determine inventory requirements. The scripts utilize historical sales data to forecast future sales and subsequently determine the amount of inventory needed to meet the demand for a specified duration.

## Scripts

1. **training.py**: Trains an ARIMA model on the provided dataset and saves the model.
2. **inference.py**: Loads the trained ARIMA model, accepts a start date and duration, and predicts sales. It then calculates the required inventory based on the predictions.

## Setup and Usage

### Requirements:
- Python 3.x
- pandas
- statsmodels
- scikit-learn
- click

To install the required packages, run:
```
pip install pandas statsmodels scikit-learn click
```

### Steps:

1. **Training the Model**:
   - Place your dataset in the project directory.
   - Ensure the dataset contains columns named `Date`, `Sale Count`, and `Inventory`.
   - Run `training.py` to train the ARIMA model:
     ```
     python training.py
     ```

2. **Making Predictions**:
   - Run `inference.py` and provide the required inputs:
     ```
     python inference.py --start_date "mm-dd-yyyy" --duration days
     ```
     Replace `mm-dd-yyyy` with the desired start date and `days` with the desired duration. If you don't provide these values directly, the script will prompt you for them.

## Outputs

- **Trained ARIMA Model**: The trained model will be saved as `arima_model.pkl`.
- **Inference**: The script will display the required inventory and forecasted sales for the specified duration.
