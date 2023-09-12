# Inventory Prediction System

This repository contains tools for predicting inventory levels using the XGBoost algorithm.

## Files

1. `training.py`: Script for training the XGBoost model on historical inventory data.
2. `inference.py`: Script for predicting inventory levels up to a given date using the trained XGBoost model.

## Detailed Overview

### training.py

- **Data Loading**: Loads a dataset from `Inventory Management/xgboost/dataset/dataset.csv`.
  
- **Data Preprocessing**:
  - Date-related features like the day of the week are created.
  - A column indicating whether the day is a national holiday in the US is added.
  - The 'Sale Count' column is removed.
  - Inventory data from previous days (lag features) are integrated to predict the current day's inventory.
  
    **Example of Lag Features**:

    Original Data:
    ```
    | Date       | Inventory |
    |------------|-----------|
    | 2023-09-01 | 100       |
    | 2023-09-02 | 105       |
    | 2023-09-03 | 103       |
    | 2023-09-04 | 104       |
    | 2023-09-05 | 102       |
    | 2023-09-06 | 101       |
    | 2023-09-07 | 108       |
    | 2023-09-08 | 107       |
    | 2023-09-09 | 109       |
    | 2023-09-10 | 110       |
    ```

    After Generating Lag Features:

    ```
    | Date       | Inventory | lag_1 | lag_2 | lag_3 | lag_4 | lag_5 | lag_6 | lag_7 |
    |------------|-----------|-------|-------|-------|-------|-------|-------|-------|
    | 2023-09-08 | 107       | 108   | 101   | 102   | 104   | 103   | 105   | 100   |
    | 2023-09-09 | 109       | 107   | 108   | 101   | 102   | 104   | 103   | 105   |
    | 2023-09-10 | 110       | 109   | 107   | 108   | 101   | 102   | 104   | 103   |
    ```

  - The 'weekday' column is label encoded for model compatibility.

- **Model Training**:
  - The dataset is split, and the XGBoost model is trained.
  - The model and encoder are saved for future use.

### inference.py

- **Dependencies and Setup**:
    - The script starts by importing required libraries such as `pandas`, `xgboost`, `joblib`, and `click`.
    - Seeds for reproducibility are set, and specific warnings are ignored for cleaner output.

- **Model and Data Loading**:
    - The pre-trained XGBoost model and the label encoder are loaded from saved files on the disk.
    - The previous dataset, which is stored in a CSV format, is loaded to make predictions on subsequent dates.

- **Command Line Interface**:
    - The script uses the `click` library to provide a command-line interface.
    - The user is prompted to input a target date using the format `YYYY-MM-DD` when running the script.
    
    ```bash
    python inference.py --date YYYY-MM-DD
    ```

    Replace `YYYY-MM-DD` with the desired end date for prediction.

- **Prediction Logic**:
    - The main logic loops through each day from the last date present in the dataset up to the provided target date.
    - For each day, features are dynamically generated. This includes:
        - Extracting the day of the week.
        - Checking if the date is a national holiday in the US.
        - Leveraging inventory values from the previous 7 days (lag features) for prediction.
    - After making a prediction for a specific date, it's appended to the dataset. This ensures that its value can be used as a lag feature for subsequent predictions.

- **Output**:
    - The script prints the predicted inventory levels for each day to the console. The output will be in the format:

    ```bash
    Predicted inventory for YYYY-MM-DD: <predicted_value_day_1>
    Predicted inventory for YYYY-MM-DD: <predicted_value_day_2>
    ...
    ```
