import pandas as pd
import joblib
import click

@click.command()
@click.option('--end_date', prompt='Enter the end date for prediction (MM-DD-YYYY)', help='The end date for which inventory should be predicted.')
def predict_inventory(end_date):
    """
    Predicts inventory up to a specified end date using a trained Random Forest model.
    
    Parameters:
    - end_date (str): The end date in 'MM-DD-YYYY' format up to which inventory prediction is needed.

    Returns:
    - None: Prints the predicted inventory for each day up to the specified end date.
    """
    # Load the trained model
    rf_model = joblib.load('rf_inventory_model.pkl')

    # Convert the user-provided date to the correct format
    end_date_formatted = pd.to_datetime(end_date, format='%m-%d-%Y')

    # Predict the inventory till the specified end date
    data = pd.read_csv('/path')
    last_date = pd.to_datetime(data['Date'].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=end_date_formatted)
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Sale Count': data['Sale Count'].mean()  # Using average sale count as placeholder
    })
    future_data['Day_of_Week'] = future_data['Date'].dt.dayofweek
    future_data['Month'] = future_data['Date'].dt.month
    future_data['Year'] = future_data['Date'].dt.year
    X_future = future_data.drop(columns=['Date'])
    future_inventory_predictions = rf_model.predict(X_future)

    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Inventory': future_inventory_predictions
    })

    print(future_predictions_df)

if __name__ == '__main__':
    predict_inventory()
