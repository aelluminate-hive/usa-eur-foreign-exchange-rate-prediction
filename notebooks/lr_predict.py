""" DATASET URL: https://datahub.io/core/exchange-rates#daily """

""" 
**I. Import Libraries**
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

""" 
**II. Constants and Model Loading**
"""
# Constants
preprocessed = 'datasets/preprocessed.csv'
LR_MODEL_DIR = 'models'
LR_MODEL_PATH = os.path.join(LR_MODEL_DIR, 'lr_model.pkl')
country = 'Malaysia'  # Replace with the desired country code

# Load the dataset
fexchange = pd.read_csv(preprocessed)

# Convert 'Date' column to datetime format
fexchange['Date'] = pd.to_datetime(fexchange['Date'])

# Filter data for years 1999 and above
fexchange = fexchange[fexchange['Date'].dt.year >= 1999]

# Filter data for the specified country
country_data = fexchange[fexchange['Country'] == country]

# Set date as index and sort by date
country_data.set_index('Date', inplace=True)
country_data.sort_index(inplace=True)

# Ensure we have a daily frequency and fill missing dates if necessary
country_data = country_data.asfreq('D')

# Handle missing values (forward fill)
country_data['Exchange rate'] = country_data['Exchange rate'].ffill()

# Create lagged features
for lag in range(1, 3):  # Create lagged features for 1 and 2 days
    country_data[f'Lagged_{lag}'] = country_data['Exchange rate'].shift(lag)

# Drop rows with NaN values (due to lagging)
country_data.dropna(inplace=True)

# Load the saved Linear Regression model
model = joblib.load(LR_MODEL_PATH)

""" 
**III. Prediction Function**
"""
def predict_exchange_rate(input_date):
    # Convert input date to datetime
    input_date = pd.to_datetime(input_date)

    # Check if the input date is within the range of the dataset
    if input_date < country_data.index[0]:
        raise ValueError(f"Input date {input_date.strftime('%Y/%m/%d')} is before the dataset range.")

    # If the input date is within the dataset range, use the historical data
    if input_date <= country_data.index[-1]:
        last_row = country_data.loc[input_date - pd.Timedelta(days=1)].copy()
    else:
        # For future dates, iteratively predict the next day's exchange rate
        last_row = country_data.iloc[-1].copy()
        current_date = country_data.index[-1] + pd.Timedelta(days=1)
        while current_date <= input_date:
            # Prepare input for the model using the last available lags
            input_features = last_row[[f'Lagged_{lag}' for lag in range(1, 3)]].values.reshape(1, -1)
            prediction = model.predict(input_features)[0]
            
            # Update last_row with new prediction for next iteration using .loc[]
            last_row.loc[f'Lagged_2'] = last_row['Lagged_1']
            last_row.loc[f'Lagged_1'] = prediction
            
            current_date += pd.Timedelta(days=1)

    # Prepare input for the model using the last available lags
    input_features = last_row[[f'Lagged_{lag}' for lag in range(1, 3)]].values.reshape(1, -1)
    prediction = model.predict(input_features)[0]

    return prediction

""" 
**IV. User Input and Prediction**
"""
if __name__ == "__main__":
    # Ask for user input
    input_date = input("Enter a date in the format YYYY/MM/DD to predict the exchange rate: ")

    try:
        # Predict the exchange rate
        predicted_rate = predict_exchange_rate(input_date)
        print(f"Predicted Exchange Rate for {input_date}: {predicted_rate}")
    except ValueError as e:
        print(e)