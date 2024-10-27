""" DATASET URL: https://datahub.io/core/exchange-rates#daily """

""" 
**I. Import Libraries**
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

""" 
**II. Constants and Model Loading**
"""
# Constants
preprocessed = 'datasets/preprocessed.csv'
LSTM_MODEL_DIR = 'models'
LSTM_MODEL_PATH = os.path.join(LSTM_MODEL_DIR, 'lstm.pkl')
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

# Fill-in missing data for days
country_data = country_data.asfreq('D')

# Handle missing values 
country_data['Exchange rate'] = country_data['Exchange rate'].ffill()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(country_data[['Exchange rate']].values)

# Define time step (30 days)
time_step = 30

# Load the saved LSTM model
model = joblib.load(LSTM_MODEL_PATH)

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
        last_sequence = scaled_data[country_data.index.get_loc(input_date) - time_step:country_data.index.get_loc(input_date)]
    else:
        # For future dates, iteratively predict the next day's exchange rate
        last_sequence = scaled_data[-time_step:]
        current_date = country_data.index[-1] + pd.Timedelta(days=1)
        while current_date <= input_date:
            # Prepare input for the model using the last available sequence
            input_features = last_sequence.reshape(1, time_step, 1)
            prediction = model.predict(input_features)
            
            # Append prediction to results
            last_sequence = np.append(last_sequence[1:], prediction, axis=0)
            
            current_date += pd.Timedelta(days=1)

    # Prepare input for the model using the last available sequence
    input_features = last_sequence.reshape(1, time_step, 1)
    prediction = model.predict(input_features)

    # Inverse transform the prediction to get the actual value
    prediction = scaler.inverse_transform(prediction)

    return prediction[0, 0]

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