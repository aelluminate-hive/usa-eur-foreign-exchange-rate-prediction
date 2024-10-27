""" DATASET URL: https://datahub.io/core/exchange-rates#daily """

""" 
**I. Import Libraries**
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

""" 
**II. Exploratory Data Analysis**
"""
# Constants
preprocessed = 'datasets/preprocessed.csv'
LSTM_MODEL_DIR = 'models'
LSTM_MODEL_PATH = os.path.join(LSTM_MODEL_DIR, 'lstm.pkl')
country = 'Malaysia'  # Replace with the desired country code


# Load the dataset
fexchange = pd.read_csv(preprocessed)

# Display first five(5) rows
print(fexchange.head())


""" 
**III. Model Development**
"""
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

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(country_data[['Exchange rate']].values)

# Define time step (30 days)
time_step = 30

# Create the dataset
X, y = create_dataset(scaled_data, time_step)

# LSTM input reshaping
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping])

# Predict 30 days
last_sequence = scaled_data[-time_step:]
predictions = []
for _ in range(30):

    # Prepare input for the model using the last available sequence
    input_features = last_sequence.reshape(1, time_step, 1)
    prediction = model.predict(input_features)
    
    # Append prediction to results
    predictions.append(prediction[0, 0])
    
    # Update last_sequence with new prediction
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

# Inverse transform the predictions to get the actual values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create a DataFrame for predictions
forecast_index = pd.date_range(start=country_data.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Predicted Exchange Rate': predictions.flatten()
})

# Output results
print(forecast_df)

# Ensure the models directory exists
if not os.path.exists(LSTM_MODEL_DIR):
    os.makedirs(LSTM_MODEL_DIR)

# Save the model (if needed)
joblib.dump(model, LSTM_MODEL_PATH)

"""
**IV. Model Evaluation**
"""
# Model Evaluation
y_true = country_data['Exchange rate'].values[time_step + 1:]
y_pred = scaler.inverse_transform(model.predict(X)).flatten()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

"""
**V. Model Evaluation Results**

Mean Absolute Error (MAE): 0.014612431770762945
Mean Squared Error (MSE): 0.0003450247077195674
Root Mean Squared Error (RMSE): 0.01857484071855173
R² Score: 0.997369383041952
Mean Absolute Percentage Error (MAPE): 0.40180434230250156%

"""