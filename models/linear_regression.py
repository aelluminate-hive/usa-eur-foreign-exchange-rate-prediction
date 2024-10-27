""" DATASET URL: https://datahub.io/core/exchange-rates#daily """

""" 
**I. Import Libraries**
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

""" 
**II. Exploratory Data Analysis**
"""
# Constants
preprocessed = 'datasets/preprocessed.csv'
LR_MODEL_DIR = 'models'
LR_MODEL_PATH = os.path.join(LR_MODEL_DIR, 'lr_model.pkl')
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

# Ensure we have a daily frequency and fill missing dates if necessary
country_data = country_data.asfreq('D')

# Handle missing values (forward fill)
country_data['Exchange rate'] = country_data['Exchange rate'].ffill()

# Create lagged features
for lag in range(1, 3):  # Create lagged features for 1 and 2 days
    country_data[f'Lagged_{lag}'] = country_data['Exchange rate'].shift(lag)

# Drop rows with NaN values (due to lagging)
country_data.dropna(inplace=True)

# Prepare features and target variable
X = country_data[[f'Lagged_{lag}' for lag in range(1, 3)]]
y = country_data['Exchange rate']

# Check if there are any samples left
if X.shape[0] == 0:
    raise ValueError("No samples left after preprocessing. Check your data filtering and preprocessing steps.")

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Forecast for the next 30 days
last_row = country_data.iloc[-1].copy()  # Use .copy() to avoid SettingWithCopyWarning
predictions = []
for _ in range(30):
    # Prepare input for the model using the last available lags
    input_features = last_row[[f'Lagged_{lag}' for lag in range(1, 3)]].values.reshape(1, -1)
    prediction = model.predict(input_features)[0]
    
    # Append prediction to results
    predictions.append(prediction)
    
    # Update last_row with new prediction for next iteration using .loc[]
    last_row.loc[f'Lagged_2'] = last_row['Lagged_1']
    last_row.loc[f'Lagged_1'] = prediction

# Create a DataFrame for predictions
forecast_index = pd.date_range(start=country_data.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Predicted Exchange Rate': predictions
})

# Output results (optional)
print(forecast_df)

# Ensure the models directory exists
if not os.path.exists(LR_MODEL_DIR):
    os.makedirs(LR_MODEL_DIR)

# Save the model (if needed)
joblib.dump(model, LR_MODEL_PATH)

"""
**IV. Model Evaluation**
"""
# Model Evaluation
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Mean Absolute Percentage Error (MAPE) to check accuracy of linear regression model
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"Cross-validated RMSE scores: {cv_rmse_scores}")
print(f"Mean cross-validated RMSE: {cv_rmse_scores.mean()}")

"""
**V. Model Evaluation Results**

Mean Absolute Error (MAE): 0.006859700050566238
Mean Squared Error (MSE): 0.00015997345659921633
Root Mean Squared Error (RMSE): 0.01264806137711295
R² Score: 0.9987729789719144
Mean Absolute Percentage Error (MAPE): 0.19448911766148128%
Cross-validated RMSE scores: [0.00247744 0.01420101 0.01544178 0.01438639 0.02002324]
Mean cross-validated RMSE: 0.013305973852264629

"""