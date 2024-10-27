"""
**I. Import necessary libraries**
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

"""
**II. Load the dataset**
"""
file_path = 'datasets/preprocessed.csv'  # Use your file path
data = pd.read_csv(file_path)

"""
**III. Filter the dataset to include data from 2015 and above**
"""
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Date'].dt.year >= 2015]

"""
**IV. Feature Engineering**
"""
data['Exchange rate MA'] = data['Exchange rate'].rolling(window=5).mean()
data['Exchange rate STD'] = data['Exchange rate'].rolling(window=5).std()
data['Exchange rate Lag1'] = data['Exchange rate'].shift(1)
data['Exchange rate Lag2'] = data['Exchange rate'].shift(2)
data['Exchange rate Diff'] = data['Exchange rate'].diff()
data['Exchange rate Log'] = np.log(data['Exchange rate'])
data = data.dropna()

"""
**V. A 'target' column based on the trend** 
"""
data['target'] = (data['Exchange rate'].diff() > 0).astype(int)  # 1 if increase, 0 if decrease
data = data.dropna()

"""
**VI. Inspect the dataset**
"""
print(data.head())
print(data.info())
print(data.isnull().sum())
print("\nAvailable Columns:", data.columns.tolist())  # Convert columns to a list for better readability

"""
**VII. Encode categorical variables**
"""
for col in data.select_dtypes(include='object').columns:
    if col != 'target':
        print(f"Encoding column: {col}")
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

"""
**VIII. Prepare the features (X) and target (y)**
"""
X = data.drop(['target', 'Date'], axis=1)  # Features
y = data['target']                         # Target

"""
**IX. Train-test split**
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
**X. Initialize RandomForestClassifier**
"""
model = RandomForestClassifier(random_state=42, class_weight='balanced')

"""
**XI. Hyperparameter tuning using RandomizedSearchCV**
"""
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

"""
**XII. Get the best model**
"""
best_model = random_search.best_estimator_
print("\nBest parameters found: ", random_search.best_params_)

y_pred = best_model.predict(X_test)

"""
**XIII. Evaluate the model**
"""
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nAccuracy Score:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

"""
**XIV. Perform Cross-Validation**
"""
scores = cross_val_score(best_model, X, y, cv=5)
print("\nCross-validation scores:", scores)
print(f"Mean cross-validation score: {scores.mean():.2f}")

"""
**XV. Save the best model to a .pkl file**
"""
model_file_path = 'random_forest_pkl.pkl'
joblib.dump(best_model, model_file_path)

"""
Confusion Matrix:
[[2159    0]
 [   1 2023]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2159
           1       1.00      1.00      1.00      2024

    accuracy                           1.00      4183
   macro avg       1.00      1.00      1.00      4183
weighted avg       1.00      1.00      1.00      4183


Accuracy Score:
Accuracy: 1.00

Cross-validation scores: [1.         0.99976094 1.         1.         0.99784792]
Mean cross-validation score: 1.00
"""