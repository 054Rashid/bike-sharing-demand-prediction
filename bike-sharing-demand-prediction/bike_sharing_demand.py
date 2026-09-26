"""Bike Sharing Demand Prediction.

Python source exported from the project notebook. The notebook remains the primary reproducible artifact.
"""

# ===== Notebook code cell 1 =====
# --- Advanced ML Project: Bike Sharing Demand Forecasting ---
#
# This project tackles a time-series regression problem using classical ML models.
# The primary challenge is extensive feature engineering from datetime objects.
#
# Workflow:
# 1. Load and parse time-series data.
# 2. Engineer a rich set of features from the 'datetime' column.
# 3. Perform exploratory data analysis (EDA) to visualize temporal patterns.
# 4. Preprocess data and apply a log transform to the skewed target variable.
# 5. Train a powerful gradient boosting model.
# 6. Evaluate the model using the appropriate RMSLE metric.
# 7. Analyze feature importance to understand the key drivers of demand.

# ===== Notebook code cell 2 =====
# 1. setup and library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import calendar

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# ===== Notebook code cell 3 =====
# 2. load and pre-process data
try:
  # we pass 'parse_dates' to automatically convert the datetime column
  df_train = pd.read_csv('train.csv',parse_dates=['datetime'])
  df_test = pd.read_csv('test.csv',parse_dates=['datetime'])
  print("datasets 'train.csv' and 'test.csv' loaded successfully ")
  print(f"Training data shape: {df_train.shape}")
  print(f"Testingg data shape: {df_test.shape}")
except FileNotFoundError:
  print("Error : 'train.csv' or 'test.csv' not found")
  print("Please upload the file from the kaggle competition to your colab enviroment")
  exit()

print("\n --- Data Information ---")
df_train.info()

# ===== Notebook code cell 4 =====
# 3. Advanced feature engineering (the core of the project)
## we will extract several new features form the 'datetime' column.

def create_time_features(df):
  df['year'] = df['datetime'].dt.year
  df['month'] = df['datetime'].dt.month
  df['day'] = df['datetime'].dt.day
  df['hour'] = df['datetime'].dt.hour
  df['day_of_week'] = df['datetime'].dt.dayofweek
  df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])
  return df

print("\n--- Enginerring new features from 'datetime' ---")
df_train = create_time_features(df_train)
df_test  = create_time_features(df_test)
print("New time-based features created")
print("Columns added: 'year','month','day','hour','day_of_week','month_name'")
print(df_train[['datetime','hour','day_of_week','month_name']].head())

# ===== Notebook code cell 5 =====
## 4.exploratory data analysis (EDA) on New Features
print("\n-- Visualizing Temporal Patterns")

# How does demand change by hour of the day?
plt.figure(figsize=(12,6))
sns.boxplot(x='hour',y='count',data=df_train)
plt.title('Demand by Hour of the Day')
plt.show()
# Observation: Clear peaks during morning (7-8 AM) and evening (5-6 PM) commutes.

# How does demand changeby day of the week?
plt.figure(figsize=(12,6))
sns.boxplot(x='day_of_week',y='count',data=df_train)
plt.title('Bike Demand by day of the week')
plt.xticks(ticks=range(7),labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.ylabel('Total Rentals (count)')
plt.show()
# observation : Higher usage during weekdays compared to weekends.

# How does weather impact demand?
plt.figure(figsize=(12,6))
sns.boxplot(x='weather',y='count',data=df_train)
plt.title('Bike Demand by Weather Condition')
plt.xticks(ticks=range(4),labels=['Clear','Mist','Light Rain/snow','Heavy Rain/snow'])
plt.ylabel('Total Rentals (count)')
plt.show()

# ===== Notebook code cell 6 =====
# 5.Preprocessing for modeling
# The target variable 'count' is highly skewed. Models perform better on normally
# distributed targets. We apply a log transform.

df_train['count'] = np.log1p(df_train['count'])

# Define features (X) and target (y)
# We drop 'casual' and 'registered' as they are components of 'count' and not available in the test set.
# We also drop 'datetime' as we've already extracted its information.

features = ['season','holiday','workingday','weather','temp','atemp',
            'humidity','windspeed','year','month','hour','day_of_week']
target = 'count'

X = df_train[features]
y = df_train[target]
# We won't use a random train_test_split here to simulate a real forecasting scenario,
# but for a Kaggle competition, we train on all available training data.

# ===== Notebook code cell 7 =====
# 6. Model Training
# Gradient Boosting is a powerful tree-based model that works very well for tabular,
# time-series regression tasks like this one.
print("\n--- Training Gradient Boosting Regressor ---")
gbr = GradientBoostingRegressor(n_estimators=500,
                                learning_rate=0.05,
                                max_depth=4,
                                random_state=42,
                                loss='squared_error')# Default , but explicit
gbr.fit(X,y)
print("Model Training complete")

# ===== Notebook code cell 8 =====
# --- 7. Evaluation ---
# The competition uses the Root Mean Squared Logarithmic Error (RMSLE).
# Since we log-transformed our target, we can use Root Mean Squared Error on our
# predictions and it will be equivalent to RMSLE on the original scale.

# For demonstration, we'll evaluate on a portion of the training data
# In a real scenario, you'd use a time-based validation set.

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
gbr.fit(X_train,y_train)
y_pred_val = gbr.predict(X_val)

# IMPORTANT :  The model predicts the log of the count. we must convert it back.
y_pred_val_original = np.expm1(y_pred_val)
y_val_original = np.expm1(y_val)

# calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val_original,y_pred_val_original))
print(f"\nModel Performance on validation set: ")
print(f"Root Mean Squared Logarithmic Error (RMLSE): {rmsle:.4f}")

# ===== Notebook code cell 9 =====
# 8. Features Importance
# let's see which features our model found most important
print("\n--- Top 10 Most Important Features ---")
feature_importance = pd.DataFrame({
    'feature' : features,
    'importance' : gbr.feature_importances_
}).sort_values('importance',ascending=False)
print(feature_importance.head(10).to_string(index=False))

# ===== Notebook code cell 10 =====
# 9.Make prediction on Test Data
print("\n-- Making Final Prediction on the test set--")
X_test = df_test[features]
test_predictions_log = gbr.predict(X_test)

# Reverse the log transformation
final_predictions = np.expm1(test_predictions_log)

# Ensure prediction are non-negative
final_predictions[final_predictions < 0] = 0

# Create the submission file for Kaggle
submission = pd.DataFrame({
    'datetime' : df_test['datetime'],
    'count'   : final_predictions
})

submission.to_csv('submission.csv',index=False)
print("\n 'submission.csv' file created successfully!")
print("you can download this file form colab and submit it to the kaggle competition")
print(submission.head())
