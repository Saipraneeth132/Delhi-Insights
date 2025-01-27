import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from pathlib import Path
import logging
import joblib  # Import joblib

# Set up logging
logging.basicConfig(filename='lightgbm_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
try:
    data = pd.read_csv(f"{Path(__file__).parent}/data/overall.csv")
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

# Check for missing values
if data.isnull().sum().any():
    logging.warning("Missing values found. Handling missing values...")
    data.fillna(data.median(), inplace=True)  # Fill missing values with median for numerical columns
    logging.info("Missing values handled.")

# Convert 'Date' to datetime and extract useful features
try:
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    logging.info("Date features extracted successfully.")
except Exception as e:
    logging.error(f"Error processing date features: {e}")
    raise

# Encode categorical variables
categorical_cols = ['TIMESLOT', 'Season', 'DayOfWeek', 'IsHoliday']
try:
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    logging.info("Categorical variables encoded successfully.")
except Exception as e:
    logging.error(f"Error encoding categorical variables: {e}")
    raise

# Define features and target
features = ['TIMESLOT', 'Year', 'Month', 'Day', 'Season', 'DayOfWeek', 'IsHoliday']
X = data[features]
y = data['DELHI']

# Split the data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")
except Exception as e:
    logging.error(f"Error splitting data: {e}")
    raise

# Create LightGBM dataset
try:
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    logging.info("LightGBM dataset created successfully.")
except Exception as e:
    logging.error(f"Error creating LightGBM dataset: {e}")
    raise

# Define LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_bin': 512,  # Increase max_bin for better handling of categorical features
    'max_cat_to_onehot': 100  # Increase max_cat_to_onehot for better categorical handling
}

# Train the model
try:
    model = lgb.train(params, train_data, num_boost_round=1000)
    logging.info("LightGBM model trained successfully.")
except Exception as e:
    logging.error(f"Error training LightGBM model: {e}")
    raise

# Make predictions
try:
    y_pred = model.predict(X_test)
    logging.info("Predictions made successfully.")
except Exception as e:
    logging.error(f"Error making predictions: {e}")
    raise

# Print predictions and actual values
print("Predictions:", y_pred[:10])  # Print first 10 predictions
print("Actual Values:", y_test.values[:10])  # Print first 10 actual values

# Evaluate the model
try:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Model evaluation - MSE: {mse}, R-squared: {r2}")
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
except Exception as e:
    logging.error(f"Error evaluating model: {e}")
    raise

# Feature Importance
try:
    feature_importance = pd.Series(model.feature_importance(), index=features).sort_values(ascending=False)
    logging.info("Feature importance calculated.")
    print("Feature Importance:")
    print(feature_importance)
except Exception as e:
    logging.error(f"Error calculating feature importance: {e}")
    raise

# Save the model using joblib with .pkl extension
try:
    joblib.dump(model, f"{Path(__file__).parent}/model/delhi_lgbm_model.pkl")  # Save the model as .pkl
    logging.info("LightGBM model saved successfully as .pkl using joblib.")
except Exception as e:
    logging.error(f"Error saving model as .pkl using joblib: {e}")
    raise

# Load the model using joblib (example)
try:
    loaded_model = joblib.load('delhi_lgbm_model.pkl')  # Load the model
    logging.info("LightGBM model loaded successfully from .pkl using joblib.")
except Exception as e:
    logging.error(f"Error loading model from .pkl using joblib: {e}")
    raise