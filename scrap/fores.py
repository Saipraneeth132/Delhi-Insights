import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import joblib

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
label_encoders = {}
categorical_cols = ['TIMESLOT', 'Season', 'DayOfWeek', 'IsHoliday']
try:
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    logging.info("Categorical variables encoded successfully.")
except Exception as e:
    logging.error(f"Error encoding categorical variables: {e}")
    raise

print(data.head())
print(data.dtypes)

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

# Train a Random Forest Regressor
try:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Random Forest model trained successfully.")
except Exception as e:
    logging.error(f"Error training model: {e}")
    raise

# Make predictions
try:
    y_pred = model.predict(X_test)
    logging.info("Predictions made successfully.")
except Exception as e:
    logging.error(f"Error making predictions: {e}")
    raise

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
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    logging.info("Feature importance calculated.")
    print("Feature Importance:")
    print(feature_importance)
except Exception as e:
    logging.error(f"Error calculating feature importance: {e}")
    raise

# Visualize actual vs predicted values
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig('actual_vs_predicted.png')
    plt.show()
    logging.info("Actual vs Predicted plot saved and displayed.")
except Exception as e:
    logging.error(f"Error visualizing actual vs predicted values: {e}")
    raise

# Save the model
try:
    joblib.dump(model, f"{Path(__file__).parent}/model/delhi_random_forest_model.pkl")
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")
    raise