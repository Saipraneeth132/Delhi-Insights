import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path


# Load the LightGBM model using joblib
model = joblib.load(f"{Path(__file__).parent}/model/delhi_lgbm_model.pkl")

# Define the date and timeslots
date = datetime(2025, 1, 18)  # Example date: October 15, 2023
timeslots = pd.date_range(start=date, end=date + pd.Timedelta(days=1), freq='5T')[:-1]

# Prepare the input data
data = {
    'TIMESLOT': timeslots.strftime('%H:%M').tolist(),
    'Year': timeslots.year.tolist(),
    'Month': timeslots.month.tolist(),
    'Day': timeslots.day.tolist(),
    'Season': [4] * len(timeslots),  # Example: Season 4 (you need to adjust this based on your data)
    'DayOfWeek': timeslots.dayofweek.tolist(),  # Monday=0, Sunday=6
    'IsHoliday': [0] * len(timeslots)  # Example: Not a holiday (0)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Mark categorical features
categorical_features = ['TIMESLOT', 'Season', 'DayOfWeek', 'IsHoliday']  # Adjust based on your model
for feature in categorical_features:
    df[feature] = df[feature].astype('category')

# Make predictions
predictions = model.predict(df)

# Add predictions to the DataFrame
df['Prediction'] = predictions

# Display the results
print(df)