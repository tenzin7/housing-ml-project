# Import necessary libraries
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Path of the file to read
file_path = 'Data/train.csv'
home_data = pd.read_csv(file_path)

#prediction target
y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Splitting data
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# Selecting model
home_model = RandomForestRegressor(random_state = 1)

# Fit the model
home_model.fit(train_X,train_y)

# Make Predictions
sale_predictions = home_model.predict(val_X)

# Print mean absolutre error
print(mean_absolute_error(val_y, sale_predictions))