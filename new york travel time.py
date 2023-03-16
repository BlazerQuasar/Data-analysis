import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime,date
from dateutil.parser import parse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Read the data
a = pd.read_csv('trip_fare_9.csv')
b = pd.read_csv('trip_data_9.csv')

# Merge the dataframes
c = pd.merge(a,b)

# Drop rows with missing values
c.dropna(inplace=True)

# Remove rows with passenger count greater than 5
c = c.drop(c[c[' passenger_count']>5].index)

# Feature Engineering: Compute distance between pickup and dropoff locations
c['distance'] = np.sqrt((c[' dropoff_longitude'] - c[' pickup_longitude'])**2 + (c[' dropoff_latitude'] - c[' pickup_latitude'])**2)

# Transform duration feature from seconds into hours
c['trip_duration_in_hour'] = c[' trip_time_in_secs'] / 3600

# Transform data into datetime
c['pickup_datetime'] = pd.to_datetime(c[' pickup_datetime'])
c['dropoff_datetime'] = pd.to_datetime(c[' dropoff_datetime'])

# Extract day and hour from pickup datetime
c['pickup_day'] = c['pickup_datetime'].dt.day
c['pickup_hour'] = c['pickup_datetime'].dt.hour

# Transform payment_type into dummy variable
payment_dummy = pd.get_dummies(c[' payment_type'])

# Combine the original data with the dummy variables
newdata = pd.concat([c[['medallion', ' hack_license', ' vendor_id', ' dropoff_longitude', ' dropoff_latitude', 'pickup_day', 'pickup_hour', ' tip_amount', ' passenger_count', ' pickup_longitude', ' pickup_latitude', ' trip_time_in_secs', 'trip_duration_in_hour', 'distance']], payment_dummy], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(newdata.drop(['trip_duration_in_hour'],axis=1), newdata['trip_duration_in_hour'], test_size=0.2, random_state=41)

# Normalize/Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest Regressor using cross-validation
param_grid = {'n_estimators':[50,100,200,400,600,800], 'max_depth':[3,4,5,6,7,8,9,10,11,12,13,14,15,16], 'min_samples_leaf':[1,2,3]}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print('model.best_score_: ', grid_search.best_score_)
print('model.best_params_: ', grid_search.best_params_)

# Train the Random Forest Regressor with the optimized hyperparameters
rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],
                            max_depth=grid_search.best_params_['max_depth'],
                            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                            random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate the model using cross-validation and Root Mean Squared Error
scores = -cross_val_score(rf, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_error')
rmse = np.sqrt(scores.mean())
print("Root Mean Squared Error for Random Forest: {:.3f}".format(rmse))

# Make predictions on the testing set
X_test_scaled = scaler.transform(X_test)
y_pred = rf.predict(X_test_scaled)

# Evaluate the model on the testing set using Root Mean Squared Error
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error on Test Set: {:.3f}".format(rmse_test))

# Error Analysis: Plot the residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()