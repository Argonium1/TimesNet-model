import statsmodels.api as sm  # Import the statsmodels library for accessing datasets and statistical models
import numpy as np           # Import NumPy for numerical operations
import pandas as pd          # Import Pandas for data manipulation and analysis
import xgboost as xgb        # Import XGBoost, a machine learning library optimized for performance
from sklearn.metrics import mean_squared_error  # Import mean_squared_error to evaluate the model
from matplotlib import pyplot as plt            # Import matplotlib.pyplot for data visualization

# Load the built-in sunspot.year dataset from the statsmodels library
data = sm.datasets.sunspots.load_pandas().data

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Plot the sunspot data over the years to get an initial sense of the trend
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(data['YEAR'], data['SUNACTIVITY'])  # Plot the Year against Sunspot Activity
plt.title('Yearly Sunspot Data')  # Title of the plot
plt.xlabel('Year')  # Label for the X-axis
plt.ylabel('Sunspot Activity')  # Label for the Y-axis
plt.show()

# Determine the split point for train-test datasets, using the last 20 years as the test set
train_size = len(data) - 20

# Split the data into training and test sets
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Output the sizes of the training and test sets
print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")

# Visualize the train-test split on a plot
plt.figure(figsize=(12, 6))
plt.plot(train['YEAR'], train['SUNACTIVITY'], label='Train')  # Plot training data
plt.plot(test['YEAR'], test['SUNACTIVITY'], label='Test')  # Plot test data
plt.title('Train-Test Split for Sunspot Activity')  # Title of the plot
plt.xlabel('Year')  # Label for the X-axis
plt.ylabel('Sunspot Activity')  # Label for the Y-axis
plt.legend()  # Add legend to distinguish between train and test data
plt.show()

# Function to create lagged features (supervised learning approach for time series)
def create_features(data, n_lags):
    df = pd.DataFrame(data)  # Convert input data to DataFrame
    columns = [df.shift(i) for i in range(n_lags + 1)]  # Create shift for lagged variables
    df = pd.concat(columns, axis=1)  # Concatenate columns to form the features
    df.columns = ['target'] + [f'lag_{i}' for i in range(1, n_lags + 1)]  # Rename columns
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Set the number of lags based on domain knowledge or experimentation
n_lags = 4  # Choose an appropriate number of lags
dataset = create_features(data['SUNACTIVITY'].values, n_lags)  # Create lagged features

# Adjust the train and test datasets according to the lagged features
train, test = dataset.iloc[:train_size-n_lags], dataset.iloc[train_size-n_lags:]

# Prepare features (X) and target (y) for both training and test sets
X_train, y_train = train.drop('target', axis=1), train['target']  # Features and target for training
X_test, y_test = test.drop('target', axis=1), test['target']  # Features and target for testing

# Convert datasets to DMatrix format, which is optimized for training with XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for XGBoost model
params = {
    'objective': 'reg:squarederror',  # Define objective function for regression
    'max_depth': 3,  # Maximum tree depth
    'eta': 0.1,  # Learning rate (step size shrinkage)
    'subsample': 0.8,  # Subsample ratio of the training instances
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'seed': 4  # Random seed for reproducibility
}

# Train the XGBoost model with specified parameters
model = xgb.train(params, dtrain, num_boost_round=100)  # Train model for a specified number of boosting rounds

# Make predictions on the test set using the trained model
y_pred = model.predict(dtest)

# Calculate the mean squared error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the actual versus predicted sunspot activity
plt.figure(figsize=(10, 6))
plt.plot(data['YEAR'].iloc[train_size:], y_test, label='Actual')  # Plot actual data
plt.plot(data['YEAR'].iloc[train_size:], y_pred, label='Predicted')  # Plot predicted data
plt.title('XGBoost Predictions for Sunspot Activity')  # Title of the plot
plt.xlabel('Year')  # Label for the X-axis
plt.ylabel('Sunspot Activity')  # Label for the Y-axis
plt.legend()  # Add legend to distinguish between actual and predicted data
plt.show()
