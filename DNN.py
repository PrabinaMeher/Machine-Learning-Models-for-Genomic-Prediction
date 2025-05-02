#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings # supress warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[3]:


# Load the dataset
file_name = sys.argv[1]
df = pd.read_csv(file_name, header = None)
X = df.iloc[:, :-1]  # Assuming the last column is the target
y = df.iloc[:, -1]

Result_Dir = 'DNN_Results_' + file_name.rsplit('.', 1)[0]
os.makedirs(Result_Dir, exist_ok=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Setup parameters for cross validation
n_splits = 5
n_repeats = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Prepare DataFrame to store results
all_results = pd.DataFrame()


# In[ ]:


# Perform Monte Carlo cross-validation
for repeat in range(n_repeats):
    iteration_results = {'MSE': [], 'RMSE': [], 'MAPE': [], 'Correlation': []}
    for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the DNN model (Deeper than ANN)
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for regression; no activation function
        ])


        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)  # set verbose=0 for less output during training

        # Predicting the test set results
        y_pred = model.predict(X_test).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        correlation, _ = pearsonr(y_test, y_pred)


        # Store results for this fold
        iteration_results['MSE'].append(mse)
        iteration_results['RMSE'].append(rmse)
        iteration_results['MAPE'].append(mape)
        iteration_results['Correlation'].append(correlation)

    # Convert iteration_results to a DataFrame and save it to a CSV file
    iteration_results_df = pd.DataFrame(iteration_results)
    iteration_results_filename = f"iteration_{repeat+1}.csv"
    iteration_results_df.to_csv(Result_Dir+str("/")+iteration_results_filename, index=False)
    print(f"Iteration {repeat+1} results saved to {iteration_results_filename}")

