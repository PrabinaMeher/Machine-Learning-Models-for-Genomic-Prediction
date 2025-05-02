#!/usr/bin/env python
# coding: utf-8

# # Support Vector Regressor

# In[1]:


import warnings # supress warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[3]:


# Load the dataset
file_name = sys.argv[1]
df = pd.read_csv(file_name, header = None)
X = df.iloc[:, :-1]  # Assuming the last column is the target
y = df.iloc[:, -1]


# In[4]:


Result_Dir = 'SVR_Results_' + file_name.rsplit('.', 1)[0]
os.makedirs(Result_Dir, exist_ok=True)


# In[5]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


# Setup parameters for Monte Carlo cross-validation
n_splits = 5
n_repeats = 5


# In[7]:


# Prepare DataFrame to store results
all_results = pd.DataFrame()


# In[8]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto'],
    'svr__epsilon': [0.1, 0.2, 0.5]
}


# In[9]:


log_file = Result_Dir+'/log.txt'


# In[10]:


# Perform Monte Carlo cross-validation
with open(log_file, 'w') as log:
    for i in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(0, 10000))
        
        iteration_results = {'MSE': [], 'RMSE': [], 'MAPE': [], 'Correlation': []}
        
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Create an SVR model and pipeline
            svr = SVR()
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardize the data
                ('svr', svr)                   # SVR model
            ])
    
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
            # Fit the model using Grid Search to find the best parameters
            grid_search.fit(X_train, y_train)
    
            # Use the best model found by GridSearchCV
            best_model = grid_search.best_estimator_
    
            # Print the best parameters for the current iteration
            log.write(f"Iteration {i + 1}, Best Parameters: {grid_search.best_params_}\n")
            log.write(f"Iteration {i + 1}, Best Score: {grid_search.best_score_}\n")
    
            # Predict on the test set
            y_pred = best_model.predict(X_test)
    
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            correlation, _ = pearsonr(y_test, y_pred)
    
            # Store results
            iteration_results['MSE'].append(mse)
            iteration_results['RMSE'].append(rmse)
            iteration_results['MAPE'].append(mape)
            iteration_results['Correlation'].append(correlation)
            

        
        # Convert iteration_results to a DataFrame and save it to a CSV file
        iteration_results_df = pd.DataFrame(iteration_results)
        iteration_results_filename = f"iteration_{i+1}.csv"
        iteration_results_df.to_csv(Result_Dir+str("/")+iteration_results_filename, index=False)
        print(f"Iteration {i+1} results saved to {iteration_results_filename}")


# In[ ]:


# Calculate the mean of all iterations and append as a new row
final_mean_results = {metric: np.mean(values) for metric, values in all_results.items()}
final_mean_results['Iteration'] = 'Mean_for_All_Repeatations'
all_results = all_results.append(final_mean_results, ignore_index=True)


# In[ ]:


# Save results to CSV
all_results.to_csv(Result_Dir+'/All_Results.csv')


