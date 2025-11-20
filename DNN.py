import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

# Disable GPU for CPU-only training
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Input / Output paths
feature_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# Load & clean dataset
df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Split features & target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize inputs & target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Function to create a DNN model with Optuna-tuned hyperparameters
def create_dnn_model(trial, input_dim):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_val = trial.suggest_float("l2_val", 1e-5, 1e-2, log=True)

    # Adaptive LR range to reduce NaN risk
    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    n_layers = trial.suggest_int("n_layers", 3, 7)
    units = trial.suggest_categorical("units", [64, 128, 256])

    model = Sequential()
    model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_val), input_dim=input_dim))
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers - 1):
        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Regression output layer

    # Gradient clipping to prevent exploding gradients
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mse")
    return model

# Outer loop: multiple runs
for run_num in range(1, 6):
    print(f"\n Run {run_num}/5 ")
    run_dir = os.path.join(output_dir, f"run{run_num}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_list = []
    best_params_list = []

    # Inner loop: 5 iterations per run
    for iteration in range(1, 6):
        print(f"\n--- Iteration {iteration}/5 ---")

        def objective_split(trial):
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = create_dnn_model(trial, X.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=1)
            y_pred = model.predict(X_test).flatten()
            return mean_squared_error(y_test, y_pred)

        # Optuna hyperparameter search
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_split, n_trials=100, show_progress_bar=True)
        best_params = study.best_trial.params

        # Retrain model with best hyperparameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        best_model = create_dnn_model(optuna.trial.FixedTrial(best_params), X.shape[1])
        best_model.fit(X_train, y_train, epochs=100, batch_size=best_params["batch_size"], verbose=1)

        # Predictions
        y_test_pred = best_model.predict(X_test).flatten()
        y_train_pred = best_model.predict(X_train).flatten()

        # Metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        corr, _ = pearsonr(y_test.flatten(), y_test_pred)
        train_loss = best_model.evaluate(X_train, y_train, verbose=0)
        test_loss = best_model.evaluate(X_test, y_test, verbose=0)

        metrics_list.append({
            "Iteration": iteration,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
            "Train_Loss": train_loss,
            "Test_Loss": test_loss
        })
        best_params_list.append(best_params)

    # Save results for this run
    pd.DataFrame(metrics_list).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.DataFrame(best_params_list).to_csv(os.path.join(run_dir, "best_hyperparameters.csv"), index=False)

print("\nAll runs completed. Results saved in subfolders.")

