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

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Load input/output paths
feature_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Split features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize features and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# ANN model creation function
def create_ann_model(trial, input_dim):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_val = trial.suggest_float("l2_val", 1e-5, 1e-2, log=True)

    # Adjust LR range based on activation
    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_categorical("units", [32, 64, 128])

    model = Sequential()
    model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_val), input_dim=input_dim))
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers - 1):
        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Output layer for regression

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mse")
    return model

# Training + tuning loop
for split_num in range(1, 6):  # 5 runs
    run_dir = os.path.join(output_dir, f"run{split_num}")
    os.makedirs(run_dir, exist_ok=True)

    split_metrics = []
    split_best_params = []

    for iter_num in range(1, 6):  # 5 iterations per run
        print(f"\n Run {split_num}/5, Iteration {iter_num}/5 ")

        # Objective function for Optuna
        def objective_split(trial):
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = create_ann_model(trial, X.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=1)
            y_pred = model.predict(X_test).flatten()
            return mean_squared_error(y_test, y_pred)

        # Hyperparameter tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_split, n_trials=100, show_progress_bar=True)
        best_params = study.best_trial.params

        # Retrain with best params
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        best_model = create_ann_model(optuna.trial.FixedTrial(best_params), X.shape[1])
        best_model.fit(X_train, y_train, epochs=100, batch_size=best_params["batch_size"], verbose=1)

        # Predictions
        y_test_pred = best_model.predict(X_test).flatten()
        y_train_pred = best_model.predict(X_train).flatten()

        # Metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        corr, _ = pearsonr(y_test.flatten(), y_test_pred)
        train_loss = best_model.evaluate(X_train, y_train, verbose=1)
        test_loss = best_model.evaluate(X_test, y_test, verbose=1)

        # Store metrics & params
        split_metrics.append({
            "Run": split_num,
            "Iteration": iter_num,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
            "Train_Loss": train_loss,
            "Test_Loss": test_loss
        })

        best_params_with_meta = best_params.copy()
        best_params_with_meta.update({"Run": split_num, "Iteration": iter_num})
        split_best_params.append(best_params_with_meta)

    # Save results for this run
    pd.DataFrame(split_metrics).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.DataFrame(split_best_params).to_csv(os.path.join(run_dir, "best_hyperparameters.csv"), index=False)

print("\nAll runs and iterations complete. Results saved in separate folders.")

