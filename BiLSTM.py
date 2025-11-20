import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# LOAD DATA
feature_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(feature_file)

# Drop NaN and inf/-inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Split features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Standardize features & target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
y = scaler_Y.fit_transform(y)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# MODEL CREATION FOR OPTUNA
def create_model(trial, input_shape):
    activation = trial.suggest_categorical("activation", ["tanh", "sigmoid"])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.2, 0.4, 0.5])
    l2_val = trial.suggest_categorical("l2_val", [0.0001, 0.001, 0.01])

    # Adaptive LR range
    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])

    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_units, activation=activation, kernel_regularizer=l2(l2_val)),
                            input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mse")
    return model

def objective(trial, X_train, X_test, y_train, y_test):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    model = create_model(trial, (X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=1)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    return mse

def create_model_from_params(params, input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(params["lstm_units"], activation=params["activation"],
                                 kernel_regularizer=l2(params["l2_val"])), input_shape=input_shape))
    model.add(Dropout(params["dropout_rate"]))
    model.add(Dense(1))

    if params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

    model.compile(optimizer=optimizer, loss="mse")
    return model

# Outer loop: Runs
for run_num in range(1, 6):
    run_dir = os.path.join(output_dir, f"run{run_num}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_list = []
    best_params_list = []

    # Inner loop: Iterations with fresh random splits
    for iteration in range(1, 6):
        print(f"\n Run {run_num}/5 | Iteration {iteration}/5 ")

        # New random split every iteration
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Optuna tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test),
                       n_trials=10, show_progress_bar=True)
        best_params = study.best_trial.params

        # Retrain with best params
        best_model = create_model_from_params(best_params, (X.shape[1], X.shape[2]))
        best_model.fit(X_train, y_train, epochs=50, batch_size=best_params["batch_size"], verbose=1)

        # Predictions and metrics
        y_test_pred = best_model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        corr, _ = pearsonr(y_test.flatten(), y_test_pred)
        train_loss = best_model.evaluate(X_train, y_train, verbose=1)
        test_loss = best_model.evaluate(X_test, y_test, verbose=1)

        metrics_list.append({
            "Run": run_num,
            "Iteration": iteration,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
            "Train_Loss": train_loss,
            "Test_Loss": test_loss
        })

        best_params_list.append({
            "Run": run_num,
            "Iteration": iteration,
            **best_params
        })

    # Save per-run results
    pd.DataFrame(metrics_list).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.DataFrame(best_params_list).to_csv(os.path.join(run_dir, "best_hyperparameters.csv"), index=False)

print("\nAll runs and iterations done. Results saved.")

