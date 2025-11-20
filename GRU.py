import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

# Disable GPU (optional)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.set_visible_devices([], 'GPU')

# LOAD DATA
feature_file = sys.argv[1]
output_dir = "GRU"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(feature_file)

# Drop NaN and inf/-inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Split features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)  # Ensure 2D for scaler

# Standardize features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Standardize target
scaler_Y = StandardScaler()
y = scaler_Y.fit_transform(y)

# Reshape for RNN input: (samples, timesteps=1, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

def create_model(trial, input_shape):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.2, 0.4, 0.5])
    l2_val = trial.suggest_categorical("l2_val", [0.0001, 0.001, 0.01])
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
    layer_sizes = trial.suggest_categorical("layer_sizes", [[32], [64, 32], [128, 64, 32]])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    model = Sequential()
    units = layer_sizes[0]
    return_sequences = len(layer_sizes) > 1
    model.add(GRU(units, activation=activation, return_sequences=return_sequences,
                  kernel_regularizer=l2(l2_val), input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for units in layer_sizes[1:-1]:
        model.add(GRU(units, activation=activation, return_sequences=True,
                      kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    if len(layer_sizes) > 1:
        units = layer_sizes[-1]
        model.add(GRU(units, activation=activation, return_sequences=False,
                      kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse")
    return model

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = create_model(trial, input_shape=(X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()
    mse = mean_squared_error(y_true, y_pred)
    return mse

def create_model_from_params(params, input_shape):
    activation = params["activation"]
    dropout_rate = params["dropout_rate"]
    l2_val = params["l2_val"]
    learning_rate = params["learning_rate"]
    layer_sizes = params["layer_sizes"]
    optimizer_name = params["optimizer"]

    model = Sequential()
    units = layer_sizes[0]
    return_sequences = len(layer_sizes) > 1
    model.add(GRU(units, activation=activation, return_sequences=return_sequences,
                  kernel_regularizer=l2(l2_val), input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for units in layer_sizes[1:-1]:
        model.add(GRU(units, activation=activation, return_sequences=True,
                      kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    if len(layer_sizes) > 1:
        units = layer_sizes[-1]
        model.add(GRU(units, activation=activation, return_sequences=False,
                      kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse")
    return model

for split_num in range(1, 6):
    print(f"\nSplit {split_num}/5")
    run_dir = os.path.join(output_dir, f"run{split_num}")
    os.makedirs(run_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    metrics_runs = []
    params_runs = []

    for iteration in range(1, 6):
        print(f"  Iteration {iteration}/5")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_trial.params

        best_model = create_model_from_params(best_params, input_shape=(X.shape[1], X.shape[2]))
        best_model.fit(X_train, y_train, epochs=100, batch_size=best_params["batch_size"], verbose=1)

        y_test_pred = best_model.predict(X_test).flatten()
        y_train_pred = best_model.predict(X_train).flatten()

        y_test_true = y_test.flatten()
        y_train_true = y_train.flatten()

        mse = mean_squared_error(y_test_true, y_test_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test_true, y_test_pred)
        corr, _ = pearsonr(y_test_true, y_test_pred)

        train_loss = best_model.evaluate(X_train, y_train, verbose=1)
        test_loss = best_model.evaluate(X_test, y_test, verbose=1)

        run_metrics = {
            "Split": split_num,
            "Iteration": iteration,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
            "Train_Loss": train_loss,
            "Test_Loss": test_loss
        }

        metrics_runs.append(run_metrics)
        params_runs.append({"Split": split_num, "Iteration": iteration, **best_params})

    # Save aggregated metrics and best params per run (split)
    pd.DataFrame(metrics_runs).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.DataFrame(params_runs).to_csv(os.path.join(run_dir, "best_hyperparameters.csv"), index=False)

print("\nAll splits done. Results saved in separate run folders.")

