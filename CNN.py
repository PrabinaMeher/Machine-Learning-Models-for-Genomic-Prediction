import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

feature_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Scale features and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
y = scaler_Y.fit_transform(y.reshape(-1, 1)).flatten()

# Reshape for CNN input
X = X.reshape((X.shape[0], X.shape[1], 1))


def create_model(trial, input_shape):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.2, 0.4, 0.5])
    l2_val = trial.suggest_categorical("l2_val", [0.0001, 0.001, 0.01])

    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    layer_filters = trial.suggest_categorical("layer_filters", [[32], [64, 32], [128, 64, 32]])
    kernel_size = trial.suggest_categorical("kernel_size", [1, 3, 5])

    model = Sequential()
    model.add(Conv1D(filters=layer_filters[0], kernel_size=kernel_size,
                     activation=activation, kernel_regularizer=l2(l2_val),
                     input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for filters in layer_filters[1:]:
        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation=activation, kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mse")
    return model

def create_model_from_params(params, input_shape):
    activation = params["activation"]
    dropout_rate = params["dropout_rate"]
    l2_val = params["l2_val"]
    learning_rate = params["learning_rate"]
    layer_filters = params["layer_filters"]
    kernel_size = params["kernel_size"]
    optimizer_name = params["optimizer"]

    model = Sequential()
    model.add(Conv1D(filters=layer_filters[0], kernel_size=kernel_size,
                     activation=activation, kernel_regularizer=l2(l2_val),
                     input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for filters in layer_filters[1:]:
        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation=activation, kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse")
    return model


# OUTER LOOP: RUNS

nsplits = 5  

for split_num in range(1, nsplits + 1):
    run_dir = os.path.join(output_dir, f"run{split_num}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_list = []
    best_params_list = []

    print(f"\n Starting Run {split_num}/{nsplits}")

    # INNER LOOP: ITERATIONS PER RUN
    for iteration in range(1, 6):
        print(f"\n Iteration {iteration}/5 ")

        # New random split each iteration
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        def objective(trial):
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            model = create_model(trial, input_shape=(X.shape[1], X.shape[2]))
            model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=1)
            y_pred = model.predict(X_test).flatten()
            mse = mean_squared_error(y_test, y_pred)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        best_params = study.best_trial.params

        # Retrain best model
        best_model = create_model_from_params(best_params, input_shape=(X.shape[1], X.shape[2]))
        best_model.fit(X_train, y_train, epochs=100, batch_size=best_params["batch_size"], verbose=1)

        y_test_pred = best_model.predict(X_test).flatten()
        y_train_pred = best_model.predict(X_train).flatten()

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

        best_params_list.append({
            "Iteration": iteration,
            **best_params
        })

    # Save per-run results
    pd.DataFrame(metrics_list).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.DataFrame(best_params_list).to_csv(os.path.join(run_dir, "best_hyperparameters.csv"), index=False)

print("\nAll runs complete. Results saved in separate run folders.")

