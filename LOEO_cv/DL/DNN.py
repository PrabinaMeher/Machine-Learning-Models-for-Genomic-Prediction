#!/usr/bin/env python3

import os
import sys
import random
import gc
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
import optuna

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_INNER_SPLITS = 5
N_TRIALS = 100
EPOCHS = 50
BASE_SEED = 1000



dataset_files = [os.path.abspath(x) for x in sys.argv[1:]]

split_file = os.path.join(
    os.path.dirname(dataset_files[0]),
    "LOEO_splits",
    "LOEO_data_split.csv"
)

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"LOEO split file not found:\n{split_file}\n"
        "Run generate_LOEO_splits.py first."
    )

env_names = [
    os.path.splitext(os.path.basename(f))[0]
    for f in dataset_files
]

raw_datasets = {}

for name, file in zip(env_names, dataset_files):
    df = pd.read_csv(file, header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.shape[1] < 2:
        raise ValueError(
            f"{file}: expected features + phenotype."
        )

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    raw_datasets[name] = (X, y)


n_samples = len(raw_datasets[env_names[0]][0])
n_features = raw_datasets[env_names[0]][0].shape[1]

for name in env_names:
    if len(raw_datasets[name][0]) != n_samples:
        raise ValueError(
            "All environments must contain the same number of genotypes."
        )

    if raw_datasets[name][0].shape[1] != n_features:
        raise ValueError(
            "All environments must contain the same number of features."
        )


loeo_splits = pd.read_csv(split_file)

required_columns = {
    "LOEO_Iteration",
    "Environment",
    "ID",
    "Set"
}

if not required_columns.issubset(loeo_splits.columns):
    raise ValueError(
        f"LOEO split file must contain: {sorted(required_columns)}"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_DNN_model(trial, input_dim):
    n_layers = trial.suggest_int(
        "n_layers", 3, 7
    )

    dropout_rate = trial.suggest_float(
        "dropout_rate", 0.2, 0.5
    )

    l2_val = trial.suggest_float(
        "l2_val", 1e-4, 1e-2, log=True
    )

    activation = trial.suggest_categorical(
        "activation",
        ["relu", "tanh", "sigmoid"]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["adam", "sgd"]
    )

    units = []

    for i in range(n_layers):
        units.append(
            trial.suggest_categorical(
                f"units_layer_{i+1}",
                [16, 32, 64, 128]
            )
        )

    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            model.add(
                Dense(
                    units[i],
                    activation=activation,
                    kernel_regularizer=l2(l2_val),
                    input_dim=input_dim
                )
            )
        else:
            model.add(
                Dense(
                    units[i],
                    activation=activation,
                    kernel_regularizer=l2(l2_val)
                )
            )

        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model


def make_objective(X_train_raw, y_train_raw, iteration):
    kfold = KFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=BASE_SEED + iteration
    )

    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size",
            [64, 128, 256]
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(X_train_raw),
            start=1
        ):
            set_seed(
                BASE_SEED
                + iteration * 1000
                + trial.number * 10
                + fold
            )

            X_inner_train = X_train_raw[train_idx]
            X_inner_val = X_train_raw[val_idx]

            y_inner_train = y_train_raw[train_idx]
            y_inner_val = y_train_raw[val_idx]

            scaler_X = StandardScaler().fit(
                X_inner_train
            )

            scaler_y = StandardScaler().fit(
                y_inner_train
            )

            X_inner_train_s = scaler_X.transform(
                X_inner_train
            )

            X_inner_val_s = scaler_X.transform(
                X_inner_val
            )

            y_inner_train_s = scaler_y.transform(
                y_inner_train
            )

            y_inner_val_s = scaler_y.transform(
                y_inner_val
            )

            model = create_DNN_model(
                trial,
                X_inner_train_s.shape[1]
            )

            model.fit(
                X_inner_train_s,
                y_inner_train_s.flatten(),
                epochs=EPOCHS,
                batch_size=batch_size,
                verbose=0
            )

            pred = model.predict(
                X_inner_val_s,
                verbose=0
            ).flatten()

            if np.isnan(pred).any() or np.isinf(pred).any():
                return np.inf

            score = mean_squared_error(
                y_inner_val_s.flatten(),
                pred
            )

            fold_scores.append(score)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        return float(np.mean(fold_scores))

    return objective


out_root = os.path.join(
    os.path.dirname(dataset_files[0]),
    "DNN_LOEO_Optuna"
)

os.makedirs(out_root, exist_ok=True)

all_metrics_rows = []
all_best_params = []
all_trials = []
all_predictions = []


for iteration, leave_name in enumerate(
    env_names,
    start=1
):

    other_names = [
        n for n in env_names
        if n != leave_name
    ]

    print("\n" + "=" * 70)
    print(
        f"LOEO {iteration}/{len(env_names)}"
    )
    print(
        f"Train: {'+'.join(other_names)}"
    )
    print(
        f"Test : {leave_name}"
    )
    print("=" * 70)

    env_dir = os.path.join(
        out_root,
        leave_name
    )

    preds_dir = os.path.join(
        env_dir,
        "predictions"
    )

    os.makedirs(preds_dir, exist_ok=True)

    X_train_raw = np.vstack([
        raw_datasets[n][0]
        for n in other_names
    ])

    y_train_raw = np.vstack([
        raw_datasets[n][1]
        for n in other_names
    ])

    X_test_raw, y_test_raw = raw_datasets[
        leave_name
    ]

    objective = make_objective(
        X_train_raw,
        y_train_raw,
        iteration
    )

    study_seed = BASE_SEED + iteration

    set_seed(study_seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=study_seed
        )
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    best_params = study.best_trial.params.copy()

    best_record = best_params.copy()
    best_record.update({
        "LOEO_Iteration": leave_name,
        "Train_Envs": "+".join(other_names),
        "Test_Env": leave_name,
        "Best_Trial": study.best_trial.number,
        "Objective_Value": study.best_trial.value
    })

    all_best_params.append(best_record)

    pd.DataFrame([best_record]).to_csv(
        os.path.join(
            env_dir,
            "best_hyperparameters.csv"
        ),
        index=False
    )

    for trial in study.trials:
        trial_record = trial.params.copy()
        trial_record.update({
            "LOEO_Iteration": leave_name,
            "Train_Envs": "+".join(other_names),
            "Test_Env": leave_name,
            "Trial_Number": trial.number,
            "Trial_State": str(trial.state),
            "Objective_Value": trial.value
        })
        all_trials.append(trial_record)

    pd.DataFrame(all_trials).to_csv(
        os.path.join(
            out_root,
            "all_optuna_trials.csv"
        ),
        index=False
    )

    scaler_X = StandardScaler().fit(
        X_train_raw
    )

    scaler_y = StandardScaler().fit(
        y_train_raw
    )

    X_train = scaler_X.transform(
        X_train_raw
    )

    X_test = scaler_X.transform(
        X_test_raw
    )

    y_train = scaler_y.transform(
        y_train_raw
    ).flatten()

    set_seed(
        BASE_SEED
        + iteration * 10000
        + 9999
    )

    class FixedTrial:
        def __init__(self, params):
            self.params = params

        def suggest_int(self, name, low, high, step=1):
            return self.params[name]

        def suggest_float(self, name, low, high, step=None, log=False):
            return self.params[name]

        def suggest_categorical(self, name, choices):
            return self.params[name]

    final_trial = FixedTrial(best_params)

    model = create_DNN_model(
        final_trial,
        X_train.shape[1]
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=best_params["batch_size"],
        verbose=1
    )

    y_pred_scaled = model.predict(
        X_test,
        verbose=1
    ).flatten()

    y_pred_orig = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    y_test_orig = y_test_raw.flatten()

    mse = mean_squared_error(
        y_test_orig,
        y_pred_orig
    )

    rmse = np.sqrt(mse)

    mape = mean_absolute_percentage_error(
        y_test_orig,
        y_pred_orig
    )

    if (
        len(y_test_orig) > 1
        and np.std(y_test_orig) > 0
        and np.std(y_pred_orig) > 0
    ):
        corr, _ = pearsonr(
            y_test_orig,
            y_pred_orig
        )
    else:
        corr = np.nan

    metrics = pd.DataFrame([{
        "LOEO_Iteration": leave_name,
        "Train_Envs": "+".join(other_names),
        "Test_Env": leave_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "Correlation": corr
    }])

    metrics.to_csv(
        os.path.join(
            env_dir,
            "Metrics.csv"
        ),
        index=False
    )

    all_metrics_rows.append(metrics)

    predictions = pd.DataFrame({
        "LOEO_Iteration": leave_name,
        "Train_Envs": "+".join(other_names),
        "Test_Env": leave_name,
        "Row_Index": np.arange(
            len(y_test_orig)
        ),
        "ID": [
            f"ID{i + 1}"
            for i in range(len(y_test_orig))
        ],
        "Actual": y_test_orig,
        "Predicted": y_pred_orig
    })

    predictions.to_csv(
        os.path.join(
            env_dir,
            "ActualVsPredicted.csv"
        ),
        index=False
    )

    all_predictions.append(predictions)

    with open(
        os.path.join(env_dir, "log.txt"),
        "w"
    ) as f:
        f.write(
            f"Train Environments: {'+'.join(other_names)}\n"
        )
        f.write(
            f"Test Environment: {leave_name}\n"
        )
        f.write(
            f"Optuna Trials: {N_TRIALS}\n"
        )
        f.write(
            f"Best Hyperparameters:\n{best_params}\n"
        )
        f.write(
            f"Test MSE: {mse}\n"
        )
        f.write(
            f"Test RMSE: {rmse}\n"
        )
        f.write(
            f"Test MAPE: {mape}\n"
        )
        f.write(
            f"Test Correlation: {corr}\n"
        )

    del model
    tf.keras.backend.clear_session()
    gc.collect()


combined_metrics = pd.concat(
    all_metrics_rows,
    ignore_index=True
)

combined_metrics.to_csv(
    os.path.join(
        out_root,
        "ALL_LOEO_summary.csv"
    ),
    index=False
)

pd.DataFrame(all_best_params).to_csv(
    os.path.join(
        out_root,
        "ALL_LOEO_best_hyperparameters.csv"
    ),
    index=False
)

if all_predictions:
    pd.concat(
        all_predictions,
        ignore_index=True
    ).to_csv(
        os.path.join(
            out_root,
            "ALL_LOEO_predictions.csv"
        ),
        index=False
    )

print("\n" + "=" * 70)
print("LOEO DNN OPTUNA COMPLETE")
print("=" * 70)
print(combined_metrics)
