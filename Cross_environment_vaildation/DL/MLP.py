import sys
import os
import random
import gc
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected and enabled: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPU detected - running on CPU.")

N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 100
BASE_SEED = 1000
EPOCHS = 100

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: python MLP_Cross_Environment_Optuna.py env1.csv env2.csv ..."
    )

feature_files = [os.path.abspath(x) for x in sys.argv[1:]]
dataset_dir = os.path.dirname(feature_files[0])

env_names = [
    os.path.splitext(os.path.basename(f))[0].replace("_dataset_with_ID", "")
    for f in feature_files
]

output_dir = os.path.join(dataset_dir, "MLP_Cross_Environment_Optuna_out")
preds_dir = os.path.join(output_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)

split_file = os.path.join(dataset_dir, "cross_environment_data_split.csv")
if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Common split file not found: {split_file}\n"
        "Run the common split generator first."
    )

outer_splits = pd.read_csv(split_file)
required_outer = {"Repetition", "Fold", "ID", "Set"}
if not required_outer.issubset(outer_splits.columns):
    raise ValueError(f"Split file must contain: {sorted(required_outer)}")

data = {}
for env, file in zip(env_names, feature_files):
    df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[1] < 3:
        raise ValueError(f"{file}: expected ID + features + target.")

    data[env] = {
        "df": df,
        "ids": df.iloc[:, 0].values,
        "X": df.iloc[:, 1:-1].values,
        "y": df.iloc[:, -1].values.reshape(-1, 1),
    }

reference_env = env_names[0]
reference_ids = data[reference_env]["ids"]

for env in env_names:
    if not np.array_equal(data[env]["ids"], reference_ids):
        raise ValueError(f"IDs/order do not match between {reference_env} and {env}.")
    if data[env]["X"].shape[1] != data[reference_env]["X"].shape[1]:
        raise ValueError(f"Feature count does not match between {reference_env} and {env}.")

if set(reference_ids) != set(outer_splits["ID"]):
    raise ValueError("Dataset IDs and common split IDs do not match exactly.")

N_FEATURES = data[reference_env]["X"].shape[1]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def build_mlp(layer_units, layer_activations, dropout_rate, l2_val,
              learning_rate, optimizer_name, input_shape):
    model = Sequential()
    model.add(Dense(
        layer_units[0],
        activation=layer_activations[0],
        kernel_regularizer=l2(l2_val),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))

    for units, activation in zip(layer_units[1:], layer_activations[1:]):
        model.add(Dense(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_val)
        ))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, clipnorm=1.0
        )

    model.compile(optimizer=optimizer, loss="mse")
    return model

def suggest_mlp_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    l2_val = trial.suggest_float("l2_val", 1e-4, 1e-2, log=True)

    first_activation = trial.suggest_categorical(
        "activation_layer_1", ["relu", "tanh", "sigmoid"]
    )

    if first_activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3

    learning_rate = trial.suggest_float(
        "learning_rate", lr_min, lr_max, log=True
    )

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    layer_units = [
        trial.suggest_int("units_layer_1", 32, 256, step=32)
    ]
    layer_activations = [first_activation]

    for i in range(2, n_layers + 1):
        layer_units.append(
            trial.suggest_int(f"units_layer_{i}", 32, 256, step=32)
        )
        layer_activations.append(
            trial.suggest_categorical(
                f"activation_layer_{i}",
                ["relu", "tanh", "sigmoid"]
            )
        )

    return {
        "n_layers": n_layers,
        "dropout_rate": dropout_rate,
        "l2_val": l2_val,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        **{
            f"units_layer_{i}": layer_units[i - 1]
            for i in range(1, n_layers + 1)
        },
        **{
            f"activation_layer_{i}": layer_activations[i - 1]
            for i in range(1, n_layers + 1)
        },
    }

def build_from_params(params, input_shape):
    units = [params[f"units_layer_{i}"] for i in range(1, params["n_layers"] + 1)]
    activations = [
        params[f"activation_layer_{i}"]
        for i in range(1, params["n_layers"] + 1)
    ]
    return build_mlp(
        units, activations,
        params["dropout_rate"],
        params["l2_val"],
        params["learning_rate"],
        params["optimizer"],
        input_shape
    )

def make_inner_objective(X_outer_train, y_outer_train,
                         repetition, outer_fold, outer_train_ids):
    subset = outer_splits[
        (outer_splits["Repetition"] == repetition) &
        (outer_splits["Fold"] == outer_fold)
    ].copy()

    rng = np.random.RandomState(
        (BASE_SEED + repetition) * 100 + outer_fold
    )
    shuffled_ids = np.array(outer_train_ids, dtype=object)
    rng.shuffle(shuffled_ids)

    fold_indices = np.array_split(
        np.arange(len(shuffled_ids)), N_INNER_SPLITS
    )

    id_to_pos = {sample_id: i for i, sample_id in enumerate(outer_train_ids)}
    fixed_inner_folds = []

    for val_idx in fold_indices:
        val_ids = set(shuffled_ids[val_idx])
        train_ids = [x for x in shuffled_ids if x not in val_ids]
        validation_ids = list(shuffled_ids[val_idx])

        fixed_inner_folds.append((
            np.array([id_to_pos[x] for x in train_ids], dtype=int),
            np.array([id_to_pos[x] for x in validation_ids], dtype=int)
        ))

    def objective(trial):
        params = suggest_mlp_params(trial)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        fold_scores = []

        for inner_fold_number, (train_idx, val_idx) in enumerate(
            fixed_inner_folds, start=1
        ):
            set_seed(
                BASE_SEED
                + repetition * 10000
                + outer_fold * 100
                + trial.number * 10
                + inner_fold_number
            )

            X_train = X_outer_train[train_idx]
            X_val = X_outer_train[val_idx]
            y_train = y_outer_train[train_idx]
            y_val = y_outer_train[val_idx]

            sx = StandardScaler().fit(X_train)
            sy = StandardScaler().fit(y_train)

            X_train_s = sx.transform(X_train)
            X_val_s = sx.transform(X_val)
            y_train_s = sy.transform(y_train)
            y_val_s = sy.transform(y_val)

            model = build_from_params(params, (N_FEATURES,))
            model.fit(
                X_train_s, y_train_s,
                epochs=EPOCHS,
                batch_size=batch_size,
                verbose=0
            )

            pred = model.predict(X_val_s, verbose=0).flatten()
            if np.isnan(pred).any() or np.isinf(pred).any():
                return np.inf

            fold_scores.append(mean_squared_error(y_val_s, pred))

            del model, sx, sy
            tf.keras.backend.clear_session()
            gc.collect()

        return float(np.mean(fold_scores))

    return objective

all_metrics = []
all_best_params = []
all_trial_records = []
all_predictions = []

id_to_global_pos = {sample_id: i for i, sample_id in enumerate(reference_ids)}

for repetition in range(1, N_REPETITIONS + 1):
    for outer_fold in range(1, N_OUTER_SPLITS + 1):
        print(
            f"\n===== Repetition {repetition}/{N_REPETITIONS} | "
            f"Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
        )

        outer_subset = outer_splits[
            (outer_splits["Repetition"] == repetition) &
            (outer_splits["Fold"] == outer_fold)
        ]

        outer_train_ids = outer_subset[
            outer_subset["Set"] == "train"
        ]["ID"].tolist()

        outer_test_ids = outer_subset[
            outer_subset["Set"] == "test"
        ]["ID"].tolist()

        outer_train_idx = np.array(
            [id_to_global_pos[x] for x in outer_train_ids], dtype=int
        )
        outer_test_idx = np.array(
            [id_to_global_pos[x] for x in outer_test_ids], dtype=int
        )

        for train_env in env_names:
            print(f"Training environment: {train_env}")

            X_train = data[train_env]["X"][outer_train_idx]
            y_train = data[train_env]["y"][outer_train_idx]

            objective = make_inner_objective(
                X_train, y_train,
                repetition, outer_fold, outer_train_ids
            )

            study_seed = BASE_SEED + repetition * 100 + outer_fold
            set_seed(study_seed)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=study_seed)
            )
            study.optimize(
                objective,
                n_trials=N_TRIALS,
                show_progress_bar=False
            )

            best_params = study.best_trial.params.copy()

            for trial in study.trials:
                record = trial.params.copy()
                record.update({
                    "Training_Environment": train_env,
                    "Repetition": repetition,
                    "Outer_Fold": outer_fold,
                    "Trial_Number": trial.number,
                    "Trial_State": str(trial.state),
                    "Objective_Value": trial.value
                })
                all_trial_records.append(record)

            record = best_params.copy()
            record.update({
                "Training_Environment": train_env,
                "Repetition": repetition,
                "Outer_Fold": outer_fold,
                "Best_Trial": study.best_trial.number,
                "Objective_Value": study.best_trial.value
            })
            all_best_params.append(record)

            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train)

            X_train_s = scaler_X.transform(X_train)
            y_train_s = scaler_y.transform(y_train)

            set_seed(
                BASE_SEED
                + repetition * 10000
                + outer_fold * 100
                + 9999
            )

            final_model = build_from_params(best_params, (N_FEATURES,))
            final_model.fit(
                X_train_s,
                y_train_s,
                epochs=EPOCHS,
                batch_size=best_params["batch_size"],
                verbose=0
            )

            for test_env in env_names:
                X_test = data[test_env]["X"][outer_test_idx]
                y_test = data[test_env]["y"][outer_test_idx]

                X_test_s = scaler_X.transform(X_test)
                y_test_s = scaler_y.transform(y_test)

                pred_s = final_model.predict(
                    X_test_s, verbose=0
                ).flatten()

                mse = mean_squared_error(y_test_s, pred_s)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test_s, pred_s)

                if len(y_test_s) > 1 and np.std(pred_s) > 0:
                    corr, _ = pearsonr(y_test_s.flatten(), pred_s)
                else:
                    corr = np.nan

                all_metrics.append({
                    "Training_Environment": train_env,
                    "Testing_Environment": test_env,
                    "Repetition": repetition,
                    "Outer_Fold": outer_fold,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Correlation": corr
                })

                pred_orig = scaler_y.inverse_transform(
                    pred_s.reshape(-1, 1)
                ).flatten()

                pred_df = pd.DataFrame({
                    "Training_Environment": train_env,
                    "Testing_Environment": test_env,
                    "Repetition": repetition,
                    "Outer_Fold": outer_fold,
                    "ID": outer_test_ids,
                    "Actual": y_test.flatten(),
                    "Predicted": pred_orig
                })

                all_predictions.append(pred_df)

            del final_model, scaler_X, scaler_y
            tf.keras.backend.clear_session()
            gc.collect()

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(
    os.path.join(output_dir, "all_cross_environment_metrics.csv"),
    index=False
)

pd.DataFrame(all_best_params).to_csv(
    os.path.join(output_dir, "all_best_hyperparameters.csv"),
    index=False
)

pd.DataFrame(all_trial_records).to_csv(
    os.path.join(output_dir, "all_optuna_trials.csv"),
    index=False
)

if all_predictions:
    preds_df = pd.concat(all_predictions, ignore_index=True)
    preds_df.to_csv(
        os.path.join(output_dir, "all_cross_environment_predictions.csv"),
        index=False
    )

    for (train_env, test_env), group in preds_df.groupby(
        ["Training_Environment", "Testing_Environment"]
    ):
        group.to_csv(
            os.path.join(
                preds_dir,
                f"{train_env}_to_{test_env}_predictions.csv"
            ),
            index=False
        )

if not metrics_df.empty:
    pair_summary = metrics_df.groupby(
        ["Training_Environment", "Testing_Environment"]
    )[["MSE", "RMSE", "MAPE", "Correlation"]].agg(["mean", "std"])

    pair_summary.to_csv(
        os.path.join(output_dir, "cross_environment_pair_summary.csv")
    )

print(
    f"\nCompleted {len(metrics_df)} cross-environment evaluations."
)
