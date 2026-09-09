import sys
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected - running on CPU.")

# Configuration
N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 100

BASE_SEED = 1000
MODEL_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

feature_file = sys.argv[1]
dataset_dir = os.path.dirname(os.path.abspath(feature_file))
feature_basename = os.path.splitext(os.path.basename(feature_file))[0]

output_dir = os.path.join(
    dataset_dir,
    f"{feature_basename}_{MODEL_NAME}_out"
)
os.makedirs(output_dir, exist_ok=True)

preds_dir = os.path.join(output_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)

# Fixed split files are expected next to the dataset.
split_file = os.path.join(dataset_dir, f"{feature_basename}_data_split.csv")
inner_split_file = os.path.join(dataset_dir, f"{feature_basename}_inner_data_split.csv")

print(f"Model: {MODEL_NAME} | Dataset: {feature_file}")
print(f"Outer split file: {split_file}")
print(f"Inner split file: {inner_split_file}")

# Load data
df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# The fixed split workflow expects the first column to be the permanent ID
# and the last column to be the target.
if df.shape[1] < 3:
    raise ValueError("Expected at least ID + one feature + target column.")

ids = df.iloc[:, 0].values
X_all = df.iloc[:, 1:-1].values
y_all = df.iloc[:, -1].values.reshape(-1, 1)
N_FEATURES = X_all.shape[1]

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Fixed outer split file not found: {split_file}\n"
        "Use the common fixed split generator first."
    )

outer_splits = pd.read_csv(split_file)
required_outer_cols = {"Repetition", "Fold", "ID", "Set"}
if not required_outer_cols.issubset(outer_splits.columns):
    raise ValueError(
        f"Outer split file must contain columns: {sorted(required_outer_cols)}"
    )

# Validate IDs
data_ids = set(ids)
split_ids = set(outer_splits["ID"])
if data_ids != split_ids:
    raise ValueError(
        "Dataset IDs and IDs in data_split.csv do not match exactly."
    )

# Create/load one shared inner split file.
# This file is intended to be reused by DNN, ANN, GRU, BiLSTM, CNN, etc.
def get_or_create_inner_splits():
    if os.path.exists(inner_split_file):
        inner_df = pd.read_csv(inner_split_file)
        required = {"Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"}
        if not required.issubset(inner_df.columns):
            raise ValueError(
                f"Inner split file must contain columns: {sorted(required)}"
            )
        return inner_df

    rows = []

    for repetition in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            outer_train_ids = outer_splits[
                (outer_splits["Repetition"] == repetition) &
                (outer_splits["Fold"] == outer_fold) &
                (outer_splits["Set"] == "train")
            ]["ID"].tolist()

            if len(outer_train_ids) == 0:
                raise ValueError(
                    f"No outer-training IDs found for repetition={repetition}, "
                    f"outer_fold={outer_fold}."
                )

            # Same deterministic inner split generation used for every model.
            rng = np.random.RandomState(
                (BASE_SEED + repetition) * 100 + outer_fold
            )
            shuffled_ids = np.array(outer_train_ids, dtype=object)
            rng.shuffle(shuffled_ids)

            fold_indices = np.array_split(
                np.arange(len(shuffled_ids)), N_INNER_SPLITS
            )

            for inner_fold, val_idx in enumerate(fold_indices, start=1):
                val_ids = set(shuffled_ids[val_idx])
                for sample_id in shuffled_ids:
                    rows.append({
                        "Repetition": repetition,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": sample_id,
                        "Set": "validation" if sample_id in val_ids else "train"
                    })

    inner_df = pd.DataFrame(rows)
    inner_df.to_csv(inner_split_file, index=False)
    print(f"Created shared inner split file: {inner_split_file}")
    return inner_df

inner_splits = get_or_create_inner_splits()


def validate_inner_splits(repetition, outer_fold, outer_train_ids):
    subset = inner_splits[
        (inner_splits["Repetition"] == repetition) &
        (inner_splits["Outer_Fold"] == outer_fold)
    ].copy()

    expected_ids = set(outer_train_ids)
    if set(subset["ID"]) != expected_ids:
        raise ValueError(
            f"Inner split IDs do not exactly match outer-train IDs for "
            f"repetition={repetition}, outer_fold={outer_fold}."
        )

    if set(subset["Set"]) != {"train", "validation"}:
        raise ValueError(
            f"Inner split must contain both train and validation sets for "
            f"repetition={repetition}, outer_fold={outer_fold}."
        )

    for inner_fold in range(1, N_INNER_SPLITS + 1):
        fold = subset[subset["Inner_Fold"] == inner_fold]
        if len(fold) == 0:
            raise ValueError(
                f"Missing inner fold {inner_fold} for repetition={repetition}, "
                f"outer_fold={outer_fold}."
            )
        train_ids = set(fold.loc[fold["Set"] == "train", "ID"])
        val_ids = set(fold.loc[fold["Set"] == "validation", "ID"])
        if train_ids & val_ids:
            raise ValueError("Inner train/validation overlap detected.")
        if train_ids | val_ids != expected_ids:
            raise ValueError("Inner fold does not cover all outer-train IDs.")


def reshape_for_gru(X_2d):
    return X_2d.reshape((X_2d.shape[0], X_2d.shape[1], 1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_gru_model(trial, input_shape):
    activation = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid"]
    )
    dropout_rate = trial.suggest_categorical(
        "dropout_rate", [0.2, 0.4, 0.5]
    )
    l2_val = trial.suggest_categorical(
        "l2_val", [0.0001, 0.001, 0.01]
    )

    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3

    learning_rate = trial.suggest_float(
        "learning_rate", lr_min, lr_max, log=True
    )
    layer_sizes = trial.suggest_categorical(
        "layer_sizes", [(32,), (64, 32), (128, 64, 32)]
    )
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["adam", "sgd"]
    )

    return build_gru(
        activation, dropout_rate, l2_val, learning_rate,
        layer_sizes, optimizer_name, input_shape
    )


def build_gru(
    activation, dropout_rate, l2_val, learning_rate,
    layer_sizes, optimizer_name, input_shape
):
    model = Sequential()

    units = layer_sizes[0]
    return_sequences = len(layer_sizes) > 1

    model.add(
        GRU(
            units,
            activation=activation,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_val),
            input_shape=input_shape
        )
    )
    model.add(Dropout(dropout_rate))

    for units in layer_sizes[1:-1]:
        model.add(
            GRU(
                units,
                activation=activation,
                return_sequences=True,
                kernel_regularizer=l2(l2_val)
            )
        )
        model.add(Dropout(dropout_rate))

    if len(layer_sizes) > 1:
        units = layer_sizes[-1]
        model.add(
            GRU(
                units,
                activation=activation,
                return_sequences=False,
                kernel_regularizer=l2(l2_val)
            )
        )
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


def create_gru_model_from_params(params, input_shape):
    return build_gru(
        params["activation"],
        params["dropout_rate"],
        params["l2_val"],
        params["learning_rate"],
        params["layer_sizes"],
        params["optimizer"],
        input_shape
    )


def make_inner_objective(
    X_outer_train,
    y_outer_train,
    inner_repetition,
    inner_outer_fold,
    outer_train_ids,
    n_inner_splits=N_INNER_SPLITS
):
    validate_inner_splits(
        inner_repetition, inner_outer_fold, outer_train_ids
    )

    subset = inner_splits[
        (inner_splits["Repetition"] == inner_repetition) &
        (inner_splits["Outer_Fold"] == inner_outer_fold)
    ].copy()

    id_to_pos = {sample_id: i for i, sample_id in enumerate(outer_train_ids)}

    # Fixed inner folds: these indices are created ONCE and reused
    # by every Optuna trial for this outer fold.
    fixed_inner_folds = []
    for inner_fold in range(1, n_inner_splits + 1):
        fold = subset[subset["Inner_Fold"] == inner_fold]

        train_ids = fold.loc[fold["Set"] == "train", "ID"].tolist()
        val_ids = fold.loc[fold["Set"] == "validation", "ID"].tolist()

        inner_train_idx = np.array(
            [id_to_pos[x] for x in train_ids], dtype=int
        )
        inner_val_idx = np.array(
            [id_to_pos[x] for x in val_ids], dtype=int
        )

        fixed_inner_folds.append(
            (inner_train_idx, inner_val_idx)
        )

    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )
        fold_scores = []

        for inner_train_idx, inner_val_idx in fixed_inner_folds:
            # Deterministic model initialization per trial/fold.
            set_seed(
                BASE_SEED
                + inner_repetition * 10000
                + inner_outer_fold * 100
                + trial.number
            )

            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            # Leakage-free scaling: fit ONLY on inner-train.
            sx = StandardScaler().fit(X_inner_train)
            sy = StandardScaler().fit(y_inner_train)

            X_inner_train_s = reshape_for_gru(
                sx.transform(X_inner_train)
            )
            X_inner_val_s = reshape_for_gru(
                sx.transform(X_inner_val)
            )
            y_inner_train_s = sy.transform(y_inner_train)
            y_inner_val_s = sy.transform(y_inner_val)

            model = create_gru_model(
                trial, input_shape=(N_FEATURES, 1)
            )

            model.fit(
                X_inner_train_s,
                y_inner_train_s,
                epochs=100,
                batch_size=batch_size,
                verbose=0
            )

            y_pred = model.predict(
                X_inner_val_s, verbose=0
            ).flatten()

            fold_scores.append(
                mean_squared_error(y_inner_val_s, y_pred)
            )

            tf.keras.backend.clear_session()

        return float(np.mean(fold_scores))

    return objective


# Repeated nested CV: fixed 10 repetitions x 5 outer folds = 50 runs
all_metrics = []
all_best_params = []
all_trial_records = []
all_best_trial_records = []

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

        if set(outer_train_ids) & set(outer_test_ids):
            raise ValueError("Outer train/test overlap detected.")

        id_to_global_pos = {
            sample_id: i for i, sample_id in enumerate(ids)
        }

        outer_train_idx = np.array(
            [id_to_global_pos[x] for x in outer_train_ids],
            dtype=int
        )
        outer_test_idx = np.array(
            [id_to_global_pos[x] for x in outer_test_ids],
            dtype=int
        )

        X_outer_train = X_all[outer_train_idx]
        X_outer_test = X_all[outer_test_idx]
        y_outer_train = y_all[outer_train_idx]
        y_outer_test = y_all[outer_test_idx]

        # Hyperparameter search ONLY inside outer-train.
        objective = make_inner_objective(
            X_outer_train,
            y_outer_train,
            inner_repetition=repetition,
            inner_outer_fold=outer_fold,
            outer_train_ids=outer_train_ids
        )

        set_seed(BASE_SEED + repetition * 100 + outer_fold)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=BASE_SEED + repetition * 100 + outer_fold
            )
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=False
        )

        best_params = study.best_trial.params

        # Save Optuna trial history for this outer fold.
        for trial in study.trials:
            record = trial.params.copy()
            record.update({
                "Repetition": repetition,
                "Outer_Fold": outer_fold,
                "Trial_Number": trial.number,
                "Trial_State": str(trial.state),
                "Objective_Value": trial.value
            })
            all_trial_records.append(record)

        best_record = best_params.copy()
        best_record.update({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "Trial_Number": study.best_trial.number,
            "Objective_Value": study.best_trial.value
        })
        all_best_trial_records.append(best_record)

        # Final scaler is fitted ONLY on full outer-train.
        scaler_X = StandardScaler().fit(X_outer_train)
        scaler_y = StandardScaler().fit(y_outer_train)

        X_outer_train_s = reshape_for_gru(
            scaler_X.transform(X_outer_train)
        )
        X_outer_test_s = reshape_for_gru(
            scaler_X.transform(X_outer_test)
        )
        y_outer_train_s = scaler_y.transform(y_outer_train)
        y_outer_test_s = scaler_y.transform(y_outer_test)

        # Retrain final model on FULL outer-train.
        set_seed(
            BASE_SEED
            + repetition * 10000
            + outer_fold * 100
            + 9999
        )

        final_model = create_gru_model_from_params(
            best_params,
            input_shape=(N_FEATURES, 1)
        )

        final_model.fit(
            X_outer_train_s,
            y_outer_train_s,
            epochs=100,
            batch_size=best_params["batch_size"],
            verbose=0
        )

        # Evaluate ONCE on held-out outer-test.
        y_test_pred_s = final_model.predict(
            X_outer_test_s, verbose=0
        ).flatten()

        mse = mean_squared_error(y_outer_test_s, y_test_pred_s)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(
            y_outer_test_s, y_test_pred_s
        )
        corr, _ = pearsonr(
            y_outer_test_s.flatten(),
            y_test_pred_s
        )

        all_metrics.append({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr
        })

        best_params_with_meta = best_params.copy()
        best_params_with_meta["layer_sizes"] = str(
            best_params_with_meta["layer_sizes"]
        )
        best_params_with_meta.update({
            "Repetition": repetition,
            "Outer_Fold": outer_fold
        })
        all_best_params.append(best_params_with_meta)

        # Predictions returned to original target scale.
        y_test_pred_orig = scaler_y.inverse_transform(
            y_test_pred_s.reshape(-1, 1)
        ).flatten()
        y_test_actual_orig = y_outer_test.flatten()

        preds_df = pd.DataFrame({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "ID": outer_test_ids,
            "Row_Index": outer_test_idx,
            "Actual": y_test_actual_orig,
            "Predicted": y_test_pred_orig
        })

        preds_df.to_csv(
            os.path.join(
                preds_dir,
                f"predictions_rep{repetition}_fold{outer_fold}.csv"
            ),
            index=False
        )

        tf.keras.backend.clear_session()

# Aggregate results across all 50 runs.
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df[
    ["Repetition", "Outer_Fold", "MSE", "RMSE", "MAPE", "Correlation"]
]
metrics_df.to_csv(
    os.path.join(output_dir, "all_metrics_50_runs.csv"),
    index=False
)

best_params_df = pd.DataFrame(all_best_params)
cols = ["Repetition", "Outer_Fold"] + [
    c for c in best_params_df.columns
    if c not in ("Repetition", "Outer_Fold")
]
best_params_df = best_params_df[cols]
best_params_df.to_csv(
    os.path.join(
        output_dir,
        "all_best_hyperparameters_50_runs.csv"
    ),
    index=False
)

trials_df = pd.DataFrame(all_trial_records)
trials_df.to_csv(
    os.path.join(output_dir, "all_optuna_trials_50_runs.csv"),
    index=False
)

best_trials_df = pd.DataFrame(all_best_trial_records)
best_trials_df.to_csv(
    os.path.join(output_dir, "best_optuna_trials_50_runs.csv"),
    index=False
)

prediction_files = [
    os.path.join(preds_dir, f)
    for f in sorted(os.listdir(preds_dir))
    if f.endswith(".csv")
]

all_preds = pd.concat(
    [pd.read_csv(f) for f in prediction_files],
    ignore_index=True
)
all_preds.to_csv(
    os.path.join(output_dir, "all_predictions_50_runs.csv"),
    index=False
)

summary = metrics_df[
    ["MSE", "RMSE", "MAPE", "Correlation"]
].agg(["mean", "std"]).T
summary.columns = ["Mean", "Std"]
summary.to_csv(
    os.path.join(
        output_dir,
        "summary_mean_std_across_50_runs.csv"
    )
)

print(
    "\n===== Summary across all 50 runs "
    "(10 repetitions x 5 outer folds) ====="
)
print(summary)
print(f"\nAll {N_REPETITIONS * N_OUTER_SPLITS} runs complete.")
print("- Per-run metrics: all_metrics_50_runs.csv")
print("- Per-run best params: all_best_hyperparameters_50_runs.csv")
print("- Optuna trials: all_optuna_trials_50_runs.csv")
print("- Best Optuna trials: best_optuna_trials_50_runs.csv")
print("- Per-run predictions: predictions/")
print("- All predictions: all_predictions_50_runs.csv")
print("- Aggregate mean/std: summary_mean_std_across_50_runs.csv")
