import sys
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


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

# Fixed split files must be in the same directory as the dataset.
dataset_with_id_file = os.path.join(dataset_dir, f"{feature_basename}_dataset_with_ID.csv")
outer_split_file = os.path.join(dataset_dir, f"{feature_basename}_data_split.csv")
inner_split_file = os.path.join(dataset_dir, f"{feature_basename}_inner_data_split.csv")

output_dir = os.path.join(dataset_dir, f"{feature_basename}_{MODEL_NAME}_out")
os.makedirs(output_dir, exist_ok=True)
preds_dir = os.path.join(output_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)


# Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# Load data

df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# The fixed-ID dataset should contain ID as first column and target as last.
if "ID" in df.columns:
    ids = df["ID"].astype(str).values
    X_all = df.iloc[:, 1:-1].values
else:
    # Fallback for a feature file whose first column is already ID-like.
    ids = df.iloc[:, 0].astype(str).values
    X_all = df.iloc[:, 1:-1].values

y_all = df.iloc[:, -1].values.reshape(-1, 1)
N_FEATURES = X_all.shape[1]

if not os.path.exists(dataset_with_id_file):
    raise FileNotFoundError(
        f"Fixed-ID dataset not found: {dataset_with_id_file}\n"
        "Use the same dataset_with_ID.csv created by the fixed split generator."
    )
if not os.path.exists(outer_split_file):
    raise FileNotFoundError(
        f"Fixed outer split file not found: {outer_split_file}\n"
        "Use the shared data_split.csv created by the fixed split generator."
    )

id_to_pos = {str(v): i for i, v in enumerate(ids)}


# Validate and load fixed outer splits

outer_split = pd.read_csv(outer_split_file)
required_outer = {"Repetition", "Fold", "ID", "Set"}
if not required_outer.issubset(outer_split.columns):
    raise ValueError(f"Outer split file must contain columns: {sorted(required_outer)}")

outer_split["ID"] = outer_split["ID"].astype(str)

if set(outer_split["ID"]) != set(ids):
    missing = sorted(set(ids) - set(outer_split["ID"]))
    extra = sorted(set(outer_split["ID"]) - set(ids))
    raise ValueError(
        f"Outer split IDs do not match dataset IDs. Missing={missing[:5]}, Extra={extra[:5]}"
    )


# Create/load one shared inner split file
# This file is reused by ALL models.

def create_or_load_inner_splits():
    if os.path.exists(inner_split_file):
        inner = pd.read_csv(inner_split_file)
        required_inner = {"Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"}
        if not required_inner.issubset(inner.columns):
            raise ValueError(
                f"Inner split file must contain columns: {sorted(required_inner)}"
            )
        inner["ID"] = inner["ID"].astype(str)
        return inner

    rows = []
    for repetition in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            current = outer_split[
                (outer_split["Repetition"] == repetition)
                & (outer_split["Fold"] == outer_fold)
                & (outer_split["Set"].str.lower() == "train")
            ]
            outer_train_ids = current["ID"].tolist()

            if len(outer_train_ids) == 0:
                raise ValueError(
                    f"No outer-training IDs found for repetition {repetition}, fold {outer_fold}."
                )

            inner_seed = (BASE_SEED + repetition) * 100 + outer_fold
            kf = KFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=inner_seed
            )

            for inner_fold, (itr, iva) in enumerate(
                kf.split(np.arange(len(outer_train_ids))), start=1
            ):
                for pos in itr:
                    rows.append({
                        "Repetition": repetition,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": outer_train_ids[pos],
                        "Set": "train"
                    })
                for pos in iva:
                    rows.append({
                        "Repetition": repetition,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": outer_train_ids[pos],
                        "Set": "validation"
                    })

    inner = pd.DataFrame(rows)
    inner.to_csv(inner_split_file, index=False)
    print(f"Created shared inner split file: {inner_split_file}")
    return inner

inner_split = create_or_load_inner_splits()


# Validate inner split structure

def get_outer_train_ids(repetition, outer_fold):
    return outer_split[
        (outer_split["Repetition"] == repetition)
        & (outer_split["Fold"] == outer_fold)
        & (outer_split["Set"].str.lower() == "train")
    ]["ID"].tolist()

def get_outer_test_ids(repetition, outer_fold):
    return outer_split[
        (outer_split["Repetition"] == repetition)
        & (outer_split["Fold"] == outer_fold)
        & (outer_split["Set"].str.lower() == "test")
    ]["ID"].tolist()

def validate_inner_for_outer(repetition, outer_fold):
    outer_train = set(get_outer_train_ids(repetition, outer_fold))
    sub = inner_split[
        (inner_split["Repetition"] == repetition)
        & (inner_split["Outer_Fold"] == outer_fold)
    ]

    if len(sub) == 0:
        raise ValueError(f"No inner splits for repetition {repetition}, outer fold {outer_fold}.")

    if set(sub["ID"]) != outer_train:
        raise ValueError(
            f"Inner split IDs do not exactly match outer-train IDs for "
            f"repetition {repetition}, outer fold {outer_fold}."
        )

    for inner_fold in range(1, N_INNER_SPLITS + 1):
        f = sub[sub["Inner_Fold"] == inner_fold]
        train_ids = set(f.loc[f["Set"].str.lower() == "train", "ID"])
        val_ids = set(f.loc[f["Set"].str.lower() == "validation", "ID"])

        if train_ids & val_ids:
            raise ValueError(
                f"Overlap in inner fold {inner_fold}, repetition {repetition}, outer fold {outer_fold}."
            )
        if train_ids | val_ids != outer_train:
            raise ValueError(
                f"Inner fold {inner_fold} does not cover exactly the outer-training IDs."
            )


# DNN architecture/search space (same as original)

def build_dnn(activation, dropout_rate, l2_val, learning_rate,
              optimizer_name, n_layers, units, input_dim):
    model = Sequential()
    model.add(Dense(
        units, activation=activation,
        kernel_regularizer=l2(l2_val),
        input_dim=input_dim
    ))
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers - 1):
        model.add(Dense(
            units, activation=activation,
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

def sample_dnn_params(trial):
    activation = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid"]
    )
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_val = trial.suggest_float("l2_val", 1e-5, 1e-2, log=True)

    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3

    learning_rate = trial.suggest_float(
        "learning_rate", lr_min, lr_max, log=True
    )
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    n_layers = trial.suggest_int("n_layers", 3, 7)
    units = trial.suggest_categorical("units", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    return {
        "activation": activation,
        "dropout_rate": dropout_rate,
        "l2_val": l2_val,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "n_layers": n_layers,
        "units": units,
        "batch_size": batch_size,
    }


# Inner objective

def make_inner_objective(
    X_outer_train, y_outer_train,
    outer_train_ids, repetition, outer_fold, n_features
):
    sub = inner_split[
        (inner_split["Repetition"] == repetition)
        & (inner_split["Outer_Fold"] == outer_fold)
    ]

    id_to_local = {str(v): i for i, v in enumerate(outer_train_ids)}

    fold_indices = []
    for inner_fold in range(1, N_INNER_SPLITS + 1):
        f = sub[sub["Inner_Fold"] == inner_fold]
        train_ids = f.loc[f["Set"].str.lower() == "train", "ID"].astype(str).tolist()
        val_ids = f.loc[f["Set"].str.lower() == "validation", "ID"].astype(str).tolist()

        inner_train_idx = np.array([id_to_local[x] for x in train_ids], dtype=int)
        inner_val_idx = np.array([id_to_local[x] for x in val_ids], dtype=int)
        fold_indices.append((inner_train_idx, inner_val_idx))

    def objective(trial):
        params = sample_dnn_params(trial)
        fold_scores = []

        for inner_fold_number, (inner_train_idx, inner_val_idx) in enumerate(
            fold_indices, start=1
        ):
            set_seed(
                BASE_SEED
                + repetition * 10000
                + outer_fold * 100
                + trial.number * 10
                + inner_fold_number
            )

            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            # Leakage-free scaling: fit only on inner training data.
            sx = StandardScaler().fit(X_inner_train)
            sy = StandardScaler().fit(y_inner_train)

            X_inner_train_s = sx.transform(X_inner_train)
            X_inner_val_s = sx.transform(X_inner_val)
            y_inner_train_s = sy.transform(y_inner_train)
            y_inner_val_s = sy.transform(y_inner_val)

            model = build_dnn(
                params["activation"],
                params["dropout_rate"],
                params["l2_val"],
                params["learning_rate"],
                params["optimizer"],
                params["n_layers"],
                params["units"],
                n_features
            )

            model.fit(
                X_inner_train_s, y_inner_train_s,
                epochs=100,
                batch_size=params["batch_size"],
                verbose=0
            )

            y_pred = model.predict(X_inner_val_s, verbose=0).flatten()
            fold_scores.append(
                mean_squared_error(y_inner_val_s.flatten(), y_pred)
            )
            tf.keras.backend.clear_session()

        return float(np.mean(fold_scores))

    return objective


# Repeated nested CV

all_metrics = []
all_best_params = []

for repetition in range(1, N_REPETITIONS + 1):
    outer_train_ids = get_outer_train_ids(repetition, 1)  # overwritten below
    for outer_fold in range(1, N_OUTER_SPLITS + 1):
        print(
            f"\n===== Repetition {repetition}/{N_REPETITIONS} | "
            f"Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
        )

        validate_inner_for_outer(repetition, outer_fold)

        outer_train_ids = get_outer_train_ids(repetition, outer_fold)
        outer_test_ids = get_outer_test_ids(repetition, outer_fold)

        train_pos = [id_to_pos[x] for x in outer_train_ids]
        test_pos = [id_to_pos[x] for x in outer_test_ids]

        X_outer_train = X_all[train_pos]
        X_outer_test = X_all[test_pos]
        y_outer_train = y_all[train_pos]
        y_outer_test = y_all[test_pos]

        # Hyperparameter search uses ONLY fixed inner folds within outer train.
        rep_seed = BASE_SEED + repetition
        set_seed(rep_seed)

        objective = make_inner_objective(
            X_outer_train, y_outer_train,
            outer_train_ids, repetition, outer_fold, N_FEATURES
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=rep_seed)
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_trial.params

        # Final scaler is fitted ONLY on the full outer-training set.
        scaler_X = StandardScaler().fit(X_outer_train)
        scaler_y = StandardScaler().fit(y_outer_train)

        X_outer_train_s = scaler_X.transform(X_outer_train)
        X_outer_test_s = scaler_X.transform(X_outer_test)
        y_outer_train_s = scaler_y.transform(y_outer_train)
        y_outer_test_s = scaler_y.transform(y_outer_test)

        # Final model is trained on all outer-training samples.
        set_seed(rep_seed * 100 + outer_fold)
        final_model = build_dnn(
            best_params["activation"],
            best_params["dropout_rate"],
            best_params["l2_val"],
            best_params["learning_rate"],
            best_params["optimizer"],
            best_params["n_layers"],
            best_params["units"],
            N_FEATURES
        )

        final_model.fit(
            X_outer_train_s, y_outer_train_s,
            epochs=100,
            batch_size=best_params["batch_size"],
            verbose=0
        )

        # Outer test is used once for final evaluation.
        y_test_pred_s = final_model.predict(X_outer_test_s, verbose=0).flatten()

        mse = mean_squared_error(y_outer_test_s.flatten(), y_test_pred_s)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(
            y_outer_test_s.flatten(), y_test_pred_s
        )
        corr, _ = pearsonr(
            y_outer_test_s.flatten(), y_test_pred_s
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
        best_params_with_meta.update({
            "Repetition": repetition,
            "Outer_Fold": outer_fold
        })
        all_best_params.append(best_params_with_meta)

        # Save predictions in original target scale.
        y_test_pred_orig = scaler_y.inverse_transform(
            y_test_pred_s.reshape(-1, 1)
        ).flatten()
        y_test_actual_orig = y_outer_test.flatten()

        preds_df = pd.DataFrame({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "ID": outer_test_ids,
            "Row_Index": test_pos,
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

        # Save Optuna trial history for this outer run.
        trials_df = study.trials_dataframe()
        trials_df.to_csv(
            os.path.join(
                output_dir,
                f"optuna_trials_rep{repetition}_fold{outer_fold}.csv"
            ),
            index=False
        )

        pd.DataFrame([best_params]).to_csv(
            os.path.join(
                output_dir,
                f"best_trial_rep{repetition}_fold{outer_fold}.csv"
            ),
            index=False
        )

        tf.keras.backend.clear_session()


# Aggregate results

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
    os.path.join(output_dir, "all_best_hyperparameters_50_runs.csv"),
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
    os.path.join(output_dir, "summary_mean_std_across_50_runs.csv")
)

print("\n===== Summary across all 50 runs (10 repetitions x 5 outer folds) =====")
print(summary)
print(f"\nAll {N_REPETITIONS * N_OUTER_SPLITS} runs complete.")
print("- Per-run metrics:        all_metrics_50_runs.csv")
print("- Per-run best params:    all_best_hyperparameters_50_runs.csv")
print("- Per-run predictions:    predictions/predictions_rep{{r}}_fold{{f}}.csv")
print("- All predictions merged: all_predictions_50_runs.csv")
print("- Aggregate mean/std:     summary_mean_std_across_50_runs.csv")
