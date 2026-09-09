#!/usr/bin/env python3
import os
import sys
import random
import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# CONFIGURATION

N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 100
INNER_EPOCHS = 100
FINAL_EPOCHS = 100
BASE_SEED = 1000



# REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)



# DATASET

def load_dataset(dataset_file):
    df = pd.read_csv(dataset_file)

    if df.empty:
        raise ValueError("The dataset is empty.")
    if "ID" not in df.columns:
        raise ValueError("Dataset must contain the permanent ID column.")
    if df["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in dataset.")
    if df["ID"].isna().any():
        raise ValueError("Missing IDs found in dataset.")

    # ID is identifier only; last column is phenotype/target.
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    ids = df["ID"].astype(str).values

    return df, X, y, ids



# FIXED OUTER SPLITS

def load_fixed_outer_splits(split_file, ids):
    splits = pd.read_csv(split_file)
    required = {"Repetition", "Fold", "ID", "Set"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Outer split file missing columns: {sorted(missing)}")

    splits["ID"] = splits["ID"].astype(str)
    dataset_ids = set(ids)

    if set(splits["ID"]) != dataset_ids:
        raise ValueError("Dataset IDs and outer split-file IDs do not match.")

    expected = N_REPETITIONS * N_OUTER_SPLITS
    actual = splits[["Repetition", "Fold"]].drop_duplicates().shape[0]
    if actual != expected:
        raise ValueError(
            f"Expected {expected} outer splits, found {actual}."
        )

    for rep in range(1, N_REPETITIONS + 1):
        for fold in range(1, N_OUTER_SPLITS + 1):
            cur = splits[
                (splits["Repetition"] == rep) &
                (splits["Fold"] == fold)
            ]
            train = set(cur.loc[cur["Set"] == "train", "ID"])
            test = set(cur.loc[cur["Set"] == "test", "ID"])

            if train & test:
                raise ValueError(
                    f"Outer train/test overlap: repetition {rep}, fold {fold}"
                )
            if train | test != dataset_ids:
                raise ValueError(
                    f"Outer IDs incomplete: repetition {rep}, fold {fold}"
                )

    print(f"Fixed outer split validation passed: {expected} runs.")
    return splits



# FIXED INNER SPLITS

def build_or_load_inner_splits(inner_split_file, outer_splits, ids):
    if os.path.exists(inner_split_file):
        inner = pd.read_csv(inner_split_file)
        required = {"Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"}
        missing = required - set(inner.columns)
        if missing:
            raise ValueError(
                f"Inner split file missing columns: {sorted(missing)}"
            )
        inner["ID"] = inner["ID"].astype(str)
        validate_inner_splits(inner, outer_splits, ids)
        print(f"Using existing fixed inner split file: {inner_split_file}")
        return inner

    print("Inner split file not found. Creating it once...")
    id_to_index = {sid: i for i, sid in enumerate(ids)}
    records = []

    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            cur = outer_splits[
                (outer_splits["Repetition"] == rep) &
                (outer_splits["Fold"] == outer_fold)
            ]
            outer_train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
            outer_train_idx = np.array([id_to_index[x] for x in outer_train_ids])

            # Deterministic seed unique to each repetition/outer fold.
            inner_seed = (BASE_SEED + rep) * 100 + outer_fold
            inner_kf = KFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=inner_seed
            )

            for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(
                inner_kf.split(outer_train_idx), start=1
            ):
                for pos in inner_train_pos:
                    records.append({
                        "Repetition": rep,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": outer_train_ids[int(pos)],
                        "Set": "train"
                    })

                for pos in inner_val_pos:
                    records.append({
                        "Repetition": rep,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": outer_train_ids[int(pos)],
                        "Set": "validation"
                    })

    inner = pd.DataFrame(
        records,
        columns=["Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"]
    )
    inner.to_csv(inner_split_file, index=False)
    validate_inner_splits(inner, outer_splits, ids)
    print(f"Saved fixed inner splits: {inner_split_file}")
    return inner


def validate_inner_splits(inner, outer_splits, ids):
    dataset_ids = set(ids)

    expected_combinations = N_REPETITIONS * N_OUTER_SPLITS * N_INNER_SPLITS
    actual_combinations = (
        inner[["Repetition", "Outer_Fold", "Inner_Fold"]]
        .drop_duplicates()
        .shape[0]
    )
    if actual_combinations != expected_combinations:
        raise ValueError(
            f"Expected {expected_combinations} inner folds, "
            f"found {actual_combinations}."
        )

    if not set(inner["Set"]).issubset({"train", "validation"}):
        raise ValueError("Inner split Set must contain only train/validation.")

    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            outer_cur = outer_splits[
                (outer_splits["Repetition"] == rep) &
                (outer_splits["Fold"] == outer_fold)
            ]
            outer_train = set(
                outer_cur.loc[outer_cur["Set"] == "train", "ID"]
            )

            inner_cur = inner[
                (inner["Repetition"] == rep) &
                (inner["Outer_Fold"] == outer_fold)
            ]

            if set(inner_cur["ID"]) != outer_train:
                raise ValueError(
                    f"Inner IDs do not exactly match outer-train IDs: "
                    f"rep {rep}, outer fold {outer_fold}"
                )

            for inner_fold in range(1, N_INNER_SPLITS + 1):
                cur = inner_cur[inner_cur["Inner_Fold"] == inner_fold]
                tr = set(cur.loc[cur["Set"] == "train", "ID"])
                va = set(cur.loc[cur["Set"] == "validation", "ID"])

                if tr & va:
                    raise ValueError(
                        f"Inner train/validation overlap: rep {rep}, "
                        f"outer {outer_fold}, inner {inner_fold}"
                    )
                if tr | va != outer_train:
                    raise ValueError(
                        f"Inner IDs incomplete: rep {rep}, "
                        f"outer {outer_fold}, inner {inner_fold}"
                    )

    if not set(inner["ID"]).issubset(dataset_ids):
        raise ValueError("Inner split contains unknown dataset IDs.")

    print(
        f"Fixed inner split validation passed: "
        f"{expected_combinations} inner folds."
    )



# BiLSTM MODEL

def reshape_for_bilstm(X_2d):
    return X_2d.reshape((X_2d.shape[0], X_2d.shape[1], 1))


def build_bilstm(
    activation,
    dropout_rate,
    l2_val,
    learning_rate,
    layer_sizes,
    optimizer_name,
    input_shape
):
    model = Sequential()
    units = layer_sizes[0]
    return_sequences = len(layer_sizes) > 1

    model.add(Bidirectional(
        LSTM(
            units,
            activation=activation,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_val)
        ),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))

    for units in layer_sizes[1:-1]:
        model.add(Bidirectional(
            LSTM(
                units,
                activation=activation,
                return_sequences=True,
                kernel_regularizer=l2(l2_val)
            )
        ))
        model.add(Dropout(dropout_rate))

    if len(layer_sizes) > 1:
        units = layer_sizes[-1]
        model.add(Bidirectional(
            LSTM(
                units,
                activation=activation,
                return_sequences=False,
                kernel_regularizer=l2(l2_val)
            )
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


def create_bilstm_model(trial, input_shape):
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

    return build_bilstm(
        activation,
        dropout_rate,
        l2_val,
        learning_rate,
        layer_sizes,
        optimizer_name,
        input_shape
    )


def create_bilstm_model_from_params(params, input_shape):
    return build_bilstm(
        params["activation"],
        params["dropout_rate"],
        params["l2_val"],
        params["learning_rate"],
        params["layer_sizes"],
        params["optimizer"],
        input_shape
    )



# INNER OPTUNA OBJECTIVE

def make_inner_objective(
    X_outer_train,
    y_outer_train,
    ids_outer_train,
    inner_splits,
    rep,
    outer_fold
):
    id_to_local_index = {
        sid: i for i, sid in enumerate(ids_outer_train)
    }

    fold_indices = []
    cur_all = inner_splits[
        (inner_splits["Repetition"] == rep) &
        (inner_splits["Outer_Fold"] == outer_fold)
    ]

    for inner_fold in range(1, N_INNER_SPLITS + 1):
        cur = cur_all[cur_all["Inner_Fold"] == inner_fold]

        train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
        val_ids = cur.loc[cur["Set"] == "validation", "ID"].tolist()

        train_idx = np.array([id_to_local_index[x] for x in train_ids])
        val_idx = np.array([id_to_local_index[x] for x in val_ids])

        fold_indices.append((inner_fold, train_idx, val_idx))

    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )
        fold_scores = []

        for inner_fold, inner_train_idx, inner_val_idx in fold_indices:
            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            # Leakage-free scaling: fit only on inner-training data.
            sx = StandardScaler().fit(X_inner_train)
            sy = StandardScaler().fit(y_inner_train)

            X_inner_train_s = reshape_for_bilstm(
                sx.transform(X_inner_train)
            )
            X_inner_val_s = reshape_for_bilstm(
                sx.transform(X_inner_val)
            )
            y_inner_train_s = sy.transform(y_inner_train)
            y_inner_val_s = sy.transform(y_inner_val)

            # Unique deterministic seed for trial + inner fold.
            seed = (
                BASE_SEED
                + rep * 100000
                + outer_fold * 1000
                + trial.number * 10
                + inner_fold
            )
            set_seed(seed)

            tf.keras.backend.clear_session()
            model = create_bilstm_model(
                trial, input_shape=(X_outer_train.shape[1], 1)
            )

            model.fit(
                X_inner_train_s,
                y_inner_train_s,
                epochs=INNER_EPOCHS,
                batch_size=batch_size,
                verbose=0
            )

            pred = model.predict(X_inner_val_s, verbose=0).flatten()

            if np.isnan(pred).any() or np.isinf(pred).any():
                tf.keras.backend.clear_session()
                return np.inf

            score = mean_squared_error(
                y_inner_val_s.flatten(), pred
            )
            fold_scores.append(score)

            del model
            del sx, sy
            del X_inner_train_s, X_inner_val_s
            del y_inner_train_s, y_inner_val_s, pred
            tf.keras.backend.clear_session()
            gc.collect()

        # ONE objective value for the trial = mean across all 5 inner folds.
        return float(np.mean(fold_scores))

    return objective



# MAIN

def main():
    parser = argparse.ArgumentParser(
        description=(
            "BiLSTM using fixed 10 x 5 outer CV and fixed 5-fold "
            "inner CV."
        )
    )
    parser.add_argument(
        "dataset_file",
        help="*_dataset_with_ID.csv"
    )
    parser.add_argument(
        "outer_split_file",
        help="*_data_split.csv"
    )
    parser.add_argument(
        "--inner_split_file",
        default=None,
        help="Fixed inner split CSV. Created once if it does not exist."
    )
    parser.add_argument(
        "--output_dir",
        default=None
    )
    args = parser.parse_args()

    dataset_file = os.path.abspath(args.dataset_file)
    outer_split_file = os.path.abspath(args.outer_split_file)

    name = os.path.splitext(os.path.basename(dataset_file))[0]
    if name.endswith("_dataset_with_ID"):
        name = name[:-len("_dataset_with_ID")]

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(
            os.path.dirname(dataset_file),
            f"{name}_BiLSTM_fixed_splits_out"
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.inner_split_file:
        inner_split_file = os.path.abspath(args.inner_split_file)
    else:
        # Keep the shared split beside the outer split file.
        inner_split_file = os.path.join(
            os.path.dirname(outer_split_file),
            "inner_data_split.csv"
        )

    predictions_dir = os.path.join(output_dir, "predictions")
    optuna_dir = os.path.join(output_dir, "optuna_trials")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)

    print("=" * 75)
    print("BiLSTM - FIXED REPEATED NESTED CROSS-VALIDATION")
    print("=" * 75)
    print(f"Dataset      : {dataset_file}")
    print(f"Outer splits : {outer_split_file}")
    print(f"Inner splits : {inner_split_file}")
    print(f"Output       : {output_dir}")
    print("Outer        : 10 repetitions x 5 fixed folds = 50 runs")
    print(f"Inner        : {N_INNER_SPLITS}-fold fixed CV")
    print(f"Optuna       : {N_TRIALS} trials per outer fold")
    print("=" * 75)

    df, X_all, y_all, ids_all = load_dataset(dataset_file)
    outer_splits = load_fixed_outer_splits(outer_split_file, ids_all)
    inner_splits = build_or_load_inner_splits(
        inner_split_file,
        outer_splits,
        ids_all
    )

    id_to_index = {sid: i for i, sid in enumerate(ids_all)}

    all_metrics = []
    all_params = []
    all_predictions = []

    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            print(
                f"\n===== Repetition {rep}/{N_REPETITIONS} | "
                f"Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
            )

            cur = outer_splits[
                (outer_splits["Repetition"] == rep) &
                (outer_splits["Fold"] == outer_fold)
            ]

            train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
            test_ids = cur.loc[cur["Set"] == "test", "ID"].tolist()

            # EXACT fixed outer train/test assignment.
            train_idx = np.array([id_to_index[x] for x in train_ids])
            test_idx = np.array([id_to_index[x] for x in test_ids])

            X_outer_train = X_all[train_idx]
            X_outer_test = X_all[test_idx]
            y_outer_train = y_all[train_idx]
            y_outer_test = y_all[test_idx]

           
            # OPTUNA: each trial is evaluated on the SAME 5 inner folds.
            # One trial gets ONE objective = mean of its 5 inner MSEs.
           
            rep_seed = BASE_SEED + rep
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=rep_seed)
            )

            study.optimize(
                make_inner_objective(
                    X_outer_train,
                    y_outer_train,
                    train_ids,
                    inner_splits,
                    rep,
                    outer_fold
                ),
                n_trials=N_TRIALS,
                show_progress_bar=False
            )

            best = study.best_trial.params

            # Save all Optuna trials.
            trials_df = study.trials_dataframe(
                attrs=("number", "value", "state", "params")
            )
            trials_df.insert(0, "Repetition", rep)
            trials_df.insert(1, "Outer_Fold", outer_fold)
            trials_df.to_csv(
                os.path.join(
                    optuna_dir,
                    f"optuna_trials_rep{rep}_fold{outer_fold}.csv"
                ),
                index=False
            )

            # Save best Optuna result.
            best_save = best.copy()
            best_save["layer_sizes"] = str(best_save["layer_sizes"])
            pd.DataFrame([{
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "Best_Trial": study.best_trial.number,
                "Best_Inner_MSE": study.best_value,
                **best_save
            }]).to_csv(
                os.path.join(
                    optuna_dir,
                    f"optuna_best_rep{rep}_fold{outer_fold}.csv"
                ),
                index=False
            )

           
            # FINAL OUTER MODEL
            # Fit scalers ONLY on all outer-training samples.
           
            sx = StandardScaler().fit(X_outer_train)
            sy = StandardScaler().fit(y_outer_train)

            X_outer_train_s = reshape_for_bilstm(
                sx.transform(X_outer_train)
            )
            X_outer_test_s = reshape_for_bilstm(
                sx.transform(X_outer_test)
            )
            y_outer_train_s = sy.transform(y_outer_train)
            y_outer_test_s = sy.transform(y_outer_test)

            set_seed(BASE_SEED + rep * 100 + outer_fold)
            tf.keras.backend.clear_session()

            final_model = create_bilstm_model_from_params(
                best,
                input_shape=(X_all.shape[1], 1)
            )

            final_model.fit(
                X_outer_train_s,
                y_outer_train_s,
                epochs=FINAL_EPOCHS,
                batch_size=best["batch_size"],
                verbose=0
            )

            pred_s = final_model.predict(
                X_outer_test_s,
                verbose=0
            ).flatten()

            if np.isnan(pred_s).any() or np.isinf(pred_s).any():
                raise RuntimeError(
                    f"NaN/Inf predictions in rep {rep}, outer fold {outer_fold}"
                )

           
            # OUTER TEST METRICS
           
            mse = mean_squared_error(
                y_outer_test_s.flatten(), pred_s
            )
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(
                y_outer_test_s.flatten(), pred_s
            )

            if (
                np.std(y_outer_test_s) > 0 and
                np.std(pred_s) > 0
            ):
                corr, _ = pearsonr(
                    y_outer_test_s.flatten(), pred_s
                )
            else:
                corr = np.nan

            all_metrics.append({
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "Correlation": corr,
                "Best_Inner_MSE": study.best_value,
                "Best_Trial": study.best_trial.number
            })

            all_params.append({
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "Best_Inner_MSE": study.best_value,
                "Best_Trial": study.best_trial.number,
                **best_save
            })

            pred_orig = sy.inverse_transform(
                pred_s.reshape(-1, 1)
            ).flatten()

            pred_df = pd.DataFrame({
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "ID": test_ids,
                "Row_Index": test_idx,
                "Actual": y_outer_test.flatten(),
                "Predicted": pred_orig
            })

            pred_df.to_csv(
                os.path.join(
                    predictions_dir,
                    f"predictions_rep{rep}_fold{outer_fold}.csv"
                ),
                index=False
            )
            all_predictions.append(pred_df)

            print(f"Best inner MSE: {study.best_value:.6f}")
            print(
                f"Outer-test MSE={mse:.6f} | "
                f"RMSE={rmse:.6f} | "
                f"MAPE={mape:.6f} | "
                f"Correlation={corr:.6f}"
            )

            del final_model
            del sx, sy
            del X_outer_train_s, X_outer_test_s
            del y_outer_train_s, y_outer_test_s, pred_s
            tf.keras.backend.clear_session()
            gc.collect()

    
    # FINAL OUTPUTS
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        os.path.join(output_dir, "all_metrics_50_runs.csv"),
        index=False
    )

    params_df = pd.DataFrame(all_params)
    params_df.to_csv(
        os.path.join(output_dir, "all_best_hyperparameters_50_runs.csv"),
        index=False
    )

    pd.concat(all_predictions, ignore_index=True).to_csv(
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

    print("\n" + "=" * 75)
    print("BiLSTM ANALYSIS COMPLETED")
    print("=" * 75)
    print(summary)
    print("\nSaved:")
    print("  all_metrics_50_runs.csv")
    print("  all_best_hyperparameters_50_runs.csv")
    print("  all_predictions_50_runs.csv")
    print("  summary_mean_std_across_50_runs.csv")
    print("  predictions/")
    print("  optuna_trials/")
    print(f"  Fixed inner split file: {inner_split_file}")


if __name__ == "__main__":
    main()
