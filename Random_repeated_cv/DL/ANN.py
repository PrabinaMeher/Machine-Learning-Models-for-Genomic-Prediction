#!/usr/bin/env python3
import os
import sys
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 100
INNER_EPOCHS = 100
FINAL_EPOCHS = 100
BASE_SEED = 1000


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_dataset(dataset_file):
    df = pd.read_csv(dataset_file)
    if df.empty:
        raise ValueError("The dataset is empty.")
    if "ID" not in df.columns:
        raise ValueError("Dataset must contain the permanent ID column.")
    if df["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found.")
    if df["ID"].isna().any():
        raise ValueError("Missing IDs found.")

    # ID = identifier only; last column = target.
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    ids = df["ID"].astype(str).values
    return df, X, y, ids


def load_fixed_splits(split_file, ids):
    splits = pd.read_csv(split_file)
    required = {"Repetition", "Fold", "ID", "Set"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")

    splits["ID"] = splits["ID"].astype(str)
    dataset_ids = set(ids)

    if set(splits["ID"]) != dataset_ids:
        raise ValueError("Dataset IDs and split-file IDs do not match.")

    expected = N_REPETITIONS * N_OUTER_SPLITS
    actual = splits[["Repetition", "Fold"]].drop_duplicates().shape[0]
    if actual != expected:
        raise ValueError(f"Expected {expected} outer splits, found {actual}.")

    for rep in range(1, N_REPETITIONS + 1):
        for fold in range(1, N_OUTER_SPLITS + 1):
            cur = splits[(splits["Repetition"] == rep) &
                         (splits["Fold"] == fold)]
            train = set(cur.loc[cur["Set"] == "train", "ID"])
            test = set(cur.loc[cur["Set"] == "test", "ID"])
            if train & test:
                raise ValueError(f"Train/test overlap: rep {rep}, fold {fold}")
            if train | test != dataset_ids:
                raise ValueError(f"IDs incomplete: rep {rep}, fold {fold}")

    print(f"Fixed split validation passed: {expected} outer runs.")
    return splits


def create_ann_model(trial, input_dim):
    activation = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_val = trial.suggest_float("l2_val", 1e-5, 1e-2, log=True)

    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3

    learning_rate = trial.suggest_float(
        "learning_rate", lr_min, lr_max, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_categorical("units", [32, 64, 128])

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation=activation,
              kernel_regularizer=l2(l2_val)),
        Dropout(dropout_rate)
    ])

    for _ in range(n_layers - 1):
        model.add(Dense(units, activation=activation,
                        kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss="mse")
    return model


def make_inner_objective(X_train, y_train, inner_seed, rep, outer_fold):
    inner_kf = KFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=inner_seed)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        scores = []

        for inner_fold, (tr_idx, va_idx) in enumerate(
                inner_kf.split(X_train), start=1):

            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr)

            X_tr_s = sx.transform(X_tr)
            X_va_s = sx.transform(X_va)
            y_tr_s = sy.transform(y_tr)
            y_va_s = sy.transform(y_va)

            set_seed(BASE_SEED + rep * 100000 +
                     outer_fold * 1000 + trial.number * 10 + inner_fold)

            model = create_ann_model(trial, X_train.shape[1])
            model.fit(X_tr_s, y_tr_s,
                      epochs=INNER_EPOCHS,
                      batch_size=batch_size,
                      verbose=0)

            pred = model.predict(X_va_s, verbose=0).flatten()
            scores.append(mean_squared_error(y_va_s.flatten(), pred))
            tf.keras.backend.clear_session()

        return float(np.mean(scores))

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="ANN using fixed 10 x 5 outer splits.")
    parser.add_argument("dataset_file",
                        help="*_dataset_with_ID.csv")
    parser.add_argument("split_file",
                        help="*_data_split.csv")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    dataset_file = os.path.abspath(args.dataset_file)
    split_file = os.path.abspath(args.split_file)

    name = os.path.splitext(os.path.basename(dataset_file))[0]
    if name.endswith("_dataset_with_ID"):
        name = name[:-len("_dataset_with_ID")]

    output_dir = (os.path.abspath(args.output_dir)
                  if args.output_dir else
                  os.path.join(os.path.dirname(dataset_file),
                               f"{name}_ANN_fixed_splits_out"))

    predictions_dir = os.path.join(output_dir, "predictions")
    optuna_dir = os.path.join(output_dir, "optuna_trials")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)

    print("=" * 75)
    print("ANN - FIXED REPEATED NESTED CROSS-VALIDATION")
    print("=" * 75)
    print(f"Dataset: {dataset_file}")
    print(f"Splits : {split_file}")
    print(f"Output : {output_dir}")
    print("Outer : 10 repetitions x 5 fixed folds = 50 runs")
    print(f"Inner : {N_INNER_SPLITS}-fold CV")
    print(f"Optuna: {N_TRIALS} trials per outer fold")

    df, X_all, y_all, ids = load_dataset(dataset_file)
    splits = load_fixed_splits(split_file, ids)

    id_to_index = {sid: i for i, sid in enumerate(ids)}

    all_metrics = []
    all_params = []
    all_predictions = []

    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):

            print(f"\n===== Repetition {rep}/{N_REPETITIONS} | "
                  f"Outer fold {outer_fold}/{N_OUTER_SPLITS} =====")

            cur = splits[(splits["Repetition"] == rep) &
                         (splits["Fold"] == outer_fold)]

            train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
            test_ids = cur.loc[cur["Set"] == "test", "ID"].tolist()

            # EXACT train/test assignment from the fixed split file.
            train_idx = np.array([id_to_index[x] for x in train_ids])
            test_idx = np.array([id_to_index[x] for x in test_ids])

            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            rep_seed = BASE_SEED + rep
            inner_seed = rep_seed * 100 + outer_fold

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=rep_seed))

            study.optimize(
                make_inner_objective(
                    X_tr, y_tr, inner_seed, rep, outer_fold),
                n_trials=N_TRIALS,
                show_progress_bar=False)

            best = study.best_trial.params

            # Save every Optuna trial for this outer fold.
            trials_df = study.trials_dataframe(
                attrs=("number", "value", "state", "params"))
            trials_df.insert(0, "Repetition", rep)
            trials_df.insert(1, "Outer_Fold", outer_fold)
            trials_df.to_csv(
                os.path.join(
                    optuna_dir,
                    f"optuna_trials_rep{rep}_fold{outer_fold}.csv"),
                index=False)

            # Save best Optuna result.
            pd.DataFrame([{
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "Best_Trial": study.best_trial.number,
                "Best_Inner_MSE": study.best_value,
                **best
            }]).to_csv(
                os.path.join(
                    optuna_dir,
                    f"optuna_best_rep{rep}_fold{outer_fold}.csv"),
                index=False)

            # Scale using outer-training data only.
            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr)

            X_tr_s = sx.transform(X_tr)
            X_te_s = sx.transform(X_te)
            y_tr_s = sy.transform(y_tr)
            y_te_s = sy.transform(y_te)

            set_seed(BASE_SEED + rep * 100 + outer_fold)

            final_model = create_ann_model(
                optuna.trial.FixedTrial(best), X_all.shape[1])

            final_model.fit(
                X_tr_s, y_tr_s,
                epochs=FINAL_EPOCHS,
                batch_size=best["batch_size"],
                verbose=0)

            pred_s = final_model.predict(X_te_s, verbose=0).flatten()

            mse = mean_squared_error(y_te_s.flatten(), pred_s)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(
                y_te_s.flatten(), pred_s)

            if np.std(y_te_s) > 0 and np.std(pred_s) > 0:
                corr, _ = pearsonr(y_te_s.flatten(), pred_s)
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
                **best
            })

            pred_orig = sy.inverse_transform(
                pred_s.reshape(-1, 1)).flatten()

            pred_df = pd.DataFrame({
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "ID": test_ids,
                "Row_Index": test_idx,
                "Actual": y_te.flatten(),
                "Predicted": pred_orig
            })

            pred_df.to_csv(
                os.path.join(
                    predictions_dir,
                    f"predictions_rep{rep}_fold{outer_fold}.csv"),
                index=False)

            all_predictions.append(pred_df)

            print(f"Best inner MSE: {study.best_value:.6f}")
            print(f"Test MSE={mse:.6f} | RMSE={rmse:.6f} | "
                  f"MAPE={mape:.6f} | Correlation={corr:.6f}")

            tf.keras.backend.clear_session()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        os.path.join(output_dir, "all_metrics_50_runs.csv"), index=False)

    params_df = pd.DataFrame(all_params)
    params_df.to_csv(
        os.path.join(output_dir,
                     "all_best_hyperparameters_50_runs.csv"),
        index=False)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        os.path.join(output_dir, "all_predictions_50_runs.csv"),
        index=False)

    summary = metrics_df[
        ["MSE", "RMSE", "MAPE", "Correlation"]
    ].agg(["mean", "std"]).T
    summary.columns = ["Mean", "Std"]
    summary.to_csv(
        os.path.join(output_dir,
                     "summary_mean_std_across_50_runs.csv"))

    print("\n" + "=" * 75)
    print("ANN ANALYSIS COMPLETED")
    print("=" * 75)
    print(summary)
    print("\nSaved:")
    print("  all_metrics_50_runs.csv")
    print("  all_best_hyperparameters_50_runs.csv")
    print("  all_predictions_50_runs.csv")
    print("  summary_mean_std_across_50_runs.csv")
    print("  predictions/")
    print("  optuna_trials/")


if __name__ == "__main__":
    main()
