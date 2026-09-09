#!/usr/bin/env python3
import os
import sys
import random
import argparse
import warnings
import gc
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
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
        print(f"GPU(s) detected and enabled: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"Could not set memory growth (must be set before initialization): {e}")
else:
    print("No GPU detected - running on CPU.")

# Configuration
N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_TRIALS = 100
INNER_EPOCHS = 100
FINAL_EPOCHS = 100
BASE_SEED = 1000


def set_seed(seed):
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
        raise ValueError(f"Expected {expected} outer splits, found {actual}.")

    for rep in range(1, N_REPETITIONS + 1):
        for fold in range(1, N_OUTER_SPLITS + 1):
            cur = splits[(splits["Repetition"] == rep) &
                         (splits["Fold"] == fold)]
            train = set(cur.loc[cur["Set"] == "train", "ID"])
            test = set(cur.loc[cur["Set"] == "test", "ID"])
            if train & test:
                raise ValueError(f"Outer train/test overlap: rep {rep}, fold {fold}")
            if train | test != dataset_ids:
                raise ValueError(f"Outer IDs incomplete: rep {rep}, fold {fold}")

    print(f"Fixed outer split validation passed: {expected} outer runs.")
    return splits


def generate_or_load_fixed_inner_splits(inner_split_file, outer_splits, ids):
    """Create deterministic inner folds once, or reuse an existing shared file."""
    if os.path.exists(inner_split_file):
        inner = pd.read_csv(inner_split_file)
        required = {"Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"}
        missing = required - set(inner.columns)
        if missing:
            raise ValueError(f"Inner split file missing columns: {sorted(missing)}")
        inner["ID"] = inner["ID"].astype(str)
        if not set(inner["ID"]).issubset(set(ids)):
            raise ValueError("Inner split file contains IDs not present in the dataset.")
    else:
        records = []
        id_set = set(ids)
        for rep in range(1, N_REPETITIONS + 1):
            for outer_fold in range(1, N_OUTER_SPLITS + 1):
                cur = outer_splits[(outer_splits["Repetition"] == rep) &
                                   (outer_splits["Fold"] == outer_fold)]
                outer_train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
                if len(outer_train_ids) < N_INNER_SPLITS:
                    raise ValueError("Not enough outer-training samples for inner CV.")

                # Deterministic seed unique to each repetition/outer fold.
                inner_seed = (BASE_SEED + rep) * 100 + outer_fold
                kf = KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=inner_seed)
                outer_train_ids = np.asarray(outer_train_ids, dtype=str)

                for inner_fold, (tr_idx, va_idx) in enumerate(kf.split(outer_train_ids), start=1):
                    for idx in tr_idx:
                        records.append({
                            "Repetition": rep,
                            "Outer_Fold": outer_fold,
                            "Inner_Fold": inner_fold,
                            "ID": outer_train_ids[idx],
                            "Set": "train"
                        })
                    for idx in va_idx:
                        records.append({
                            "Repetition": rep,
                            "Outer_Fold": outer_fold,
                            "Inner_Fold": inner_fold,
                            "ID": outer_train_ids[idx],
                            "Set": "validation"
                        })

        inner = pd.DataFrame(records, columns=[
            "Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"
        ])
        os.makedirs(os.path.dirname(os.path.abspath(inner_split_file)), exist_ok=True)
        inner.to_csv(inner_split_file, index=False)
        print(f"Created fixed inner split file: {inner_split_file}")

    # Validate every inner split against the corresponding outer-training IDs.
    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            outer_cur = outer_splits[(outer_splits["Repetition"] == rep) &
                                     (outer_splits["Fold"] == outer_fold)]
            outer_train = set(outer_cur.loc[outer_cur["Set"] == "train", "ID"])
            cur = inner[(inner["Repetition"] == rep) &
                        (inner["Outer_Fold"] == outer_fold)]
            folds = sorted(cur["Inner_Fold"].unique())
            if folds != list(range(1, N_INNER_SPLITS + 1)):
                raise ValueError(f"Invalid inner folds for rep {rep}, outer fold {outer_fold}: {folds}")

            for inner_fold in folds:
                one = cur[cur["Inner_Fold"] == inner_fold]
                tr = set(one.loc[one["Set"] == "train", "ID"])
                va = set(one.loc[one["Set"] == "validation", "ID"])
                if tr & va:
                    raise ValueError(f"Inner train/validation overlap: rep {rep}, outer {outer_fold}, inner {inner_fold}")
                if tr | va != outer_train:
                    raise ValueError(f"Inner IDs do not exactly match outer-train IDs: rep {rep}, outer {outer_fold}, inner {inner_fold}")

    print("Fixed inner split validation passed: 50 outer runs x 5 inner folds.")
    return inner


def reshape_for_cnn(X_2d):
    return X_2d.reshape((X_2d.shape[0], X_2d.shape[1], 1))


def build_cnn(activation, dropout_rate, l2_val, learning_rate,
              optimizer_name, layer_filters, kernel_size, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=layer_filters[0], kernel_size=kernel_size,
                     activation=activation, kernel_regularizer=l2(l2_val)))
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


def suggest_cnn_params(trial):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.2, 0.4, 0.5])
    l2_val = trial.suggest_categorical("l2_val", [0.0001, 0.001, 0.01])

    if activation == "relu":
        lr_min, lr_max = 1e-5, 5e-4
    else:
        lr_min, lr_max = 5e-5, 1e-3
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    layer_filters = trial.suggest_categorical(
        "layer_filters", [(32,), (64, 32), (128, 64, 32)])
    kernel_size = trial.suggest_categorical("kernel_size", [1, 3, 5])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    return (activation, dropout_rate, l2_val, learning_rate,
            optimizer_name, layer_filters, kernel_size, batch_size)


def create_cnn_model(trial, input_shape):
    (activation, dropout_rate, l2_val, learning_rate,
     optimizer_name, layer_filters, kernel_size, _) = suggest_cnn_params(trial)
    return build_cnn(activation, dropout_rate, l2_val, learning_rate,
                     optimizer_name, layer_filters, kernel_size, input_shape)


def create_cnn_model_from_params(params, input_shape):
    return build_cnn(
        params["activation"], params["dropout_rate"], params["l2_val"],
        params["learning_rate"], params["optimizer"], params["layer_filters"],
        params["kernel_size"], input_shape)


def make_inner_objective(X_outer_train, y_outer_train, outer_train_ids,
                         inner_split_df, rep, outer_fold, id_to_local_index):
    """Each Optuna trial is evaluated on the SAME pre-saved 5 inner folds."""
    cur_inner = inner_split_df[(inner_split_df["Repetition"] == rep) &
                               (inner_split_df["Outer_Fold"] == outer_fold)]

    fold_indices = []
    for inner_fold in range(1, N_INNER_SPLITS + 1):
        one = cur_inner[cur_inner["Inner_Fold"] == inner_fold]
        tr_ids = one.loc[one["Set"] == "train", "ID"].tolist()
        va_ids = one.loc[one["Set"] == "validation", "ID"].tolist()
        tr_idx = np.array([id_to_local_index[x] for x in tr_ids], dtype=int)
        va_idx = np.array([id_to_local_index[x] for x in va_ids], dtype=int)
        fold_indices.append((tr_idx, va_idx))

    def objective(trial):
        # Suggest once per trial. The SAME parameter set is evaluated on all 5 inner folds.
        activation, dropout_rate, l2_val, learning_rate, optimizer_name, \
            layer_filters, kernel_size, batch_size = suggest_cnn_params(trial)

        scores = []
        for inner_fold, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
            X_tr, X_va = X_outer_train[tr_idx], X_outer_train[va_idx]
            y_tr, y_va = y_outer_train[tr_idx], y_outer_train[va_idx]

            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr)
            X_tr_s = reshape_for_cnn(sx.transform(X_tr))
            X_va_s = reshape_for_cnn(sx.transform(X_va))
            y_tr_s = sy.transform(y_tr)
            y_va_s = sy.transform(y_va)

            set_seed(BASE_SEED + rep * 100000 + outer_fold * 1000 +
                     trial.number * 10 + inner_fold)

            model = build_cnn(
                activation, dropout_rate, l2_val, learning_rate,
                optimizer_name, layer_filters, kernel_size,
                input_shape=(X_outer_train.shape[1], 1))
            model.fit(X_tr_s, y_tr_s, epochs=INNER_EPOCHS,
                      batch_size=batch_size, verbose=0)
            pred = model.predict(X_va_s, verbose=0).flatten()

            if np.isnan(pred).any() or np.isinf(pred).any():
                tf.keras.backend.clear_session()
                gc.collect()
                return np.inf

            scores.append(mean_squared_error(y_va_s.flatten(), pred))
            del model, sx, sy, X_tr_s, X_va_s, y_tr_s, y_va_s, pred
            tf.keras.backend.clear_session()
            gc.collect()

        # ONE objective value for this trial = mean MSE across the 5 inner folds.
        return float(np.mean(scores))

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="CNN using fixed 10 x 5 outer splits and fixed 5-fold inner CV.")
    parser.add_argument("dataset_file", help="*_dataset_with_ID.csv")
    parser.add_argument("split_file", help="*_data_split.csv")
    parser.add_argument("--inner_split_file", default=None,
                        help="Shared fixed inner split CSV. Created once if absent.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    dataset_file = os.path.abspath(args.dataset_file)
    split_file = os.path.abspath(args.split_file)

    name = os.path.splitext(os.path.basename(dataset_file))[0]
    if name.endswith("_dataset_with_ID"):
        name = name[:-len("_dataset_with_ID")]

    output_dir = (os.path.abspath(args.output_dir) if args.output_dir else
                  os.path.join(os.path.dirname(dataset_file),
                               f"{name}_CNN_fixed_splits_out"))
    inner_split_file = (os.path.abspath(args.inner_split_file)
                        if args.inner_split_file else
                        os.path.join(os.path.dirname(split_file), "inner_data_split.csv"))

    predictions_dir = os.path.join(output_dir, "predictions")
    optuna_dir = os.path.join(output_dir, "optuna_trials")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)

    print("=" * 75)
    print("CNN - FIXED REPEATED NESTED CROSS-VALIDATION")
    print("=" * 75)
    print(f"Dataset: {dataset_file}")
    print(f"Outer splits: {split_file}")
    print(f"Inner splits: {inner_split_file}")
    print("Outer: 10 repetitions x 5 fixed folds = 50 runs")
    print(f"Inner: {N_INNER_SPLITS}-fold fixed CV")
    print(f"Optuna: {N_TRIALS} trials per outer fold")

    df, X_all, y_all, ids = load_dataset(dataset_file)
    outer_splits = load_fixed_outer_splits(split_file, ids)
    inner_splits = generate_or_load_fixed_inner_splits(inner_split_file, outer_splits, ids)

    id_to_index = {sid: i for i, sid in enumerate(ids)}
    all_metrics = []
    all_params = []
    all_predictions = []

    for rep in range(1, N_REPETITIONS + 1):
        for outer_fold in range(1, N_OUTER_SPLITS + 1):
            print(f"\n===== Repetition {rep}/{N_REPETITIONS} | Outer fold {outer_fold}/{N_OUTER_SPLITS} =====")

            cur = outer_splits[(outer_splits["Repetition"] == rep) &
                               (outer_splits["Fold"] == outer_fold)]
            train_ids = cur.loc[cur["Set"] == "train", "ID"].tolist()
            test_ids = cur.loc[cur["Set"] == "test", "ID"].tolist()
            train_idx = np.array([id_to_index[x] for x in train_ids], dtype=int)
            test_idx = np.array([id_to_index[x] for x in test_ids], dtype=int)

            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]
            id_to_local_index = {sid: i for i, sid in enumerate(train_ids)}

            rep_seed = BASE_SEED + rep
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=rep_seed))
            study.optimize(
                make_inner_objective(X_tr, y_tr, train_ids, inner_splits,
                                     rep, outer_fold, id_to_local_index),
                n_trials=N_TRIALS,
                show_progress_bar=False)
            best = study.best_trial.params

            trials_df = study.trials_dataframe(
                attrs=("number", "value", "state", "params"))
            trials_df.insert(0, "Repetition", rep)
            trials_df.insert(1, "Outer_Fold", outer_fold)
            trials_df.to_csv(
                os.path.join(optuna_dir, f"optuna_trials_rep{rep}_fold{outer_fold}.csv"),
                index=False)

            best_save = best.copy()
            best_save["layer_filters"] = str(best_save["layer_filters"])
            pd.DataFrame([{
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "Best_Trial": study.best_trial.number,
                "Best_Inner_MSE": study.best_value,
                **best_save
            }]).to_csv(
                os.path.join(optuna_dir, f"optuna_best_rep{rep}_fold{outer_fold}.csv"),
                index=False)

            # FINAL scaling: fit ONLY on complete outer-training set.
            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr)
            X_tr_s = reshape_for_cnn(sx.transform(X_tr))
            X_te_s = reshape_for_cnn(sx.transform(X_te))
            y_tr_s = sy.transform(y_tr)
            y_te_s = sy.transform(y_te)

            set_seed(BASE_SEED + rep * 100 + outer_fold)
            final_model = create_cnn_model_from_params(best, input_shape=(X_all.shape[1], 1))
            final_model.fit(X_tr_s, y_tr_s, epochs=FINAL_EPOCHS,
                            batch_size=best["batch_size"], verbose=0)

            # Outer test is used ONLY here for final evaluation.
            pred_s = final_model.predict(X_te_s, verbose=0).flatten()
            mse = mean_squared_error(y_te_s.flatten(), pred_s)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_te_s.flatten(), pred_s)
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
                **best_save
            })

            pred_orig = sy.inverse_transform(pred_s.reshape(-1, 1)).flatten()
            pred_df = pd.DataFrame({
                "Repetition": rep,
                "Outer_Fold": outer_fold,
                "ID": test_ids,
                "Row_Index": test_idx,
                "Actual": y_te.flatten(),
                "Predicted": pred_orig
            })
            pred_df.to_csv(
                os.path.join(predictions_dir, f"predictions_rep{rep}_fold{outer_fold}.csv"),
                index=False)
            all_predictions.append(pred_df)

            print(f"Best inner MSE: {study.best_value:.6f}")
            print(f"Test MSE={mse:.6f} | RMSE={rmse:.6f} | MAPE={mape:.6f} | Correlation={corr:.6f}")

            del final_model, sx, sy, X_tr_s, X_te_s, y_tr_s, y_te_s, pred_s
            tf.keras.backend.clear_session()
            gc.collect()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "all_metrics_50_runs.csv"), index=False)

    params_df = pd.DataFrame(all_params)
    params_df.to_csv(os.path.join(output_dir, "all_best_hyperparameters_50_runs.csv"), index=False)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        os.path.join(output_dir, "all_predictions_50_runs.csv"), index=False)

    summary = metrics_df[["MSE", "RMSE", "MAPE", "Correlation"]].agg(["mean", "std"]).T
    summary.columns = ["Mean", "Std"]
    summary.to_csv(os.path.join(output_dir, "summary_mean_std_across_50_runs.csv"))

    print("\n" + "=" * 75)
    print("CNN ANALYSIS COMPLETED")
    print("=" * 75)
    print(summary)
    print("\nSaved:")
    print("  all_metrics_50_runs.csv")
    print("  all_best_hyperparameters_50_runs.csv")
    print("  all_predictions_50_runs.csv")
    print("  summary_mean_std_across_50_runs.csv")
    print("  predictions/")
    print("  optuna_trials/")
    print(f"  shared inner splits: {inner_split_file}")


if __name__ == "__main__":
    main()
