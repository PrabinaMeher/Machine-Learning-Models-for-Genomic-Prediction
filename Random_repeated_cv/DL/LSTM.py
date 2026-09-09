import sys
import os
import random
import gc
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
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

# Fixed split files
split_file = os.path.join(dataset_dir, f"{feature_basename}_data_split.csv")
inner_split_file = os.path.join(
    dataset_dir, f"{feature_basename}_inner_data_split.csv"
)

print(f"Model: {MODEL_NAME} | Dataset: {feature_file}")
print(f"Outer split file: {split_file}")
print(f"Inner split file: {inner_split_file}")

# Load and clean data
df = pd.read_csv(feature_file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# First column = permanent ID, last column = phenotype/target
if df.shape[1] < 3:
    raise ValueError("Expected ID + at least one feature + target.")

ids = df.iloc[:, 0].values
X_all = df.iloc[:, 1:-1].values
y_all = df.iloc[:, -1].values.reshape(-1, 1)
N_FEATURES = X_all.shape[1]

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Fixed outer split file not found: {split_file}\n"
        "Run the common fixed-split generator first."
    )

outer_splits = pd.read_csv(split_file)
required_outer = {"Repetition", "Fold", "ID", "Set"}
if not required_outer.issubset(outer_splits.columns):
    raise ValueError(
        f"Outer split file must contain: {sorted(required_outer)}"
    )

if set(ids) != set(outer_splits["ID"]):
    raise ValueError(
        "Dataset IDs and IDs in data_split.csv do not match exactly."
    )


def create_shared_inner_splits():
    """
    Load the shared inner split file if it exists.
    Otherwise create it once using deterministic splits.
    All models should reuse this same file.
    """
    if os.path.exists(inner_split_file):
        inner_df = pd.read_csv(inner_split_file)
        required = {
            "Repetition", "Outer_Fold", "Inner_Fold", "ID", "Set"
        }
        if not required.issubset(inner_df.columns):
            raise ValueError(
                f"Inner split file must contain: {sorted(required)}"
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

            if not outer_train_ids:
                raise ValueError(
                    f"No outer-train IDs for repetition={repetition}, "
                    f"outer_fold={outer_fold}."
                )

            # Same deterministic inner split construction used by the
            # fixed-split workflow.
            rng = np.random.RandomState(
                (BASE_SEED + repetition) * 100 + outer_fold
            )

            shuffled_ids = np.array(outer_train_ids, dtype=object)
            rng.shuffle(shuffled_ids)

            fold_indices = np.array_split(
                np.arange(len(shuffled_ids)),
                N_INNER_SPLITS
            )

            for inner_fold, val_idx in enumerate(
                fold_indices, start=1
            ):
                val_ids = set(shuffled_ids[val_idx])

                for sample_id in shuffled_ids:
                    rows.append({
                        "Repetition": repetition,
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "ID": sample_id,
                        "Set": (
                            "validation"
                            if sample_id in val_ids
                            else "train"
                        )
                    })

    inner_df = pd.DataFrame(rows)
    inner_df.to_csv(inner_split_file, index=False)
    print(f"Created shared inner split file: {inner_split_file}")
    return inner_df


inner_splits = create_shared_inner_splits()


def validate_inner_splits(repetition, outer_fold, outer_train_ids):
    subset = inner_splits[
        (inner_splits["Repetition"] == repetition) &
        (inner_splits["Outer_Fold"] == outer_fold)
    ].copy()

    expected_ids = set(outer_train_ids)

    if set(subset["ID"]) != expected_ids:
        raise ValueError(
            f"Inner IDs do not exactly match outer-train IDs for "
            f"repetition={repetition}, outer_fold={outer_fold}."
        )

    for inner_fold in range(1, N_INNER_SPLITS + 1):
        fold = subset[
            subset["Inner_Fold"] == inner_fold
        ]

        if len(fold) == 0:
            raise ValueError(
                f"Missing inner fold {inner_fold} for "
                f"repetition={repetition}, outer_fold={outer_fold}."
            )

        train_ids = set(
            fold.loc[fold["Set"] == "train", "ID"]
        )
        val_ids = set(
            fold.loc[fold["Set"] == "validation", "ID"]
        )

        if train_ids & val_ids:
            raise ValueError("Inner train/validation overlap detected.")

        if train_ids | val_ids != expected_ids:
            raise ValueError(
                "Inner fold does not cover all outer-training IDs."
            )


def reshape_for_lstm(X_2d):
    return X_2d.reshape(
        (X_2d.shape[0], X_2d.shape[1], 1)
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_lstm_model(trial, input_shape):
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
        "layer_sizes",
        [(32,), (64, 32), (128, 64, 32)]
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer", ["adam", "sgd"]
    )

    return build_lstm(
        activation,
        dropout_rate,
        l2_val,
        learning_rate,
        layer_sizes,
        optimizer_name,
        input_shape
    )


def build_lstm(
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

    model.add(
        LSTM(
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
            LSTM(
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
            LSTM(
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
            learning_rate=learning_rate,
            clipnorm=1.0
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            clipnorm=1.0
        )

    model.compile(optimizer=optimizer, loss="mse")
    return model


def create_lstm_model_from_params(params, input_shape):
    return build_lstm(
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
    repetition,
    outer_fold,
    outer_train_ids
):
    """
    Uses the SAME fixed inner folds for every Optuna trial.
    The outer-test fold is never referenced.
    """
    validate_inner_splits(
        repetition,
        outer_fold,
        outer_train_ids
    )

    subset = inner_splits[
        (inner_splits["Repetition"] == repetition) &
        (inner_splits["Outer_Fold"] == outer_fold)
    ].copy()

    id_to_pos = {
        sample_id: i
        for i, sample_id in enumerate(outer_train_ids)
    }

    fixed_inner_folds = []

    for inner_fold in range(1, N_INNER_SPLITS + 1):
        fold = subset[
            subset["Inner_Fold"] == inner_fold
        ]

        train_ids = fold.loc[
            fold["Set"] == "train", "ID"
        ].tolist()

        val_ids = fold.loc[
            fold["Set"] == "validation", "ID"
        ].tolist()

        inner_train_idx = np.array(
            [id_to_pos[x] for x in train_ids],
            dtype=int
        )
        inner_val_idx = np.array(
            [id_to_pos[x] for x in val_ids],
            dtype=int
        )

        fixed_inner_folds.append(
            (inner_train_idx, inner_val_idx)
        )

    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )

        fold_scores = []

        for inner_fold_number, (
            inner_train_idx,
            inner_val_idx
        ) in enumerate(fixed_inner_folds, start=1):

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

            # Genotypic standardization:
            # fit ONLY on inner-training data.
            sx = StandardScaler().fit(X_inner_train)

            # Phenotypic standardization:
            # fit ONLY on inner-training phenotype.
            sy = StandardScaler().fit(y_inner_train)

            X_inner_train_s = reshape_for_lstm(
                sx.transform(X_inner_train)
            )
            X_inner_val_s = reshape_for_lstm(
                sx.transform(X_inner_val)
            )

            y_inner_train_s = sy.transform(y_inner_train)
            y_inner_val_s = sy.transform(y_inner_val)

            model = create_lstm_model(
                trial,
                input_shape=(N_FEATURES, 1)
            )

            model.fit(
                X_inner_train_s,
                y_inner_train_s,
                epochs=100,
                batch_size=batch_size,
                verbose=0
            )

            y_pred = model.predict(
                X_inner_val_s,
                verbose=0
            ).flatten()

            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                return np.inf

            fold_scores.append(
                mean_squared_error(
                    y_inner_val_s,
                    y_pred
                )
            )

            del model
            del sx, sy
            tf.keras.backend.clear_session()
            gc.collect()

        return float(np.mean(fold_scores))

    return objective


# Repeated nested CV using fixed outer and inner splits
all_metrics = []
all_best_params = []
all_trial_records = []
all_best_trial_records = []

id_to_global_pos = {
    sample_id: i
    for i, sample_id in enumerate(ids)
}

for repetition in range(1, N_REPETITIONS + 1):

    for outer_fold in range(1, N_OUTER_SPLITS + 1):

        print(
            f"\n===== Repetition {repetition}/{N_REPETITIONS} | "
            f"Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
        )

        final_model = None

        try:
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
                raise ValueError(
                    "Outer train/test overlap detected."
                )

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

            # Hyperparameter search only inside outer-training data.
            objective = make_inner_objective(
                X_outer_train,
                y_outer_train,
                repetition=repetition,
                outer_fold=outer_fold,
                outer_train_ids=outer_train_ids
            )

            study_seed = BASE_SEED + repetition * 100 + outer_fold
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

            if len(study.trials) == 0 or study.best_trial is None:
                print(
                    f"No successful Optuna trial in repetition "
                    f"{repetition}, fold {outer_fold}"
                )
                continue

            best_params = study.best_trial.params

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

            # Final standardization:
            # fit X and y scalers ONLY on full outer-training data.
            scaler_X = StandardScaler().fit(X_outer_train)
            scaler_y = StandardScaler().fit(y_outer_train)

            X_outer_train_s = reshape_for_lstm(
                scaler_X.transform(X_outer_train)
            )
            X_outer_test_s = reshape_for_lstm(
                scaler_X.transform(X_outer_test)
            )

            y_outer_train_s = scaler_y.transform(y_outer_train)
            y_outer_test_s = scaler_y.transform(y_outer_test)

            # Final model trained on ALL outer-training samples.
            set_seed(
                BASE_SEED
                + repetition * 10000
                + outer_fold * 100
                + 9999
            )

            final_model = create_lstm_model_from_params(
                best_params,
                input_shape=(N_FEATURES, 1)
            )

            final_model.fit(
                X_outer_train_s,
                y_outer_train_s,
                epochs=100,
                batch_size=best_params["batch_size"],
                verbose=1
            )

            # Outer test used ONLY for final evaluation.
            y_test_pred_s = final_model.predict(
                X_outer_test_s,
                verbose=0
            ).flatten()

            if np.isnan(y_test_pred_s).any() or np.isinf(y_test_pred_s).any():
                print(
                    f"NaN predictions in repetition {repetition}, "
                    f"fold {outer_fold}. Skipping fold."
                )
                continue

            mse = mean_squared_error(
                y_outer_test_s,
                y_test_pred_s
            )
            rmse = np.sqrt(mse)

            mape = mean_absolute_percentage_error(
                y_outer_test_s,
                y_test_pred_s
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
            best_params_with_meta["Repetition"] = repetition
            best_params_with_meta["Outer_Fold"] = outer_fold
            all_best_params.append(best_params_with_meta)

            # Convert predictions back to original phenotype scale.
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

        except Exception:
            print("\n" + "=" * 70)
            print(
                f"ERROR in repetition {repetition}, "
                f"fold {outer_fold}"
            )
            traceback.print_exc()
            print("=" * 70)

        finally:
            try:
                del final_model
            except Exception:
                pass

            tf.keras.backend.clear_session()
            gc.collect()


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

if not best_params_df.empty:
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

pd.DataFrame(all_trial_records).to_csv(
    os.path.join(
        output_dir,
        "all_optuna_trials_50_runs.csv"
    ),
    index=False
)

pd.DataFrame(all_best_trial_records).to_csv(
    os.path.join(
        output_dir,
        "best_optuna_trials_50_runs.csv"
    ),
    index=False
)

prediction_files = sorted(
    f for f in os.listdir(preds_dir)
    if f.endswith(".csv")
)

if prediction_files:
    all_preds = pd.concat(
        [
            pd.read_csv(
                os.path.join(preds_dir, f)
            )
            for f in prediction_files
        ],
        ignore_index=True
    )

    all_preds.to_csv(
        os.path.join(
            output_dir,
            "all_predictions_50_runs.csv"
        ),
        index=False
    )
else:
    print("No prediction files generated.")

if not metrics_df.empty:
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

print(
    f"\nCompleted {len(metrics_df)} successful outer-fold evaluations "
    f"out of {N_REPETITIONS * N_OUTER_SPLITS}."
)
