import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr



# CONFIGURATION
N_REPETITIONS = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
BASE_SEED = 1000


PARAM_GRID = {
    "model__n_estimators": [100, 200, 500],
    "model__learning_rate": [0.01, 0.1, 0.5],
}



# ARGUMENTS
parser = argparse.ArgumentParser(
    description=(
        "Run fixed 10x5 cross-environment CatBoost "
        "using GridSearchCV and common fixed splits."
    )
)

parser.add_argument(
    "input_files",
    nargs="+",
    help=(
        "Environment CSV files. They must contain identical samples "
        "in identical row order."
    ),
)

args = parser.parse_args()

input_files = [
    os.path.abspath(f)
    for f in args.input_files
]

if len(input_files) < 2:
    raise ValueError(
        "Cross-environment analysis requires at least two environment CSV files."
    )

# DATASET NAMES / PATHS
dataset_names = [
    os.path.splitext(os.path.basename(f))[0]
    for f in input_files
]

MODEL_NAME = os.path.splitext(
    os.path.basename(sys.argv[0])
)[0]

datasets_dir = os.path.dirname(
    os.path.abspath(input_files[0])
)



# LOAD ENVIRONMENT DATA
dfs = []

for file_path in input_files:

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        header=None
    )

    if df.empty:
        raise ValueError(
            f"Dataset is empty:\n{file_path}"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    dfs.append(df)


X_list = [
    df.iloc[:, :-1].values
    for df in dfs
]

y_list = [
    df.iloc[:, -1].values
    for df in dfs
]



# CHECK ENVIRONMENT ALIGNMENT
row_counts = [
    X.shape[0]
    for X in X_list
]

if len(set(row_counts)) > 1:
    raise ValueError(
        "\nFATAL: Environment datasets have different row counts:\n"
        f"{dict(zip(dataset_names, row_counts))}\n\n"
        "Cross-environment fixed row-position splits require "
        "identical sample counts."
    )

N_SAMPLES = row_counts[0]

feature_counts = [
    X.shape[1]
    for X in X_list
]

if len(set(feature_counts)) > 1:
    raise ValueError(
        "\nFATAL: Environment datasets have different numbers "
        f"of feature columns:\n"
        f"{dict(zip(dataset_names, feature_counts))}"
    )

if N_SAMPLES < N_OUTER_SPLITS:
    raise ValueError(
        f"Only {N_SAMPLES} samples are available, but "
        f"{N_OUTER_SPLITS} outer folds are required."
    )



# FIND COMMON FIXED SPLIT FILES
candidate_dirs = [
    os.path.join(datasets_dir, "fixed_splits"),
    datasets_dir,
    os.path.join(
        datasets_dir,
        "fixed_splits",
        dataset_names[0]
    ),
    os.path.join(
        os.path.dirname(datasets_dir),
        "fixed_splits"
    ),
    os.path.join(
        os.path.dirname(datasets_dir),
        "fixed_splits",
        dataset_names[0]
    ),
]


def find_file(filename):

    checked = []

    for directory in candidate_dirs:

        candidate = os.path.join(
            directory,
            filename
        )

        checked.append(candidate)

        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"\nRequired fixed split file was not found:\n"
        f"{filename}\n\n"
        "Searched:\n"
        + "\n".join(checked)
        + "\n\nRun generate_fixed_splits.py first."
    )


outer_split_file = find_file(
    "cross_environment_data_split.csv"
)

dataset_with_id_file = find_file(
    f"{dataset_names[0]}_dataset_with_ID.csv"
)



# LOAD PERMANENT IDS
id_df = pd.read_csv(
    dataset_with_id_file
)

if "ID" not in id_df.columns:
    raise ValueError(
        f"{dataset_with_id_file} must contain an ID column."
    )

if len(id_df) != N_SAMPLES:
    raise ValueError(
        "The dataset_with_ID.csv does not have the same "
        "number of rows as the environment datasets."
    )

ids = id_df["ID"].astype(str).values

if len(set(ids)) != len(ids):
    raise ValueError(
        "Permanent IDs are not unique."
    )



# LOAD COMMON OUTER SPLITS
outer_df = pd.read_csv(
    outer_split_file
)

required_outer = {
    "Repetition",
    "Fold",
    "ID",
    "Set",
}

missing = required_outer - set(
    outer_df.columns
)

if missing:
    raise ValueError(
        f"Outer split file is missing columns: {sorted(missing)}"
    )

outer_df["ID"] = outer_df["ID"].astype(str)

if set(outer_df["ID"]) != set(ids):
    raise ValueError(
        "IDs in the outer split file do not exactly match "
        "the permanent IDs."
    )



# BUILD COMMON OUTER INDEX SPLITS FROM IDS
id_to_index = {
    str(sample_id): i
    for i, sample_id in enumerate(ids)
}

REP_FOLD_SPLITS = {}

for repetition in range(1, N_REPETITIONS + 1):

    REP_FOLD_SPLITS[repetition] = {}

    rep_df = outer_df[
        outer_df["Repetition"] == repetition
    ]

    if len(rep_df) == 0:
        raise ValueError(
            f"No rows found for repetition {repetition}."
        )

    for outer_fold in range(1, N_OUTER_SPLITS + 1):

        fold_df = rep_df[
            rep_df["Fold"] == outer_fold
        ]

        train_ids = fold_df.loc[
            fold_df["Set"].astype(str).str.lower() == "train",
            "ID"
        ].tolist()

        test_ids = fold_df.loc[
            fold_df["Set"].astype(str).str.lower() == "test",
            "ID"
        ].tolist()

        if len(train_ids) == 0 or len(test_ids) == 0:
            raise ValueError(
                f"Missing train/test IDs for repetition "
                f"{repetition}, fold {outer_fold}."
            )

        train_idx = np.array(
            [id_to_index[x] for x in train_ids],
            dtype=int
        )

        test_idx = np.array(
            [id_to_index[x] for x in test_ids],
            dtype=int
        )

        if set(train_idx) & set(test_idx):
            raise ValueError(
                f"Train/test overlap in repetition "
                f"{repetition}, fold {outer_fold}."
            )

        REP_FOLD_SPLITS[repetition][outer_fold] = (
            train_idx,
            test_idx
        )



# BUILD DETERMINISTIC COMMON INNER FOLDS
INNER_SPLITS = {}

for repetition in range(
    1,
    N_REPETITIONS + 1
):

    INNER_SPLITS[repetition] = {}

    for outer_fold in range(
        1,
        N_OUTER_SPLITS + 1
    ):

        outer_train_idx, _ = (
            REP_FOLD_SPLITS[
                repetition
            ][outer_fold]
        )

        # Fixed ID/order for reproducibility.
        outer_train_idx_sorted = np.sort(
            outer_train_idx
        )

        inner_seed = (
            BASE_SEED
            + 10000
            + repetition * 100
            + outer_fold
        )

        inner_kf = KFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=inner_seed
        )

        INNER_SPLITS[repetition][outer_fold] = list(
            inner_kf.split(
                np.arange(
                    len(outer_train_idx_sorted)
                )
            )
        )



# CROSS-ENVIRONMENT ANALYSIS
all_train_metrics = []


for train_idx, train_name in enumerate(
    dataset_names
):

    output_dir = os.path.join(
        datasets_dir,
        f"{train_name}_{MODEL_NAME}_out"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    preds_dir = os.path.join(
        output_dir,
        "predictions"
    )

    os.makedirs(
        preds_dir,
        exist_ok=True
    )

    pairs_dir = os.path.join(
        output_dir,
        "pairs"
    )

    os.makedirs(
        pairs_dir,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print(
        f"TRAINING ENVIRONMENT: {train_name}"
    )
    print("=" * 80)

    all_metrics = []
    all_preds_rows = []
    all_best_params_rows = []

    for repetition in range(
        1,
        N_REPETITIONS + 1
    ):

        for outer_fold in range(
            1,
            N_OUTER_SPLITS + 1
        ):

            print(
                f"\n===== [{train_name}] "
                f"Repetition {repetition}/{N_REPETITIONS} "
                f"| Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
            )

            fold_train_idx, fold_test_idx = (
                REP_FOLD_SPLITS[
                    repetition
                ][outer_fold]
            )

            # Keep the exact sorted order used to create inner folds.
            fold_train_idx_sorted = np.sort(
                fold_train_idx
            )

            X_outer_train = X_list[
                train_idx
            ][fold_train_idx_sorted]

            y_outer_train = y_list[
                train_idx
            ][fold_train_idx_sorted]

            inner_splits = INNER_SPLITS[
                repetition
            ][outer_fold]

            
      
            catboost_model = CatBoostRegressor(
                loss_function="RMSE",
                random_seed=(
                    BASE_SEED
                    + repetition
                ),
                verbose=False,
                allow_writing_files=False,
            )

            pipeline = Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        catboost_model
                    ),
                ]
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=PARAM_GRID,
                cv=inner_splits,
                scoring="neg_mean_squared_error",
                refit=True,
                verbose=0,
                n_jobs=-1,
                return_train_score=False,
            )

            grid_search.fit(
                X_outer_train,
                y_outer_train
            )

            best_model = grid_search.best_estimator_
            best_params = {
                key.replace("model__", ""): value
                for key, value
                in grid_search.best_params_.items()
            }

            best_inner_rmse = np.sqrt(
                -grid_search.best_score_
            )

            print(
                f">>> Best inner-CV RMSE: "
                f"{best_inner_rmse:.5f}"
            )

            print(
                f">>> Best parameters: "
                f"{best_params}"
            )

            all_best_params_rows.append(
                {
                    "Repetition": repetition,
                    "Outer_Fold": outer_fold,
                    "Train_Dataset": train_name,
                    "Best_Inner_RMSE": best_inner_rmse,
                    **best_params,
                }
            )



            for test_idx_env, test_name in enumerate(
                dataset_names
            ):

                X_test = X_list[
                    test_idx_env
                ][fold_test_idx]

                y_test_orig = y_list[
                    test_idx_env
                ][fold_test_idx]

                y_pred_orig = (
                    best_model.predict(
                        X_test
                    )
                )

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

                all_metrics.append(
                    {
                        "Repetition": repetition,
                        "Outer_Fold": outer_fold,
                        "Train_Dataset": train_name,
                        "Test_Dataset": test_name,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAPE": mape,
                        "Correlation": corr,
                    }
                )

                for row_i, actual_v, pred_v in zip(
                    fold_test_idx,
                    y_test_orig,
                    y_pred_orig
                ):

                    all_preds_rows.append(
                        {
                            "Repetition": repetition,
                            "Outer_Fold": outer_fold,
                            "Train_Dataset": train_name,
                            "Test_Dataset": test_name,
                            "ID": ids[row_i],
                            "Row_Index": row_i,
                            "Actual": actual_v,
                            "Predicted": pred_v,
                        }
                    )

            

    

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df.to_csv(
        os.path.join(
            output_dir,
            "all_metrics_50_runs.csv"
        ),
        index=False
    )

    preds_df = pd.DataFrame(
        all_preds_rows
    )

    preds_df.to_csv(
        os.path.join(
            output_dir,
            "all_predictions_50_runs.csv"
        ),
        index=False
    )

    best_params_df = pd.DataFrame(
        all_best_params_rows
    )

    best_params_df.to_csv(
        os.path.join(
            output_dir,
            "all_best_hyperparameters_50_runs.csv"
        ),
        index=False
    )

    
    # Per-run predictions
    

    for (
        rep,
        fold
    ), group in preds_df.groupby(
        ["Repetition", "Outer_Fold"]
    ):

        group.to_csv(
            os.path.join(
                preds_dir,
                f"predictions_rep{rep}_fold{fold}.csv"
            ),
            index=False
        )

    
    # Per train/test pair
    

    for test_name in dataset_names:

        pair_metrics = metrics_df[
            metrics_df["Test_Dataset"]
            == test_name
        ]

        pair_metrics.to_csv(
            os.path.join(
                pairs_dir,
                f"{train_name}_with_{test_name}_metrics.csv"
            ),
            index=False
        )

        pair_preds = preds_df[
            preds_df["Test_Dataset"]
            == test_name
        ]

        pair_preds.to_csv(
            os.path.join(
                pairs_dir,
                f"{train_name}_with_{test_name}_predictions.csv"
            ),
            index=False
        )

    
    # Summary by test environment
    

    summary = (
        metrics_df
        .groupby("Test_Dataset")[
            [
                "MSE",
                "RMSE",
                "MAPE",
                "Correlation",
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "summary_mean_std_across_50_runs.csv"
        )
    )

    print(
        f"\n[{train_name}] Summary across 50 outer runs:"
    )

    print(summary)

    all_train_metrics.append(
        metrics_df
    )



# MASTER SUMMARY


combined_metrics = pd.concat(
    all_train_metrics,
    ignore_index=True
)

combined_summary = (
    combined_metrics
    .groupby(
        [
            "Train_Dataset",
            "Test_Dataset",
        ]
    )[
        [
            "MSE",
            "RMSE",
            "MAPE",
            "Correlation",
        ]
    ]
    .agg(["mean", "std"])
    .reset_index()
)

combined_summary.to_csv(
    os.path.join(
        datasets_dir,
        "ALL_PAIRWISE_summary.csv"
    ),
    index=False
)



# MASTER HYPERPARAMETER RECORD


all_hp_rows = []

for train_name in dataset_names:

    output_dir = os.path.join(
        datasets_dir,
        f"{train_name}_{MODEL_NAME}_out"
    )

    hp_file = os.path.join(
        output_dir,
        "all_best_hyperparameters_50_runs.csv"
    )

    if os.path.isfile(hp_file):

        hp_df = pd.read_csv(
            hp_file
        )

        all_hp_rows.append(
            hp_df
        )

if all_hp_rows:

    master_hp = pd.concat(
        all_hp_rows,
        ignore_index=True
    )

    master_hp.to_csv(
        os.path.join(
            datasets_dir,
            "ALL_BEST_HYPERPARAMETERS.csv"
        ),
        index=False
    )


print("\n" + "=" * 80)
print("CROSS-ENVIRONMENT CATBOOST ANALYSIS COMPLETE")
print("=" * 80)
print(
    f"Outer runs: {N_REPETITIONS} x {N_OUTER_SPLITS} = "
    f"{N_REPETITIONS * N_OUTER_SPLITS}"
)
print(
    f"Training environments: {len(dataset_names)}"
)
print(
    f"Pairwise evaluations: "
    f"{len(dataset_names)} x {len(dataset_names)} x "
    f"{N_REPETITIONS * N_OUTER_SPLITS}"
)
print(
    f"Results directory: {datasets_dir}"
)
