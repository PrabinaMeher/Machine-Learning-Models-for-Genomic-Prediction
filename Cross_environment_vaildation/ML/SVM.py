import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr



# CONFIGURATION
N_REPETITIONS = 10
N_OUTER_SPLITS = 5

SVR_MAX_ITER = 10000

FIXED_PARAMS = {
    "kernel": "rbf",
    "C": 300,
    "epsilon": 0.01,
    "gamma": "scale",
    "tol": 0.001,
    "degree": 3,
    "coef0": 0.0,
}



# ARGUMENTS
input_files = sys.argv[1:]

if not input_files:
    print(
        "Usage: python SVR_Cross_Environment_fixed_splits.py "
        "ENV1.csv ENV2.csv ENV3.csv ENV4.csv"
    )
    sys.exit(1)

if len(input_files) < 2:
    print("FATAL: At least two environments are required.")
    sys.exit(1)

input_files = [
    os.path.abspath(f)
    for f in input_files
]

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
        print(f"FATAL: Dataset not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(
        file_path,
        header=None
    )

    if df.empty:
        print(f"FATAL: Dataset is empty: {file_path}")
        sys.exit(1)

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
    print(
        "FATAL: Datasets have DIFFERENT row counts:",
        dict(zip(dataset_names, row_counts))
    )
    print(
        "A shared fold split across environments REQUIRES "
        "identical row counts and identical sample order."
    )
    sys.exit(1)

N_SAMPLES = row_counts[0]

feature_counts = [
    X.shape[1]
    for X in X_list
]

if len(set(feature_counts)) > 1:
    print(
        "FATAL: Datasets have different numbers of features:",
        dict(zip(dataset_names, feature_counts))
    )
    sys.exit(1)



# FIND COMMON FIXED SPLIT FILE
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

    searched = []

    for directory in candidate_dirs:

        candidate = os.path.join(
            directory,
            filename
        )

        searched.append(candidate)

        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "\nRequired fixed split file was not found:\n"
        f"{filename}\n\n"
        "Searched:\n"
        + "\n".join(searched)
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



# BUILD OUTER INDEX SPLITS FROM PERMANENT IDS
id_to_index = {
    str(sample_id): i
    for i, sample_id in enumerate(ids)
}

REP_FOLD_SPLITS = {}

for repetition in range(
    1,
    N_REPETITIONS + 1
):

    REP_FOLD_SPLITS[repetition] = {}

    rep_df = outer_df[
        outer_df["Repetition"] == repetition
    ]

    if len(rep_df) == 0:
        raise ValueError(
            f"No rows found for repetition {repetition}."
        )

    for outer_fold in range(
        1,
        N_OUTER_SPLITS + 1
    ):

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

        if not train_ids or not test_ids:
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



# BUILD SVR
def build_svr(params):

    return SVR(
        kernel=params["kernel"],
        C=params["C"],
        epsilon=params["epsilon"],
        gamma=params["gamma"],
        degree=params["degree"],
        coef0=params["coef0"],
        tol=params["tol"],
        max_iter=SVR_MAX_ITER,
    )



# CROSS-ENVIRONMENT ANALYSIS
all_train_metrics = []
HYPERPARAMS_USED = {}


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

    params = FIXED_PARAMS.copy()

    HYPERPARAMS_USED[train_name] = params

    print("\n" + "=" * 80)
    print(
        f"TRAINING ENVIRONMENT: {train_name}"
    )
    print(
        f"Fixed SVR parameters: {params}"
    )
    print("=" * 80)

    all_metrics = []
    all_preds_rows = []

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

            
            # OUTER TRAINING DATA ONLY
            
            X_fold_train_raw = X_list[
                train_idx
            ][fold_train_idx]

            y_fold_train_raw = y_list[
                train_idx
            ][fold_train_idx].reshape(-1, 1)

            
            # FIT X AND y SCALERS ONLY ON OUTER-TRAINING DATA
            scaler_X = StandardScaler()

            X_fold_train = scaler_X.fit_transform(
                X_fold_train_raw
            )

            scaler_y = StandardScaler()

            y_fold_train = scaler_y.fit_transform(
                y_fold_train_raw
            ).ravel()

            
            # FIT SVR
            

            model = build_svr(params)

            model.fit(
                X_fold_train,
                y_fold_train
            )

            
            # EVALUATE ON EVERY ENVIRONMENT
            

            for test_idx_env, test_name in enumerate(
                dataset_names
            ):

                X_test_raw = X_list[
                    test_idx_env
                ][fold_test_idx]

                y_test_orig = y_list[
                    test_idx_env
                ][fold_test_idx].ravel()

                # IMPORTANT:
                # Use ONLY the training environment/fold scaler.
                X_test = scaler_X.transform(
                    X_test_raw
                )

                y_pred_scaled = model.predict(
                    X_test
                ).reshape(-1, 1)

                y_pred_orig = scaler_y.inverse_transform(
                    y_pred_scaled
                ).ravel()

                # Metrics on ORIGINAL phenotype scale.
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

    # Per-run prediction files
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

    


    for test_name in dataset_names:

        pair_metrics = metrics_df[
            metrics_df["Test_Dataset"] == test_name
        ]

        pair_metrics.to_csv(
            os.path.join(
                pairs_dir,
                f"{train_name}_with_{test_name}_metrics.csv"
            ),
            index=False
        )

        pair_preds = preds_df[
            preds_df["Test_Dataset"] == test_name
        ]

        pair_preds.to_csv(
            os.path.join(
                pairs_dir,
                f"{train_name}_with_{test_name}_predictions.csv"
            ),
            index=False
        )

    
    # Fixed hyperparameters record
    hp_row = {
        **params,
        "Train_Dataset": train_name
    }

    pd.DataFrame(
        [hp_row]
    ).to_csv(
        os.path.join(
            output_dir,
            "hyperparameters_used.csv"
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
        f"\n[{train_name}] Summary across all 50 runs:"
    )

    print(summary)

    all_train_metrics.append(
        metrics_df
    )



# MASTER COMBINED SUMMARY
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

for train_name, params in HYPERPARAMS_USED.items():

    all_hp_rows.append(
        {
            **params,
            "Train_Dataset": train_name
        }
    )

pd.DataFrame(
    all_hp_rows
).to_csv(
    os.path.join(
        datasets_dir,
        "ALL_BEST_HYPERPARAMETERS.csv"
    ),
    index=False
)



# FINAL MESSAGE
print(
    "\nMASTER PAIRWISE SUMMARY "
    "(all Train x Test combinations)"
)

print(combined_summary)

print(
    "\nSaved pairwise summary to: "
    + os.path.join(
        datasets_dir,
        "ALL_PAIRWISE_summary.csv"
    )
)

print(
    "Saved hyperparameters to: "
    + os.path.join(
        datasets_dir,
        "ALL_BEST_HYPERPARAMETERS.csv"
    )
)
