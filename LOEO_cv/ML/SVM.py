#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


N_INNER_SPLITS = 5
BASE_SEED = 1000


dataset_files = [
    os.path.abspath(x) for x in sys.argv[1:]
]

if len(dataset_files) < 2:
    raise ValueError(
        "Provide at least two environment CSV files."
    )


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
    os.path.splitext(
        os.path.basename(f)
    )[0]
    for f in dataset_files
]


raw_datasets = {}


for name, file in zip(
    env_names,
    dataset_files
):

    df = pd.read_csv(
        file,
        header=None
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if df.shape[1] < 2:
        raise ValueError(
            f"{file}: expected features + phenotype."
        )

    X = df.iloc[:, :-1].values

    y = df.iloc[:, -1].values.reshape(
        -1,
        1
    )

    raw_datasets[name] = (
        X,
        y
    )


n_samples = len(
    raw_datasets[env_names[0]][0]
)

n_features = raw_datasets[
    env_names[0]
][0].shape[1]


for name in env_names:

    if len(
        raw_datasets[name][0]
    ) != n_samples:

        raise ValueError(
            "All environments must contain "
            "the same number of genotypes."
        )

    if raw_datasets[name][0].shape[1] != n_features:

        raise ValueError(
            "All environments must contain "
            "the same number of features."
        )


loeo_splits = pd.read_csv(
    split_file
)


required_columns = {
    "LOEO_Iteration",
    "Environment",
    "ID",
    "Set"
}


if not required_columns.issubset(
    loeo_splits.columns
):

    raise ValueError(
        f"LOEO split file must contain: "
        f"{sorted(required_columns)}"
    )


param_grid = [

    {
        "kernel": ["linear"],
        "C": [300, 350, 400, 450, 500],
        "epsilon": [0.001, 0.01, 0.05, 0.1],
        "gamma": ["scale", "auto"]
    },

    {
        "kernel": ["rbf"],
        "C": [300, 350, 400, 450, 500],
        "epsilon": [0.001, 0.01, 0.05, 0.1],
        "gamma": [
            "scale",
            "auto",
            0.001,
            0.01,
            0.1
        ]
    },

    {
        "kernel": ["sigmoid"],
        "C": [300, 350, 400, 450, 500],
        "epsilon": [0.001, 0.01, 0.05, 0.1],
        "gamma": [
            "scale",
            "auto",
            0.001,
            0.01,
            0.1
        ]
    },

    {
        "kernel": ["poly"],
        "C": [300, 350, 400, 450, 500],
        "epsilon": [0.001, 0.01, 0.05, 0.1],
        "gamma": [
            "scale",
            "auto",
            0.001,
            0.01,
            0.1
        ]
    }

]


out_root = os.path.join(
    os.path.dirname(dataset_files[0]),
    "SVR_LOEO_GridSearch"
)

os.makedirs(
    out_root,
    exist_ok=True
)


all_metrics_rows = []
all_best_params = []
all_grid_results = []
all_predictions = []


for iteration, leave_name in enumerate(
    env_names,
    start=1
):

    other_names = [
        n for n in env_names
        if n != leave_name
    ]


    print(
        "\n" + "=" * 70
    )

    print(
        f"LOEO {iteration}/{len(env_names)}"
    )

    print(
        f"Train: {'+'.join(other_names)}"
    )

    print(
        f"Test : {leave_name}"
    )

    print(
        "=" * 70
    )


    env_dir = os.path.join(
        out_root,
        leave_name
    )

    os.makedirs(
        env_dir,
        exist_ok=True
    )


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


    inner_cv = KFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=BASE_SEED + iteration
    )


    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=True
    )


    print(
        "\nRunning GridSearchCV..."
    )


    grid_search.fit(
        X_train,
        y_train
    )


    best_params = grid_search.best_params_


    print(
        "\nBest parameters:"
    )

    print(
        best_params
    )

    print(
        f"Best CV MSE: "
        f"{-grid_search.best_score_}"
    )


    best_record = {
        "LOEO_Iteration": leave_name,
        "Train_Envs": "+".join(other_names),
        "Test_Env": leave_name,
        "Best_CV_MSE": -grid_search.best_score_
    }

    best_record.update(
        best_params
    )

    all_best_params.append(
        best_record
    )


    pd.DataFrame(
        [best_record]
    ).to_csv(
        os.path.join(
            env_dir,
            "best_hyperparameters.csv"
        ),
        index=False
    )


    grid_results = pd.DataFrame(
        grid_search.cv_results_
    )


    grid_results.insert(
        0,
        "LOEO_Iteration",
        leave_name
    )

    grid_results.insert(
        1,
        "Train_Envs",
        "+".join(other_names)
    )

    grid_results.insert(
        2,
        "Test_Env",
        leave_name
    )


    grid_results[
        "mean_test_MSE"
    ] = -grid_results[
        "mean_test_score"
    ]


    grid_results[
        "mean_train_MSE"
    ] = -grid_results[
        "mean_train_score"
    ]


    grid_results.to_csv(
        os.path.join(
            env_dir,
            "grid_search_results.csv"
        ),
        index=False
    )


    all_grid_results.append(
        grid_results
    )


    final_model = SVR(
        **best_params
    )


    final_model.fit(
        X_train,
        y_train
    )


    y_pred_scaled = final_model.predict(
        X_test
    )


    y_test_orig = y_test_raw.flatten()


    y_pred_orig = scaler_y.inverse_transform(
        y_pred_scaled.reshape(
            -1,
            1
        )
    ).flatten()


    mse = mean_squared_error(
        y_test_orig,
        y_pred_orig
    )

    rmse = np.sqrt(
        mse
    )

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


    metrics = pd.DataFrame([
        {
            "LOEO_Iteration": leave_name,
            "Train_Envs": "+".join(
                other_names
            ),
            "Test_Env": leave_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr
        }
    ])


    metrics.to_csv(
        os.path.join(
            env_dir,
            "Metrics.csv"
        ),
        index=False
    )


    all_metrics_rows.append(
        metrics
    )


    predictions = pd.DataFrame(
        {
            "LOEO_Iteration": leave_name,
            "Train_Envs": "+".join(
                other_names
            ),
            "Test_Env": leave_name,
            "Row_Index": np.arange(
                len(y_test_orig)
            ),
            "ID": [
                f"ID{i + 1}"
                for i in range(
                    len(y_test_orig)
                )
            ],
            "Actual": y_test_orig,
            "Predicted": y_pred_orig
        }
    )


    predictions.to_csv(
        os.path.join(
            env_dir,
            "ActualVsPredicted.csv"
        ),
        index=False
    )


    all_predictions.append(
        predictions
    )


    with open(
        os.path.join(
            env_dir,
            "log.txt"
        ),
        "w"
    ) as f:

        f.write(
            f"Train Environments: "
            f"{'+'.join(other_names)}\n"
        )

        f.write(
            f"Test Environment: "
            f"{leave_name}\n"
        )

        f.write(
            f"Inner CV folds: "
            f"{N_INNER_SPLITS}\n"
        )

        f.write(
            f"Best Hyperparameters:\n"
            f"{best_params}\n"
        )

        f.write(
            f"Best CV MSE: "
            f"{-grid_search.best_score_}\n"
        )

        f.write(
            f"Test MSE: "
            f"{mse}\n"
        )

        f.write(
            f"Test RMSE: "
            f"{rmse}\n"
        )

        f.write(
            f"Test MAPE: "
            f"{mape}\n"
        )

        f.write(
            f"Test Correlation: "
            f"{corr}\n"
        )


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


pd.DataFrame(
    all_best_params
).to_csv(
    os.path.join(
        out_root,
        "ALL_LOEO_best_hyperparameters.csv"
    ),
    index=False
)


if all_grid_results:

    pd.concat(
        all_grid_results,
        ignore_index=True
    ).to_csv(
        os.path.join(
            out_root,
            "all_grid_search_results.csv"
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


print(
    "\n" + "=" * 70
)

print(
    "LOEO SVR GRID SEARCH COMPLETE"
)

print(
    "=" * 70
)

print(
    combined_metrics
)
