import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr



# ELM Regressor

class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_hidden_neurons=100,
        activation='sigmoid',
        alpha=1e-2,
        random_state=None
    ):
        self.n_hidden_neurons = n_hidden_neurons
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state

    def _activate(self, Z):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sine':
            return np.sin(Z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        rng = check_random_state(self.random_state)
        n_features = X.shape[1]

        # Random hidden-layer weights/biases.
        # These are generated once for each model fit and then frozen.
        self.input_weights_ = rng.normal(
            scale=1.0,
            size=(n_features, self.n_hidden_neurons)
        )
        self.biases_ = rng.normal(
            scale=1.0,
            size=(self.n_hidden_neurons,)
        )

        H = self._activate(
            X @ self.input_weights_ + self.biases_
        )

        # Closed-form ridge-regularized output weights
        I = np.eye(self.n_hidden_neurons)

        self.output_weights_ = np.linalg.solve(
            H.T @ H + self.alpha * I,
            H.T @ y
        )

        return self

    def predict(self, X):
        X = np.asarray(X)

        H = self._activate(
            X @ self.input_weights_ + self.biases_
        )

        return H @ self.output_weights_



# Configuration

N_REPETITIONS = 10
N_OUTER_SPLITS = 5
INNER_CV = 5

PARAM_GRID = {
    'n_hidden_neurons': [50, 100, 500, 1000],
    'activation': ['sigmoid', 'tanh', 'relu'],
    'alpha': [1e-3, 1e-2, 1e-1, 1.0]
}

MODEL_NAME = os.path.splitext(
    os.path.basename(sys.argv[0])
)[0]

feature_file = sys.argv[1]

dataset_dir = os.path.dirname(
    os.path.abspath(feature_file)
)

feature_basename = os.path.splitext(
    os.path.basename(feature_file)
)[0]

# Fixed split files
split_file = os.path.join(
    dataset_dir,
    f"{feature_basename}_data_split.csv"
)

inner_split_file = os.path.join(
    dataset_dir,
    f"{feature_basename}_inner_data_split.csv"
)

output_dir = os.path.join(
    dataset_dir,
    f"{feature_basename}_{MODEL_NAME}_out"
)

os.makedirs(output_dir, exist_ok=True)

preds_dir = os.path.join(
    output_dir,
    "predictions"
)

os.makedirs(preds_dir, exist_ok=True)

print(f"Model: {MODEL_NAME}  |  Dataset: {feature_file}")
print(f"Outer split file: {split_file}")
print(f"Inner split file: {inner_split_file}")
print(f"Outputs will be saved automatically under: {output_dir}")



# Load data

df = pd.read_csv(feature_file, header=None)

df = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna()
      .reset_index(drop=True)
)

# First column = permanent ID
# Last column = target
ids = df.iloc[:, 0].values
X_all = df.iloc[:, 1:-1].values
y_all = df.iloc[:, -1].values



# Load fixed OUTER split

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Required outer split file not found:\n{split_file}\n"
        "Run the fixed split generator first."
    )

outer_split_df = pd.read_csv(split_file)

required_outer_cols = {
    "Repetition", "Fold", "ID", "Set"
}

if not required_outer_cols.issubset(
    outer_split_df.columns
):
    raise ValueError(
        f"Outer split file must contain columns: "
        f"{required_outer_cols}"
    )



# Load fixed INNER split

if not os.path.exists(inner_split_file):
    raise FileNotFoundError(
        f"Required inner split file not found:\n{inner_split_file}\n"
        "Create the shared inner split file once before running "
        "the models."
    )

inner_split_df = pd.read_csv(inner_split_file)

required_inner_cols = {
    "Repetition",
    "Outer_Fold",
    "Inner_Fold",
    "ID",
    "Set"
}

if not required_inner_cols.issubset(
    inner_split_df.columns
):
    raise ValueError(
        f"Inner split file must contain columns: "
        f"{required_inner_cols}"
    )



# ID -> row index

id_to_idx = {
    sample_id: i
    for i, sample_id in enumerate(ids)
}

if len(id_to_idx) != len(ids):
    raise ValueError("Permanent IDs must be unique.")

missing_outer_ids = (
    set(outer_split_df["ID"]) - set(id_to_idx)
)

missing_inner_ids = (
    set(inner_split_df["ID"]) - set(id_to_idx)
)

if missing_outer_ids:
    raise ValueError(
        f"Outer split contains IDs not present in dataset: "
        f"{list(missing_outer_ids)[:10]}"
    )

if missing_inner_ids:
    raise ValueError(
        f"Inner split contains IDs not present in dataset: "
        f"{list(missing_inner_ids)[:10]}"
    )


all_metrics = []
all_best_params = []



# Outer repeated CV

for repetition in range(
    1,
    N_REPETITIONS + 1
):

    for outer_fold in range(
        1,
        N_OUTER_SPLITS + 1
    ):

        print(
            f"\n===== Repetition {repetition}/{N_REPETITIONS} | "
            f"Outer fold {outer_fold}/{N_OUTER_SPLITS} ====="
        )


        # Get fixed outer train/test IDs

        current_outer = outer_split_df[
            (outer_split_df["Repetition"] == repetition) &
            (outer_split_df["Fold"] == outer_fold)
        ]

        train_ids = current_outer.loc[
            current_outer["Set"] == "train",
            "ID"
        ].tolist()

        test_ids = current_outer.loc[
            current_outer["Set"] == "test",
            "ID"
        ].tolist()

        if len(train_ids) == 0 or len(test_ids) == 0:
            raise ValueError(
                f"Missing train/test IDs for "
                f"repetition={repetition}, "
                f"outer_fold={outer_fold}"
            )

        outer_train_idx = np.array(
            [id_to_idx[x] for x in train_ids],
            dtype=int
        )

        outer_test_idx = np.array(
            [id_to_idx[x] for x in test_ids],
            dtype=int
        )

        X_outer_train = X_all[outer_train_idx]
        X_outer_test = X_all[outer_test_idx]

        y_outer_train = y_all[outer_train_idx]
        y_outer_test = y_all[outer_test_idx]



        # Scale using ONLY outer-training data

        scaler = StandardScaler().fit(
            X_outer_train
        )

        X_outer_train_s = scaler.transform(
            X_outer_train
        )

        X_outer_test_s = scaler.transform(
            X_outer_test
        )



        # Fixed inner folds

        current_inner = inner_split_df[
            (inner_split_df["Repetition"] == repetition) &
            (inner_split_df["Outer_Fold"] == outer_fold)
        ]

        if current_inner.empty:
            raise ValueError(
                f"No inner split found for "
                f"repetition={repetition}, "
                f"outer_fold={outer_fold}"
            )

        # Global dataset index -> position in outer-training array
        outer_train_position = {
            idx: pos
            for pos, idx in enumerate(outer_train_idx)
        }

        cv_splits = []

        for inner_fold in sorted(
            current_inner["Inner_Fold"].unique()
        ):

            inner_train_ids = current_inner.loc[
                (current_inner["Inner_Fold"] == inner_fold) &
                (current_inner["Set"] == "train"),
                "ID"
            ].tolist()

            inner_val_ids = current_inner.loc[
                (current_inner["Inner_Fold"] == inner_fold) &
                (current_inner["Set"] == "validation"),
                "ID"
            ].tolist()

            inner_train_idx_global = np.array(
                [id_to_idx[x] for x in inner_train_ids],
                dtype=int
            )

            inner_val_idx_global = np.array(
                [id_to_idx[x] for x in inner_val_ids],
                dtype=int
            )

            # Inner training and validation must be inside
            # the corresponding outer-training set.
            if not set(
                inner_train_idx_global
            ).issubset(set(outer_train_idx)):
                raise ValueError(
                    f"Inner training data contains samples outside "
                    f"outer training set: repetition={repetition}, "
                    f"outer_fold={outer_fold}, "
                    f"inner_fold={inner_fold}"
                )

            if not set(
                inner_val_idx_global
            ).issubset(set(outer_train_idx)):
                raise ValueError(
                    f"Inner validation data contains samples outside "
                    f"outer training set: repetition={repetition}, "
                    f"outer_fold={outer_fold}, "
                    f"inner_fold={inner_fold}"
                )

            inner_train_idx = np.array(
                [
                    outer_train_position[x]
                    for x in inner_train_idx_global
                ],
                dtype=int
            )

            inner_val_idx = np.array(
                [
                    outer_train_position[x]
                    for x in inner_val_idx_global
                ],
                dtype=int
            )

            cv_splits.append(
                (inner_train_idx, inner_val_idx)
            )



        # GridSearchCV = inner hyperparameter search

        rep_seed = 1000 + repetition

        elm_model = ELMRegressor(
            random_state=rep_seed
        )

        grid_search = GridSearchCV(
            estimator=elm_model,
            param_grid=PARAM_GRID,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1,
            refit=True
        )

        grid_search.fit(
            X_outer_train_s,
            y_outer_train
        )

        # refit=True automatically refits the best model
        # on the FULL outer-training data.
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_



        # Evaluate ONCE on fixed outer-test data

        y_test_pred = best_model.predict(
            X_outer_test_s
        )

        mse = mean_squared_error(
            y_outer_test,
            y_test_pred
        )

        rmse = np.sqrt(mse)

        mape = mean_absolute_percentage_error(
            y_outer_test,
            y_test_pred
        )

        corr, _ = pearsonr(
            y_outer_test,
            y_test_pred
        )

        all_metrics.append({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
        })

        best_params_with_meta = best_params.copy()

        best_params_with_meta.update({
            "Repetition": repetition,
            "Outer_Fold": outer_fold
        })

        all_best_params.append(
            best_params_with_meta
        )



        # Save actual vs predicted values

        preds_df = pd.DataFrame({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "ID": ids[outer_test_idx],
            "Row_Index": outer_test_idx,
            "Actual": y_outer_test,
            "Predicted": y_test_pred,
        })

        preds_df.to_csv(
            os.path.join(
                preds_dir,
                f"predictions_rep{repetition}_fold{outer_fold}.csv"
            ),
            index=False
        )



# Aggregate results across all 50 runs

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_df = metrics_df[
    [
        "Repetition",
        "Outer_Fold",
        "MSE",
        "RMSE",
        "MAPE",
        "Correlation"
    ]
]

metrics_df.to_csv(
    os.path.join(
        output_dir,
        "all_metrics_50_runs.csv"
    ),
    index=False
)


best_params_df = pd.DataFrame(
    all_best_params
)

cols = (
    ["Repetition", "Outer_Fold"] +
    [
        c
        for c in best_params_df.columns
        if c not in (
            "Repetition",
            "Outer_Fold"
        )
    ]
)

best_params_df = best_params_df[cols]

best_params_df.to_csv(
    os.path.join(
        output_dir,
        "all_best_hyperparameters_50_runs.csv"
    ),
    index=False
)


all_preds = pd.concat(
    [
        pd.read_csv(
            os.path.join(preds_dir, f)
        )
        for f in sorted(
            os.listdir(preds_dir)
        )
        if f.endswith(".csv")
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


summary = metrics_df[
    [
        "MSE",
        "RMSE",
        "MAPE",
        "Correlation"
    ]
].agg(
    ["mean", "std"]
).T

summary.columns = [
    "Mean",
    "Std"
]

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
    f"\nAll {N_REPETITIONS * N_OUTER_SPLITS} runs complete."
)

print(
    "- Per-run metrics:        all_metrics_50_runs.csv"
)

print(
    "- Per-run best params:    "
    "all_best_hyperparameters_50_runs.csv"
)

print(
    "- Per-run predictions:    "
    "predictions/predictions_rep{r}_fold{f}.csv"
)

print(
    "- All predictions merged: "
    "all_predictions_50_runs.csv"
)

print(
    "- Aggregate mean/std:     "
    "summary_mean_std_across_50_runs.csv"
)
