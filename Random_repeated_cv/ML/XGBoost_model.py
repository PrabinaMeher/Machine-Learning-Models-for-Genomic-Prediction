import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

#  Configuration 
N_REPETITIONS = 10      # outer shuffling repeated 10 times
N_OUTER_SPLITS = 5      # 5 outer folds per repetition -> 10 x 5 = 50 total runs
INNER_CV = 5            # GridSearchCV's own internal CV (this IS the inner loop - no
                        # separate Optuna needed, since GridSearchCV already does
                        # proper cross-validation using ONLY the outer-train data)

PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

MODEL_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

feature_file = sys.argv[1]

# Directory where the dataset is located
dataset_dir = os.path.dirname(os.path.abspath(feature_file))

# Dataset name without extension
feature_basename = os.path.splitext(os.path.basename(feature_file))[0]

# Automatic output directory
output_dir = os.path.join(
    dataset_dir,
    f"{feature_basename}_{MODEL_NAME}_out"
)

os.makedirs(output_dir, exist_ok=True)

preds_dir = os.path.join(output_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)
# ---- Load data ----
# NOTE: header=None kept as in the original script - confirm this matches your
# actual CSV format (no header row). If your CSVs DO have a header row (as the
# DL scripts assume), remove header=None below to avoid misaligning columns.
df = pd.read_csv(feature_file, header=None)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X_all = df.iloc[:, :-1].values
y_all = df.iloc[:, -1].values
N_FEATURES = X_all.shape[1]

all_metrics = []
all_best_params = []

for repetition in range(1, N_REPETITIONS + 1):
    rep_seed = 1000 + repetition
    outer_kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=rep_seed)

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X_all), start=1):
        print(f"\n===== Repetition {repetition}/{N_REPETITIONS} | Outer fold {outer_fold}/{N_OUTER_SPLITS} =====")

        X_outer_train, X_outer_test = X_all[outer_train_idx], X_all[outer_test_idx]
        y_outer_train, y_outer_test = y_all[outer_train_idx], y_all[outer_test_idx]

        # --- Scale using ONLY outer-train data (fit here, transform both) ---
        scaler = StandardScaler().fit(X_outer_train)
        X_outer_train_s = scaler.transform(X_outer_train)
        X_outer_test_s = scaler.transform(X_outer_test)

        # --- GridSearchCV IS the inner loop: its own cv=INNER_CV folds are built
        # entirely from X_outer_train_s, never touching the outer test fold ---
        xgb_model = XGBRegressor(random_state=rep_seed, verbosity=1)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=PARAM_GRID,
            cv=INNER_CV,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_outer_train_s, y_outer_train)

        # GridSearchCV(refit=True) by default already refits best_estimator_
        # on the FULL X_outer_train_s - no separate manual retrain step needed.
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # --- Evaluate ONCE on the truly held-out outer test fold ---
        y_test_pred = best_model.predict(X_outer_test_s)

        mse = mean_squared_error(y_outer_test, y_test_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_outer_test, y_test_pred)
        corr, _ = pearsonr(y_outer_test, y_test_pred)

        row = {
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": corr,
        }
        all_metrics.append(row)

        best_params_with_meta = best_params.copy()
        best_params_with_meta.update({"Repetition": repetition, "Outer_Fold": outer_fold})
        all_best_params.append(best_params_with_meta)

        # --- Save actual vs predicted values (original scale - target was never scaled) ---
        preds_df = pd.DataFrame({
            "Repetition": repetition,
            "Outer_Fold": outer_fold,
            "Row_Index": outer_test_idx,
            "Actual": y_outer_test,
            "Predicted": y_test_pred,
        })
        preds_df.to_csv(
            os.path.join(preds_dir, f"predictions_rep{repetition}_fold{outer_fold}.csv"),
            index=False
        )

#  Aggregate results across all 50 (repetition x outer_fold) runs 
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df[["Repetition", "Outer_Fold", "MSE", "RMSE", "MAPE", "Correlation"]]
metrics_df.to_csv(os.path.join(output_dir, "all_metrics_50_runs.csv"), index=False)

best_params_df = pd.DataFrame(all_best_params)
cols = ["Repetition", "Outer_Fold"] + [c for c in best_params_df.columns if c not in ("Repetition", "Outer_Fold")]
best_params_df = best_params_df[cols]
best_params_df.to_csv(os.path.join(output_dir, "all_best_hyperparameters_50_runs.csv"), index=False)

all_preds = pd.concat(
    [pd.read_csv(os.path.join(preds_dir, f))
     for f in sorted(os.listdir(preds_dir)) if f.endswith(".csv")],
    ignore_index=True
)
all_preds.to_csv(os.path.join(output_dir, "all_predictions_50_runs.csv"), index=False)

summary = metrics_df[["MSE", "RMSE", "MAPE", "Correlation"]].agg(["mean", "std"]).T
summary.columns = ["Mean", "Std"]
summary.to_csv(os.path.join(output_dir, "summary_mean_std_across_50_runs.csv"))

print("\n===== Summary across all 50 runs (10 repetitions x 5 outer folds) =====")
print(summary)
print(f"\nAll {N_REPETITIONS * N_OUTER_SPLITS} runs complete.")
print(f"- Per-run metrics:        all_metrics_50_runs.csv")
print(f"- Per-run best params:    all_best_hyperparameters_50_runs.csv")
print(f"- Per-run predictions:    predictions/predictions_rep{{r}}_fold{{f}}.csv")
print(f"- All predictions merged: all_predictions_50_runs.csv")
print(f"- Aggregate mean/std:     summary_mean_std_across_50_runs.csv")
