import os
import sys
import argparse
import pandas as pd


def load_datasets(input_files):
    dfs = []
    dataset_names = []

    for feature_file in input_files:
        if not os.path.isfile(feature_file):
            raise FileNotFoundError(f"Dataset not found: {feature_file}")

        df = pd.read_csv(feature_file, header=None)

        if df.empty:
            raise ValueError(f"Dataset is empty: {feature_file}")

        dfs.append(df)
        dataset_names.append(
            os.path.splitext(os.path.basename(feature_file))[0]
        )

    return dfs, dataset_names


def validate_datasets(dfs, dataset_names):
    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    row_counts = [len(df) for df in dfs]
    col_counts = [len(df.columns) for df in dfs]

    for name, rows, cols in zip(dataset_names, row_counts, col_counts):
        print(f"  {name}: {rows} rows x {cols} columns")

    if len(set(row_counts)) != 1:
        raise ValueError(
            "All environments must contain the same number of rows."
        )

    if len(set(col_counts)) != 1:
        raise ValueError(
            "All environments must contain the same number of columns."
        )

    print("\nValidation PASSED")
    print(f"Number of environments : {len(dfs)}")
    print(f"Number of genotypes    : {row_counts[0]}")
    print(f"Number of columns      : {col_counts[0]}")

    return row_counts[0]


def create_loeo_split(dataset_names, n_samples):
    rows = []

    for leave_env in dataset_names:
        for env in dataset_names:
            set_type = "test" if env == leave_env else "train"

            for i in range(n_samples):
                rows.append({
                    "LOEO_Iteration": leave_env,
                    "Environment": env,
                    "ID": f"ID{i + 1}",
                    "Set": set_type
                })

    return pd.DataFrame(
        rows,
        columns=[
            "LOEO_Iteration",
            "Environment",
            "ID",
            "Set"
        ]
    )


def create_summary(dataset_names, n_samples):
    rows = []

    for leave_env in dataset_names:
        train_envs = [
            env for env in dataset_names
            if env != leave_env
        ]

        rows.append({
            "LOEO_Iteration": leave_env,
            "Train_Environments": "+".join(train_envs),
            "Test_Environment": leave_env,
            "Train_Environment_Count": len(train_envs),
            "Test_Environment_Count": 1,
            "Train_Samples": len(train_envs) * n_samples,
            "Test_Samples": n_samples
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create LOEO data splits for multiple environments."
    )

    parser.add_argument(
        "feature_files",
        nargs="+",
        help="Two or more headerless environment CSV files."
    )

    parser.add_argument(
        "--output_root",
        default="LOEO_splits",
        help="Output directory."
    )

    args = parser.parse_args()

    if len(args.feature_files) < 2:
        raise ValueError(
            "At least TWO environments are required for LOEO."
        )

    input_files = [
        os.path.abspath(f)
        for f in args.feature_files
    ]

    dfs, dataset_names = load_datasets(input_files)
    n_samples = validate_datasets(dfs, dataset_names)

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    split_df = create_loeo_split(
        dataset_names,
        n_samples
    )

    summary_df = create_summary(
        dataset_names,
        n_samples
    )

    split_file = os.path.join(
        output_root,
        "LOEO_data_split.csv"
    )

    summary_file = os.path.join(
        output_root,
        "LOEO_split_summary.csv"
    )

    split_df.to_csv(
        split_file,
        index=False
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("LOEO SPLIT GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nEnvironments : {len(dataset_names)}")
    print(f"Genotypes    : {n_samples}")
    print(f"LOEO runs    : {len(dataset_names)}")

    print("\nLOEO design:")

    for leave_env in dataset_names:
        train_envs = [
            env for env in dataset_names
            if env != leave_env
        ]

        print(
            f"  Train: {'+'.join(train_envs)}"
            f"  ->  Test: {leave_env}"
        )

    print(f"\nSplit file:")
    print(f"  {split_file}")

    print(f"\nSummary file:")
    print(f"  {summary_file}")

    print("\nThe same LOEO_data_split.csv can be used")
    print("for ANN, MLP, SVM, CatBoost, etc.")


if __name__ == "__main__":
    main()
