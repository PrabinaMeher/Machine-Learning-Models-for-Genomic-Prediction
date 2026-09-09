```python
#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold



# CONFIGURATION


N_REPETITIONS = 10
N_OUTER_SPLITS = 5

BASE_SEED = 1000



# LOAD DATASETS


def load_datasets(input_files):

    dfs = []
    dataset_names = []

    for feature_file in input_files:

        if not os.path.isfile(feature_file):
            raise FileNotFoundError(
                f"\nDataset not found:\n{feature_file}"
            )

        # IMPORTANT:
        # Input CSV files have NO header
        df = pd.read_csv(feature_file, header=None)

        if df.empty:
            raise ValueError(
                f"\nDataset is empty:\n{feature_file}"
            )

        dfs.append(df)

        dataset_name = os.path.splitext(
            os.path.basename(feature_file)
        )[0]

        dataset_names.append(dataset_name)

    return dfs, dataset_names



# CHECK DATASET COMPATIBILITY


def validate_datasets(dfs, dataset_names):

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    row_counts = [len(df) for df in dfs]
    col_counts = [len(df.columns) for df in dfs]

    print("\nDatasets:")

    for name, rows, cols in zip(
        dataset_names,
        row_counts,
        col_counts
    ):
        print(
            f"  {name}: {rows} rows × {cols} columns"
        )

    
    # Same number of rows is REQUIRED
    

    if len(set(row_counts)) != 1:

        raise ValueError(
            "\nERROR: Datasets have different numbers of rows.\n"
            "For cross-environment analysis, all datasets must "
            "contain the same genotypes in the same row order.\n"
            f"Row counts: {dict(zip(dataset_names, row_counts))}"
        )

    
    # Same number of columns is not necessarily required
    
    #
    # Usually genotype/feature columns should also match.
    # We therefore check them and stop if they differ.
    

    if len(set(col_counts)) != 1:

        raise ValueError(
            "\nERROR: Datasets have different numbers of columns.\n"
            "The datasets should contain the same feature structure.\n"
            f"Column counts: {dict(zip(dataset_names, col_counts))}"
        )

    n_samples = row_counts[0]
    n_columns = col_counts[0]

    print("\nValidation PASSED")

    print(f"  Number of environments : {len(dfs)}")
    print(f"  Number of genotypes    : {n_samples}")
    print(f"  Number of columns      : {n_columns}")

    return n_samples



# ADD PERMANENT IDS


def add_ids(dfs):

    dfs_with_ids = []

    n_samples = len(dfs[0])

    permanent_ids = [
        f"ID{i}"
        for i in range(1, n_samples + 1)
    ]

    for df in dfs:

        df = df.copy()

        # Check that input doesn't already have an ID-like
        # first column that would cause confusion.
        #
        # Since original datasets have no headers,
        # we simply insert our permanent ID as column 0.

        df.insert(
            0,
            "ID",
            permanent_ids
        )

        dfs_with_ids.append(df)

    return dfs_with_ids



# GENERATE COMMON FIXED SPLITS


def generate_fixed_splits(n_samples):

    split_records = []

    for repetition in range(
        1,
        N_REPETITIONS + 1
    ):

       
        # Each repetition gets a different seed
       

        rep_seed = BASE_SEED + repetition

        outer_kf = KFold(
            n_splits=N_OUTER_SPLITS,
            shuffle=True,
            random_state=rep_seed
        )

        print(
            f"\nRepetition {repetition}/{N_REPETITIONS}"
            f" | Seed = {rep_seed}"
        )

       
        # Create the common genotype split
        #
        # IMPORTANT:
        # These indices are NOT environment-specific.
        #
        # Every environment will use exactly these IDs.
       

        for fold, (train_idx, test_idx) in enumerate(
            outer_kf.split(np.arange(n_samples)),
            start=1
        ):

            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            shuffle_seed = BASE_SEED + repetition * 100 + fold
            rng = np.random.default_rng(shuffle_seed)

            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
            # Safety check: no overlap
           

            train_set = set(train_idx)
            test_set = set(test_idx)

            if train_set.intersection(test_set):

                raise RuntimeError(
                    f"\nTrain/test overlap detected "
                    f"in Repetition {repetition}, Fold {fold}."
                )

           
            # Safety check: all samples accounted for
           

            if (
                len(train_idx) + len(test_idx)
                != n_samples
            ):

                raise RuntimeError(
                    f"\nTrain + test != total samples "
                    f"in Repetition {repetition}, Fold {fold}."
                )

           
            # TRAIN IDs
           

            for idx in train_idx:

                split_records.append({
                    "Repetition": repetition,
                    "Fold": fold,
                    "ID": f"ID{int(idx) + 1}",
                    "Set": "train"
                })

           
            # TEST IDs
           

            for idx in test_idx:

                split_records.append({
                    "Repetition": repetition,
                    "Fold": fold,
                    "ID": f"ID{int(idx) + 1}",
                    "Set": "test"
                })

           
            # Print information
           

            train_percent = (
                100 * len(train_idx) / n_samples
            )

            test_percent = (
                100 * len(test_idx) / n_samples
            )

            print(
                f"  Fold {fold}/{N_OUTER_SPLITS}"
                f" | Train = {len(train_idx)} "
                f"({train_percent:.2f}%)"
                f" | Test = {len(test_idx)} "
                f"({test_percent:.2f}%)"
            )

    return pd.DataFrame(
        split_records,
        columns=[
            "Repetition",
            "Fold",
            "ID",
            "Set"
        ]
    )



# VALIDATE FIXED SPLITS


def validate_splits(
    splits_df,
    n_samples
):

    print("\n")
    print("=" * 70)
    print("VALIDATING COMMON CROSS-ENVIRONMENT SPLITS")
    print("=" * 70)

    expected_combinations = (
        N_REPETITIONS * N_OUTER_SPLITS
    )

    actual_combinations = (
        splits_df[
            ["Repetition", "Fold"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if actual_combinations != expected_combinations:

        raise RuntimeError(
            f"\nExpected {expected_combinations} "
            f"outer splits, but found "
            f"{actual_combinations}."
        )

    # Expected IDs
    expected_ids = {
        f"ID{i}"
        for i in range(1, n_samples + 1)
    }

    
    # Check every repetition/fold
    

    for repetition in range(
        1,
        N_REPETITIONS + 1
    ):

        for fold in range(
            1,
            N_OUTER_SPLITS + 1
        ):

            current = splits_df[
                (splits_df["Repetition"] == repetition)
                &
                (splits_df["Fold"] == fold)
            ]

            train = current[
                current["Set"] == "train"
            ]

            test = current[
                current["Set"] == "test"
            ]

            train_ids = set(
                train["ID"]
            )

            test_ids = set(
                test["ID"]
            )

           
            # Correct sample count
           

            if (
                len(train_ids) + len(test_ids)
                != n_samples
            ):

                raise RuntimeError(
                    f"\nIncorrect sample count "
                    f"in Repetition {repetition}, "
                    f"Fold {fold}."
                )

           
            # No duplicate train IDs
           

            if train["ID"].duplicated().any():

                raise RuntimeError(
                    f"\nDuplicate train ID "
                    f"in Repetition {repetition}, "
                    f"Fold {fold}."
                )

           
            # No duplicate test IDs
           

            if test["ID"].duplicated().any():

                raise RuntimeError(
                    f"\nDuplicate test ID "
                    f"in Repetition {repetition}, "
                    f"Fold {fold}."
                )

           
            # No train/test overlap
           

            if train_ids.intersection(test_ids):

                raise RuntimeError(
                    f"\nTrain/test ID overlap "
                    f"in Repetition {repetition}, "
                    f"Fold {fold}."
                )

           
            # Every ID included
           

            if (
                train_ids.union(test_ids)
                != expected_ids
            ):

                raise RuntimeError(
                    f"\nNot all IDs represented "
                    f"in Repetition {repetition}, "
                    f"Fold {fold}."
                )

    
    # Check Set column
    

    if not set(
        splits_df["Set"]
    ).issubset({"train", "test"}):

        raise RuntimeError(
            "\nInvalid value found in Set column."
        )

    print("\nValidation PASSED")
    print("   No duplicate IDs")
    print("   No train/test overlap")
    print("   Every sample appears in every fold")
    print("  All dataset IDs are accounted for")
    print("  All 50 repetition/fold combinations are valid")



# CREATE SUMMARY


def create_summary(splits_df):

    summary_records = []

    for repetition in range(
        1,
        N_REPETITIONS + 1
    ):

        for fold in range(
            1,
            N_OUTER_SPLITS + 1
        ):

            current = splits_df[
                (splits_df["Repetition"] == repetition)
                &
                (splits_df["Fold"] == fold)
            ]

            train_n = (
                current["Set"] == "train"
            ).sum()

            test_n = (
                current["Set"] == "test"
            ).sum()

            total_n = train_n + test_n

            summary_records.append({

                "Repetition": repetition,

                "Fold": fold,

                "Train_N": train_n,

                "Test_N": test_n,

                "Total_N": total_n,

                "Train_Percent":
                    100 * train_n / total_n,

                "Test_Percent":
                    100 * test_n / total_n
            })

    return pd.DataFrame(
        summary_records
    )



# MAIN


def main():

    parser = argparse.ArgumentParser(

        description=(
            "Create one common fixed "
            "10 × 5 train/test split for "
            "multiple cross-environment datasets."
        )
    )

    parser.add_argument(
        "feature_files",
        nargs="+",
        help=(
            "Two or more headerless CSV datasets "
            "containing the same genotypes in "
            "the same row order."
        )
    )

    parser.add_argument(
        "--output_root",
        default="fixed_splits",
        help=(
            "Root directory for output files."
        )
    )

    args = parser.parse_args()

    
    # Input files
    

    input_files = [
        os.path.abspath(f)
        for f in args.feature_files
    ]

    
    # Require at least 2 environments
    

    if len(input_files) < 2:

        raise ValueError(
            "\nAt least TWO datasets are required "
            "for cross-environment analysis."
        )

    
    # Load
    

    dfs, dataset_names = load_datasets(
        input_files
    )

    
    # Validate
    

    n_samples = validate_datasets(
        dfs,
        dataset_names
    )

    
    # Add permanent IDs
    

    dfs_with_ids = add_ids(dfs)

    
    # Generate COMMON splits
    

    splits_df = generate_fixed_splits(
        n_samples
    )

    
    # Validate splits
    

    validate_splits(
        splits_df,
        n_samples
    )

    
    # Create summary
    

    summary_df = create_summary(
        splits_df
    )

    
    # Output directory
    

    output_root = os.path.abspath(
        args.output_root
    )

    os.makedirs(
        output_root,
        exist_ok=True
    )

    
    # Save each dataset with the SAME IDs
    

    for df, dataset_name in zip(
        dfs_with_ids,
        dataset_names
    ):

        output_file = os.path.join(
            output_root,
            f"{dataset_name}_dataset_with_ID.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nSaved dataset with ID:"
            f"\n  {output_file}"
        )

    
    # One COMMON split file
    

    split_output = os.path.join(
        output_root,
        "cross_environment_data_split.csv"
    )

    splits_df.to_csv(
        split_output,
        index=False
    )

    
    # Summary
    

    summary_output = os.path.join(
        output_root,
        "cross_environment_split_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False
    )

    
    # Final report
    

    print("\n")
    print("=" * 70)
    print("CROSS-ENVIRONMENT SPLIT GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"\nNumber of environments : "
        f"{len(dataset_names)}"
    )

    print(
        f"Number of genotypes    : "
        f"{n_samples}"
    )

    print(
        f"Repetitions            : "
        f"{N_REPETITIONS}"
    )

    print(
        f"Outer folds            : "
        f"{N_OUTER_SPLITS}"
    )

    print(
        f"Common genotype splits : "
        f"{N_REPETITIONS * N_OUTER_SPLITS}"
    )

    print(
        f"\nSplit file:"
        f"\n  {split_output}"
    )

    print(
        f"\nSummary file:"
        f"\n  {summary_output}"
    )

    print("\nDatasets:")
    for name in dataset_names:
        print(
            f"  {name}_dataset_with_ID.csv"
        )

    print("\n" + "=" * 70)
    print("READY FOR CROSS-ENVIRONMENT MODELING")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The same Repetition/Fold/ID assignment "
        "must be used for ALL environments."
    )

    print(
        "\nExample for 4 environments:"
    )

    print(
        "  1-1, 1-2, 1-3, 1-4"
    )

    print(
        "  2-1, 2-2, 2-3, 2-4"
    )

    print(
        "  3-1, 3-2, 3-3, 3-4"
    )

    print(
        "  4-1, 4-2, 4-3, 4-4"
    )

    print(
        "\nTotal combinations for "
        f"{len(dataset_names)} environments: "
        f"{len(dataset_names)} × {len(dataset_names)} "
        f"= {len(dataset_names) ** 2} "
        "per fold."
    )

    print(
        "\nTotal model evaluations across "
        "50 splits: "
        f"{50 * len(dataset_names) ** 2}"
    )



# RUN


if __name__ == "__main__":
    main()

