import numpy as np
import pandas as pd
import argparse

METRICS_DICT = {
    "overall_test_acc": "Overall",
    "FST_1_test_acc": "Type 1",
    "FST_2_test_acc": "Type 2",
    "FST_3_test_acc": "Type 3",
    "FST_4_test_acc": "Type 4",
    "FST_5_test_acc": "Type 5",
    "FST_6_test_acc": "Type 6",
}

HOLDOUT_SETS_LIST = [
    "expert_select",
    "random_holdout",
    "br",
    "dermaamin",
    "a12",
    "a34",
    "a56",
]

global holdout_sets_best_hparams

holdout_sets_best_hparams = {
    "expert_select": {},
    "random_holdout": {},
    "br": {},
    "dermaamin": {},
    "a12": {},
    "a34": {},
    "a56": {},
}


# Define a function to calculate the standard deviation according to Numpy's default
# behavior.
# The pandas `std()` function uses Bessel's correction, which divides by `n-1` instead
# of `n` to calculate the standard deviation. We want to use Numpy's default behavior,
# which divides by `n`, so we define a custom function to use in the aggregation.
# https://stackoverflow.com/a/50307104
def std(x):
    return np.std(x)


def convertSeconds(seconds: int) -> list[int]:
    """
    Convert seconds to days, hours, minutes, and seconds.
    Code modified from: https://stackoverflow.com/a/53488429.

    Args:
        seconds (int): The number of seconds to convert.

    Returns:
        list[int]: A list containing the number of days, hours, minutes, and seconds.
    """
    d = seconds // (24 * 60 * 60)
    h = (seconds % (24 * 60 * 60)) // (60 * 60)
    m = (seconds % (60 * 60)) // 60
    s = seconds % 60

    return [d, h, m, s]


def get_total_time(results_df: pd.DataFrame) -> None:
    """
    This function takes a dataframe of hyperparameters experiment results and prints
    the total time taken to run the experiments. We print the total time in a human-
    readable format.

    Args:
        results_df (pd.DataFrame): A dataframe containing the results of the
            hyperparameters experiment. The dataframe should have a column for the
            time taken to run each experiment (`Duration`), with one row for each
            experiment.

    Returns:
        None
    """

    # Calculate the total time (seconds) taken to run all the experiments.
    total_time = results_df["Duration"].sum()

    # Convert the total time to days, hours, minutes, and seconds.
    days, hours, minutes, seconds = convertSeconds(total_time)

    # Print the total time in days, hours, minutes, and seconds.
    print(f"Total time taken to run the experiments:")
    print(f"Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}")

    return


def get_best_experiment_random_holdout(results_df: pd.DataFrame) -> None:
    """
    This function takes a dataframe of hyperparameters experiment results and prints
    the hyperparameters that gave the best overall test accuracy on the
    "random_holdout" set, as well as the mean and standard deviation of all the metrics
    for the best hyperparameters on each holdout set, aggregated over the different
    random seeds. We print the results in a format compatible with LaTeX tables.

    Args:
        results_df (pd.DataFrame): A dataframe containing the results of the
            hyperparameters experiment. The dataframe should have columns for the
            hyperparameters varied in the experiment (`n_epochs`, `optimizer`, and
            `base_lr`), the holdout set used (`holdout_set`), and the different metrics
            measured in the experiment (e.g. `overall_test_acc`, `FST_1_test_acc`,
            `FST_2_test_acc`, etc.). The dataframe should have one row for each
            experiment, with each row containing the hyperparameters used, the holdout
            set, and the metrics measured for that experiment.

    Returns:
        None
    """

    # First, we want to find the hyperparameters that gave the best overall test
    # accuracy on the "random_holdout" set, so we filter the results dataframe to
    # only include the results from the "random_holdout" set.
    results_df_random_holdout = results_df[results_df.holdout_set == "random_holdout"]

    # Next, we group the results by the 3 hyperparameters we varied in the experiment
    # (`n_epochs`, `optimizer`, and `base_lr`), and calculate the mean overall test
    # accuracy for each group, aggregated over the different random seeds.
    grouped_df = (
        results_df_random_holdout.groupby(["n_epochs", "optimizer", "base_lr"])
        .agg({"overall_test_acc": "mean"})
        .reset_index()
    )

    # We then find the hyperparameters that gave the highest overall test accuracy.
    best_hparams = grouped_df.loc[grouped_df["overall_test_acc"].idxmax()]
    print("Hyperparameters for best experiment:")
    print("Number of epochs:", best_hparams.n_epochs)
    print("Optimizer:", best_hparams.optimizer)
    print("Learning rate:", best_hparams.base_lr)

    # Finally, we filter the results dataframe to only include the results from the
    # experiments that used the best hyperparameters, and calculate the mean and
    # standard deviation of the test accuracies for each holdout set.
    results_df_best = results_df[
        (results_df.n_epochs == best_hparams.n_epochs)
        & (results_df.optimizer == best_hparams.optimizer)
        & (results_df.base_lr == best_hparams.base_lr)
    ]

    # Aggregate the results over the different random seeds for each holdout set.
    agg_results_df_best = (
        results_df_best.groupby("holdout_set")
        .agg({metric: ["mean", std] for metric in METRICS_DICT})
        .reset_index()
    )

    # Print the results in a format compatible with LaTeX tables.
    # Since our paper has metrics along rows and holdout sets along columns, we print
    # the results in the same format.
    print("\nResults for best experiment:")
    for metric in METRICS_DICT:
        metrics_str = f"{METRICS_DICT[metric]} "
        for holdout_set in HOLDOUT_SETS_LIST:
            # Get the mean and standard deviation of the metric for the holdout set.
            # We use the `.values[0]` to extract the scalar value from the pandas Series.
            mean_metric = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set, (metric, "mean")
            ].values[0]
            std_metric = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set, (metric, "std")
            ].values[0]
            # metrics_str += fr"& \rev{{{mean_metric*100:.2f}\% ± {std_metric*100:.2f}\%}} "
            metrics_str += f"& {mean_metric*100:.2f}\% ± {std_metric*100:.2f}\% "
        metrics_str += "\\\\"
        print(metrics_str)

    return


def get_best_experiment_each_holdout(results_df: pd.DataFrame) -> None:
    """
    Calculate and print the best hyperparameters and test accuracies for each holdout set.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the experiments.

    Returns:
        None
    """

    # Create a table to store the results.
    results_table = np.empty(
        (len(METRICS_DICT) + 2, len(HOLDOUT_SETS_LIST) + 1), dtype=object
    )

    # Fill in the table with the metric names and holdout set names.
    results_table[0, 0] = "Metric"
    results_table[-1, 0] = "BestHparams"
    for idx, metric in enumerate(METRICS_DICT):
        results_table[idx + 1, 0] = METRICS_DICT[metric]

    # Iterate over each holdout set.
    for idx, holdout_set in enumerate(HOLDOUT_SETS_LIST):
        results_table[0, idx + 1] = holdout_set

        # First, we want to find the hyperparameters that gave the best overall test
        # accuracy on the specified holdout set, so we filter the results dataframe to
        # only include the results from the specified holdout set.
        results_df_holdout_set = results_df[results_df.holdout_set == holdout_set]

        # Next, we group the results by the 3 hyperparameters we varied in the
        # experiment (`n_epochs`, `optimizer`, and `base_lr`), and calculate the mean
        # overall test accuracy for each group, aggregated over the different random
        # seeds.
        grouped_df = (
            results_df_holdout_set.groupby(["n_epochs", "optimizer", "base_lr"])
            .agg({"overall_test_acc": "mean"})
            .reset_index()
        )

        # We then store the hyperparameters that gave the highest overall test
        # accuracy for the specified holdout set.
        best_hparams = grouped_df.loc[grouped_df["overall_test_acc"].idxmax()]
        holdout_sets_best_hparams[holdout_set] = (
            best_hparams.n_epochs,
            best_hparams.optimizer,
            best_hparams.base_lr,
        )

        # Finally, we filter the results dataframe to only include the results from the
        # experiments that used the best hyperparameters, and calculate the mean and
        # standard deviation of the test accuracies for each holdout set.
        results_df_holdout_set_best = results_df_holdout_set[
            (results_df_holdout_set.n_epochs == best_hparams.n_epochs)
            & (results_df_holdout_set.optimizer == best_hparams.optimizer)
            & (results_df_holdout_set.base_lr == best_hparams.base_lr)
        ]

        # Aggregate the results over the different random seeds for each holdout set.
        agg_results_df_best = (
            results_df_holdout_set_best.groupby("holdout_set")
            .agg({metric: ["mean", std] for metric in METRICS_DICT})
            .reset_index()
        )

        # Iterate over each metric and store the mean and standard deviation of the
        # metric for the specified holdout set in the results table.
        for metric_idx, metric in enumerate(METRICS_DICT):
            # Get the mean and standard deviation of the metric for the holdout set.
            # We use the `.values[0]` to extract the scalar value from the pandas Series.
            mean_metric = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set, (metric, "mean")
            ].values[0]
            std_metric = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set, (metric, "std")
            ].values[0]
            results_table[metric_idx + 1, idx + 1] = (
                f"{mean_metric*100:.2f}\% ± {std_metric*100:.2f}\%"
            )

        # Store the best hyperparameters for the specified holdout set in the results
        # table.
        results_table[-1, idx + 1] = str(holdout_sets_best_hparams[holdout_set])

    # Print the results table in a format compatible with LaTeX tables.
    for row in results_table:
        # Replace any NaN values with "--" and print the row.
        # We need to first convert np.array to list of strings to perform the
        # replacement.
        row = np.array(
            [
                x.replace("nan\\% ± nan\\%", "--") if isinstance(x, str) else x
                for x in row
            ]
        )
        # Convert the strings to add the `\rev{}` command for paper revision.
        row = [rf"\rev{{{x}}}" if isinstance(x, str) else x for x in row]
        # Convert the row to a list of strings and join them with "&" to create a row
        # in the LaTeX table. Add "\\" at the end of the row to indicate a new row.
        print(" & ".join(row) + " \\\\")

    return


def get_best_experiments_all_hparams_all_holdouts(results_df: pd.DataFrame) -> None:
    results_table = np.empty(
        (len(HOLDOUT_SETS_LIST) + 1, len(HOLDOUT_SETS_LIST) + 1), dtype=object
    )

    results_table[0, 0] = "Holdout Set"
    for idx, holdout_set in enumerate(HOLDOUT_SETS_LIST):
        results_table[0, idx + 1] = holdout_set
        results_table[idx + 1, 0] = holdout_set

    for idx, holdout_set in enumerate(HOLDOUT_SETS_LIST):
        best_hparams = holdout_sets_best_hparams[holdout_set]

        for idx2, holdout_set2 in enumerate(HOLDOUT_SETS_LIST):
            results_df_holdout_set = results_df[results_df.holdout_set == holdout_set2]

            results_df_holdout_set_best = results_df_holdout_set[
                (results_df_holdout_set.n_epochs == best_hparams[0])
                & (results_df_holdout_set.optimizer == best_hparams[1])
                & (results_df_holdout_set.base_lr == best_hparams[2])
            ]

            agg_results_df_best = (
                results_df_holdout_set_best.groupby("holdout_set")
                .agg({"overall_test_acc": ["mean", std]})
                .reset_index()
            )

            overall_test_acc_mean = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set2,
                ("overall_test_acc", "mean"),
            ].values[0]
            overall_test_acc_std = agg_results_df_best.loc[
                agg_results_df_best.holdout_set == holdout_set2,
                ("overall_test_acc", "std"),
            ].values[0]
            results_table[idx2 + 1, idx + 1] = (
                f"{overall_test_acc_mean*100:.2f}\% ± {overall_test_acc_std*100:.2f}\%"
            )

    for row in results_table:
        row = np.array(
            [
                x.replace("nan\\% ± nan\\%", "--") if isinstance(x, str) else x
                for x in row
            ]
        )
        row = [rf"\rev{{{x}}}" if isinstance(x, str) else x for x in row]
        print(" & ".join(row) + " \\\\")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams_log_csv",
        type=str,
        required=True,
        help="Path to the hyperparameters experiment log CSV file",
    )
    args = parser.parse_args()

    results_df = pd.read_csv(args.hparams_log_csv)

    # Print the total time taken to run the experiments.
    get_total_time(results_df)
    print("\n")

    # Convert the learning rate to scientific notation.
    # The `replace()` function is used to format the learning rate without the leading
    # zero in the exponent. For example, 1e-03 is formatted as 1e-3.
    # Source: https://stackoverflow.com/a/14863239.
    results_df.base_lr = results_df.base_lr.apply(
        lambda x: f"{x:.0e}".replace("e-0", "e-")
    )

    ## Print the best hyperparameters and test accuracies for the "random_holdout" set.
    ## We don't need to print this since we are printing the best hyperparameters for 
    # each holdout set.
    # get_best_experiment_random_holdout(results_df)

    # Print the best hyperparameters and test accuracies for each holdout set.
    get_best_experiment_each_holdout(results_df)
    print("\n")

    # Print the best hyperparameters and test accuracies for each holdout set using the
    # best hyperparameters for each holdout set.
    get_best_experiments_all_hparams_all_holdouts(results_df)
