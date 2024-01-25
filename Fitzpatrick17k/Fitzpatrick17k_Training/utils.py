import os
import pandas as pd
import torch
import skimage
from skimage import io

# This file contains utility functions for training on the Fitzpatrick17k dataset.
# `flatten()` was actually a part of the `train.py`` file [1], but was moved here for
# better organization.
# [1]: https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py#L22


def get_train_val_test_fst(holdout_set: str) -> tuple[list[int], list[int]]:
    """
    For a given holdout set, return the train, val, and test FSTs.
    train and val FSTs are those in the name of the holdout set.
    test FSTs are those not in the name of the holdout set.
    For example, if holdout_set = "a12", then train and val FSTs are 1 and 2,
    and test FSTs are 3, 4, 5, and 6.

    Args:
        holdout_set (str): Holdout set name. Must be one of "a12", "a34", or "a56".

    Returns:
        train_val_fsts (list): List of train and val FSTs.
        test_fst (list): List of test FSTs.
    """
    assert holdout_set in ["a12", "a34", "a56"]

    # Generate a list of all possible FSTs.
    fsts = [(i + 1) for i in range(6)]

    # Get the train and val FSTs from the name of the holdout set.
    train_val_fsts = [int(holdout_set[1]), int(holdout_set[2])]

    # Get the test FSTs by removing the train and val FSTs from the list of
    # all possible FSTs.
    test_fst = [i for i in fsts if i not in train_val_fsts]

    return train_val_fsts, test_fst


def keep_common_labels(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    holdout_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only those diagnosis labels that are common in all three partitions.

    Args:
        df (pd.DataFrame): The DataFrame containing all data.
        train (pd.DataFrame): The train partition.
        val (pd.DataFrame): The val partition.
        test (pd.DataFrame): The test partition.
        holdout_type (str): The type of holdout set. Must be either "fst" or "source".

    Returns:
        train (pd.DataFrame): The train partition.
        val (pd.DataFrame): The val partition.
        test (pd.DataFrame): The test partition.
    """
    if holdout_type == "fst":
        # Find the labels that are in all three partitions, and only keep those labels
        # in all three partitions.
        combo = (
            set(train.label.unique())
            & set(val.label.unique())
            & set(test.label.unique())
        )
    elif holdout_type == "source":
        combo = set(
            df[df.url_alphanum.str.contains("dermaamin") == True].label.unique()
        ) & set(df[df.url_alphanum.str.contains("dermaamin") == False].label.unique())

    train = train[train.label.isin(combo)].reset_index()
    val = val[val.label.isin(combo)].reset_index()
    test = test[test.label.isin(combo)].reset_index()

    return train, val, test


def get_partitions(
    all_data_list: str, holdout_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get the train, val, and test partitions for a given holdout set.


    Args:
        all_data_list (str): The path to the CSV file containing all data.
        holdout_set (str): The name of the holdout set. Must be one of
                           "random_holdout", "expert_select", "a12", "a34", "a56",
                           "dermaamin", or "br".

    Returns:
        partitions (tuple): A tuple of the train, val, and test partitions.
    """
    assert holdout_set in [
        "random_holdout",
        "expert_select",
        "a12",
        "a34",
        "a56",
        "dermaamin",
        "br",
    ]

    # Initialize an empty dictionary to store the partitions.
    partitions = {"train": None, "val": None, "test": None}

    # Read the CSV file containing all data into a pandas DataFrame.
    df = pd.read_csv(all_data_list, header="infer")

    # If the holdout set is "random_holdout", then use the partitions as is.
    if holdout_set == "random_holdout":
        for part in partitions:
            partitions[part] = df[df.partition == part].reset_index()

    # If the holdout set is "expert_select", then
    # - use the partitions as is for train and val,
    # - use the QC'd data for test,
    # - remove the QC'd data from train and val.
    elif holdout_set == "expert_select":
        partitions["train"] = df[df.partition.isin(["train", "test"])]
        partitions["val"] = df[df.partition == "val"]

        partitions["test"] = df[df.qc == "1 Diagnostic"].reset_index()
        partitions["train"] = partitions["train"][
            partitions["train"].qc != "1 Diagnostic"
        ].reset_index()
        partitions["val"] = partitions["val"][
            partitions["val"].qc != "1 Diagnostic"
        ].reset_index()

    # If the holdout set is "a12", "a34", or "a56", then
    # - for train and val, use the partitions as is for the FSTs in the name of the holdout set,
    # - for train, also use the data from the test partition for the FSTs in the name of the holdout set,
    # - for test, use the partitions as is for the FSTs not in the name of the holdout set,
    # - for test, also use the data from the train and val partitions for the FSTs not in the name of the holdout set,
    # Finally, remove any labels that are not in all three partitions.
    # For example, if the holdout set is "a34", then
    # train partition is the train partition for FSTs 3 and 4, AND the test partition for FSTs 3 and 4,
    # val partition is the val partition for FSTs 3 and 4,
    # test partition is the train, val, and test partitions for FSTs 1, 2, 5, and 6.
    elif holdout_set in ["a12", "a34", "a56"]:
        # Get train, val, and test FSTs from the name of the holdout set.
        train_val_fst, test_fst = get_train_val_test_fst(holdout_set)

        partitions["train"] = pd.concat(
            [
                df[(df.partition == "train") & (df.fitzpatrick.isin(train_val_fst))],
                df[(df.partition == "test") & (df.fitzpatrick.isin(train_val_fst))],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()
        partitions["val"] = df[
            (df.partition == "val") & (df.fitzpatrick.isin(train_val_fst))
        ].reset_index()
        partitions["test"] = pd.concat(
            [
                df[(df.partition == "train") & (df.fitzpatrick.isin(test_fst))],
                df[(df.partition == "val") & (df.fitzpatrick.isin(test_fst))],
                df[(df.partition == "test") & (df.fitzpatrick.isin(test_fst))],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()

        partitions["train"], partitions["val"], partitions["test"] = keep_common_labels(
            df=df,
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
            holdout_type="fst",
        )

    elif holdout_set == "dermaamin":
        partitions["train"] = pd.concat(
            [
                df[
                    (df.partition == "train")
                    & (df.url_alphanum.str.contains("dermaamin") == False)
                ],
                df[
                    (df.partition == "test")
                    & (df.url_alphanum.str.contains("dermaamin") == False)
                ],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()
        partitions["val"] = df[
            (df.partition == "val")
            & (df.url_alphanum.str.contains("dermaamin") == False)
        ].reset_index()
        partitions["test"] = pd.concat(
            [
                df[
                    (df.partition == "train")
                    & (df.url_alphanum.str.contains("dermaamin") == True)
                ],
                df[
                    (df.partition == "val")
                    & (df.url_alphanum.str.contains("dermaamin") == True)
                ],
                df[
                    (df.partition == "test")
                    & (df.url_alphanum.str.contains("dermaamin") == True)
                ],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()

        partitions["train"], partitions["val"], partitions["test"] = keep_common_labels(
            df=df,
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
            holdout_type="source",
        )

    elif holdout_set == "br":
        partitions["train"] = pd.concat(
            [
                df[
                    (df.partition == "train")
                    & (df.url_alphanum.str.contains("dermaamin") == True)
                ],
                df[
                    (df.partition == "test")
                    & (df.url_alphanum.str.contains("dermaamin") == True)
                ],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()
        partitions["val"] = df[
            (df.partition == "val")
            & (df.url_alphanum.str.contains("dermaamin") == True)
        ].reset_index()
        partitions["test"] = pd.concat(
            [
                df[
                    (df.partition == "train")
                    & (df.url_alphanum.str.contains("dermaamin") == False)
                ],
                df[
                    (df.partition == "val")
                    & (df.url_alphanum.str.contains("dermaamin") == False)
                ],
                df[
                    (df.partition == "test")
                    & (df.url_alphanum.str.contains("dermaamin") == False)
                ],
            ],
            axis=0,
            ignore_index=True,
        ).reset_index()

        partitions["train"], partitions["val"], partitions["test"] = keep_common_labels(
            df=df,
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
            holdout_type="source",
        )

    print(f"{holdout_set}: (train, val, test)")
    print(
        f"images: ({len(partitions['train'])}, "
        f"{len(partitions['val'])}, "
        f"{len(partitions['test'])})"
    )
    print(
        f"diagnoses: ({len(set(partitions['train'].label.to_list()))}, "
        f"{len(set(partitions['val'].label.to_list()))}, "
        f"{len(set(partitions['test'].label.to_list()))})"
    )

    # Verify that there is no overlap between the partitions.
    assert pd.merge(
        partitions["train"], partitions["val"], on="md5hash", how="inner"
    ).empty
    assert pd.merge(
        partitions["train"], partitions["test"], on="md5hash", how="inner"
    ).empty
    assert pd.merge(
        partitions["val"], partitions["test"], on="md5hash", how="inner"
    ).empty

    for part in partitions:
        partitions[part]["low"] = partitions[part]["label"].astype("category").cat.codes
        partitions[part]["mid"] = (
            partitions[part]["nine_partition_label"].astype("category").cat.codes
        )
        partitions[part]["high"] = (
            partitions[part]["three_partition_label"].astype("category").cat.codes
        )
        partitions[part]["hasher"] = partitions[part]["md5hash"]

    return partitions["train"], partitions["val"], partitions["test"]


def test_all_holdout_partitions(all_data_list: str) -> None:
    """
    Test all holdout partitions.

    Args:
        all_data_list (str): The path to the CSV file containing all data.
    """
    for holdout_set in [
        "random_holdout",
        "expert_select",
        "a12",
        "a34",
        "a56",
        "dermaamin",
        "br",
    ]:
        # for holdout_set in ["a56", "dermaamin", "br"]:
        _, _, _ = get_partitions(all_data_list, holdout_set)
        print("\n")


class SkinDataset:
    def __init__(self, df, root_dir, transform=None):
        """
        Taken from:
        https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py#L99
        with 3 minor modifications:
        1. Changed input argument "csv_file" to "df". This now takes a pd.DataFrame.
        2. Added ".jpg" to the end of the image name.
        3. Added "diag" to the sample. "diag" is the abbreviated diagnosis label.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.root_dir, f"{self.df.loc[self.df.index[idx], 'hasher']}.jpg"
        )
        image = io.imread(img_name)
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], "hasher"]
        high = self.df.loc[self.df.index[idx], "high"]
        mid = self.df.loc[self.df.index[idx], "mid"]
        low = self.df.loc[self.df.index[idx], "low"]
        diagcode = self.df.loc[self.df.index[idx], "diag"]
        fitzpatrick = self.df.loc[self.df.index[idx], "fitzpatrick"]
        if self.transform:
            image = self.transform(image)
        sample = {
            "image": image,
            "high": high,
            "mid": mid,
            "low": low,
            "hasher": hasher,
            "diagcode": diagcode,
            "fitzpatrick": fitzpatrick,
        }
        return sample


def flatten(list_of_lists):
    """
    Taken as is from:
    https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py#L22
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


if __name__ == "__main__":
    # This is the path to the Fitzpatrick17k dataset.
    # all_data_list: str = "../Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_F_A2_F_0.99_0.70_FC_F_KeepOne_Out_F_OutThresh_None_0FST_F.csv"

    # This is the path to the Fitzpatrick17k-C dataset.
    all_data_list: str = "../Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv"

    test_all_holdout_partitions(all_data_list)
