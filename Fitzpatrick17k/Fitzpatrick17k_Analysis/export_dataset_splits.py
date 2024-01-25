import pandas as pd
from sklearn.model_selection import train_test_split

from definitions import *
from utils import *
from config import *


def do_save_dataset_splits(file_list_filtered):
    # Check that the directory to save the filtered file list exists.
    assert SAVE_FILTERED_FILE_LIST_PATH.exists()

    filtered_df = filelist_to_df(file_list_filtered)

    # Setting the label order for the `fst` column. This is useful for plotting.
    # https://stackoverflow.com/a/67205743
    filtered_df["fst"] = pd.Categorical(
        filtered_df["fst"], ["0", "1", "2", "3", "4", "5", "6"]
    )

    # Create a dictionary of dataframes for each split.
    df_splits_dict = {}
    splits = ["train", "val", "test"]

    # First, split the data into train and eval (val + test) in a 70:30 ratio.
    df_splits_dict["train"], tmp_df_eval = train_test_split(
        filtered_df, test_size=0.3, random_state=8888, stratify=filtered_df[["diag"]]
    )

    # Next, split the eval data into val and test in a 1:2 ratio.
    # This means that the final split is train:val:test = 70:10:20.
    df_splits_dict["val"], df_splits_dict["test"] = train_test_split(
        tmp_df_eval, test_size=0.6667, random_state=8888, stratify=tmp_df_eval[["diag"]]
    )

    # Then, for each split, add a column called `partition` to indicate the split.
    for split in splits:
        df_splits_dict[split]["partition"] = split

    # Concatenate the 3 partitions into a single dataframe.
    df_splits = pd.concat([df_splits_dict[split] for split in splits])

    # Check that the number of rows in the concatenated dataframe is the same as the
    # number of rows in the original dataframe constructed from the filtered file list.
    assert len(df_splits) == len(filtered_df)

    # Finally, we need to combine the partition labels with the dataset's metadata.
    # So, first, we read the metadata CSV file into a dataframe, and verify that the
    # number of rows in the dataframe is the same as the number of images in the
    # dataset.
    metadata_df = pd.read_csv(DATASET_METADATA_FILE, header="infer", index_col="index")
    assert len(metadata_df) == TOTAL_FITZPATRICK17K_IMAGES

    # Then, we add a column called `filename` to the metadata dataframe, which is
    # the filename without the .jpg extension.
    metadata_df["filename"] = metadata_df["new_img_name"].apply(get_filenames)

    # Also, in the dataset partitions dataframe, we update the `filepath` column to
    # remove the absolute path prefix.
    df_splits["filepath"] = df_splits["filepath"].apply(get_filename_with_dir)

    # Next, we merge the two dataframes on the `filename` column.
    df_splits = pd.merge(
        df_splits, metadata_df, on="filename", how="left", validate="one_to_one"
    )

    # Finally, save the dataframe to a CSV file.
    df_splits.to_csv(
        (SAVE_FILTERED_FILE_LIST_PATH / get_dataset_type_name()), index=False
    )
