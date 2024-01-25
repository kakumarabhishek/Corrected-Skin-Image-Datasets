# Fitzpatrick17k_Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Directory containing the code for our cleaning pipeline for the Fitzpatrick17k dataset, including the preparation of the Fitzpatrick17k-C dataset.

## Directory Structure

* [`AnalysesOutputFiles/`](AnalysesOutputFiles/): Directory containing the outputs of duplicate clustering and erroneous image detection analyses (`.txt`).
* [`DatasetSplits/`](DatasetSplits/): Directory containing consolidated metadata files (`.csv`) for various configurations of the data cleaning pipeline. Each `.csv` file contains {train, valid, test} partitions for that particular version of the dataset. The three files in this directory are:
    * [`SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv`](DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv): **The new Fitzpatrick17k-C dataset.**
    * [`SimThresh_F_A2_F_0.99_0.70_FC_F_KeepOne_Out_F_OutThresh_None_0FST_F.csv`](DatasetSplits/SimThresh_F_A2_F_0.99_0.70_FC_F_KeepOne_Out_F_OutThresh_None_0FST_F.csv): The original Fitzpatrick17k dataset.
    * [`SimThresh_T_A2_T_0.99_0.70_FC_T_All_Out_T_OutThresh_None_0FST_F.csv`](DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_All_Out_T_OutThresh_None_0FST_F.csv): A subset of the Fitzpatrick17k-C dataset that excludes all images from any duplicate clusters, unlike Fitzpatrick17k-C, which retains one image from homogenous clusters.
* [`FastdupOutputFiles/`](FastdupOutputFiles/): Directory containing the outputs of analyses using `fastdup` and `cleanvision` as well as the manual annotations from the two annotators.
* [`Fitzpatrick17k_metadata/`](Fitzpatrick17k_metadata/): Directory containing metadata for the Fitzpatrick17k dataset.
* [`annotator_agreement.py`](annotator_agreement.py): Code to calculate the overlap and the agreement between the two annotators.
* [`config.py`](config.py): Configuration parameters for the dataset cleaning pipeline.
* [`definitions.py`](definitions.py): Directory paths and filenames for the dataset cleaning pipeline.
* [`export_dataset_splits.py`](export_dataset_splits.py): Code to export the cleaned dataset to a `.csv` file.
* [`main.py`](main.py): The main dataset cleaning file that calls other modules.
* [`process_cc_duplicates.py`](process_cc_duplicates.py): Code to filter duplicate pairs and clusters detected using `fastdup` and `cleanvision`.
* [`process_outliers.py`](process_outliers.py): Code to filter erroneous images.
* [`process_similarity_threshold.py`](process_similarity_threshold.py): Code to filter images based on similarity score threshold(s).
* [`utils.py`](utils.py): Utility functions for various analysis tasks.
    
The filenames for datasets in [`DatasetSplits/`](DatasetSplits/) are obtained using `get_dataset_type_name()` from [`utils.py`](utils.py). Please look at the function's documentation to understand the naming convention adopted.