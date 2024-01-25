import os
from pathlib import Path

# Path to the dataset
DATASET_PATH = Path(
    "/localhome/kabhishe/WorkingDir/Fitzpatrick17k_all_Categorized_AbbrvName/"
)

# Path to the dataset metadata file, with filenames containing abbreviated disease
# labels and FST labels.
DATASET_METADATA_FILE = Path(
    "./Fitzpatrick17k_metadata/fitzpatrick17k_detailed_abbreviated_names.csv"
)

# Path to the output files of Fastdup, Cleanlab, and manual annotations
DATA_PATH = Path("./FastdupOutputFiles/")

# Path to the output files of the analyses
ANANLYSES_OUTPUT_PATH = Path("./AnalysesOutputFiles/")

# Fastdup feature dimensionality
FEAT_DIM = 960

# Fastdup extracted features
# We have to convert this to a str because the fastdup code expects a str, not a Path.
FASTDUP_FEATURES = f"{str(DATA_PATH)}/atrain_features.dat"
FASTDUP_FILE_LIST = f"{str(DATA_PATH)}/atrain_features.dat.csv"

# Fastdup related outputs
FASTDUP_ALL_CC = DATA_PATH / "components.csv"
FASTDUP_SELECTED_CC = DATA_PATH / "selected_components_ordered_dump_from_website.csv"
FASTDUP_SELECTED_OUTLIERS = DATA_PATH / "outlier_similarity_dump_from_website.csv"

# Fastdup related manual annotations
FASTDUP_95_PLUS_A1 = DATA_PATH / "ManualVerification_0.95+_A1.csv"
FASTDUP_95_PLUS_A2 = DATA_PATH / "ManualVerification_0.95+_A2.csv"
FASTDUP_90_95_A1 = DATA_PATH / "ManualVerification_0.90-0.95_A1.csv"

# Cleanlab related outputs
CLEANLAB_DUPLICATE_PAIRS = DATA_PATH / "cleanlab_only_duplicates.csv"

# Path to the cleaned dataset with training, validation, and test partitions.
SAVE_FILTERED_FILE_LIST_PATH = Path("./DatasetSplits/")
