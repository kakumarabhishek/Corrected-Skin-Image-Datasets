import numpy as np
from copy import deepcopy

import fastdup
from sklearn.metrics.pairwise import cosine_similarity

from definitions import *
from utils import *
from config import *
from process_cc_duplicates import do_process_cc_duplicates
from process_similarity_threshold import do_process_similarity_threshold
from process_outliers import do_process_outliers
from export_dataset_splits import do_save_dataset_splits

file_list, feats = fastdup.load_binary_feature(filename=FASTDUP_FEATURES, d=FEAT_DIM)

assert len(file_list) == TOTAL_FITZPATRICK17K_IMAGES
assert feats.shape == (TOTAL_FITZPATRICK17K_IMAGES, FEAT_DIM)

# Convert the list of file paths to a pandas DataFrame.
file_list_df = filelist_to_df(file_list)
# Dictionary mapping filename (without the .jpg) to absolute filepath.
# This is useful for looking up the filepath of a given filename.
# https://stackoverflow.com/a/17426500
filename_to_path_dict = dict(zip(file_list_df.filename, file_list_df.filepath))

file_list_filtered = deepcopy(file_list)

pairswise_similarities = cosine_similarity(feats)
assert check_symmetric(pairswise_similarities) == True

pairwise_similarities_nondiag = deepcopy(pairswise_similarities)
np.fill_diagonal(pairwise_similarities_nondiag, -1)

max_similarities = np.max(pairwise_similarities_nondiag, axis=0)
assert max_similarities.shape == (TOTAL_FITZPATRICK17K_IMAGES,)

if USE_SIMILARITY_THRESHOLD:
    file_list_filtered = do_process_similarity_threshold(
        max_similarities, file_list_filtered
    )

filenames_to_be_removed = []

if REMOVE_CC_DUPLICATES_FASTDUP_CLEANLAB:
    filenames_to_be_removed.extend(
        do_process_cc_duplicates(path_map=filename_to_path_dict)
    )

if REMOVE_OUTLIERS:
    filenames_to_be_removed.extend(do_process_outliers())

# print(len(filenames_to_be_removed), len(set(filenames_to_be_removed)))

filepaths_to_be_removed = [
    filename_to_path_dict[filename] for filename in filenames_to_be_removed
]

# print(len(filepaths_to_be_removed), len(set(filepaths_to_be_removed)))

file_list_filtered = [
    filepath
    for filepath in file_list_filtered
    if filepath not in filepaths_to_be_removed
]

file_list_filtered_diseases = [
    get_disease_name(filepath) for filepath in file_list_filtered
]

if REMOVE_MISSING_FST:
    file_list_filtered = [
        filepath for filepath in file_list_filtered if get_fst_label(filepath) != "0"
    ]

print(
    f"Number of images after removing duplicates and outliers: {len(file_list_filtered)}"
)
print(
    f"Number of unique diseases after removing duplicates and outliers: {len(set(file_list_filtered_diseases))}"
)

if SAVE_FILTERED_FILE_LISTS:
    do_save_dataset_splits(file_list_filtered)
    print("Saved filtered file lists.")
