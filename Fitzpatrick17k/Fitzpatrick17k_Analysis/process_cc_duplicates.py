import pandas as pd
from copy import deepcopy
from pathlib import Path

from definitions import *
from utils import *
from config import *


def do_process_cc_duplicates(path_map: Dict[str, str]) -> Set[str]:
    assert Path(FASTDUP_ALL_CC).exists()
    assert Path(FASTDUP_SELECTED_CC).exists()
    assert Path(FASTDUP_95_PLUS_A1).exists()
    assert Path(FASTDUP_90_95_A1).exists()

    # Working with connected components

    all_cc_df = pd.read_csv(
        FASTDUP_ALL_CC,
        header="infer",
        converters={"files": pd.eval, "files_ids": pd.eval},
    )
    selected_cc_df = pd.read_csv(FASTDUP_SELECTED_CC, header="infer")

    selected_cc_df_detailed = pd.merge(
        selected_cc_df,
        all_cc_df[["component_id", "files", "files_ids"]],
        on="component_id",
        how="left",
    )
    selected_cc_df_detailed["filenames_with_dir"] = selected_cc_df_detailed.apply(
        lambda row: [get_filename_with_dir(filepath) for filepath in row["files"]],
        axis=1,
    )

    selected_cc_df_detailed["filenames"] = selected_cc_df_detailed.apply(
        lambda row: [get_filenames(filepath) for filepath in row["files"]], axis=1
    )

    selected_cc_filelist = selected_cc_df_detailed["filenames"].to_list()
    selected_cc_filelist_flat = [
        item for cluster in selected_cc_filelist for item in cluster
    ]

    print(f"Number of selected CCs: {len(selected_cc_df)}")
    print(f"Number of images in selected CCs: {len(selected_cc_filelist_flat)}")

    # Working with duplicates

    if USE_A2_ANNOTATIONS:
        duplicate_annotations_df = pd.concat(
            [
                pd.read_csv(
                    FASTDUP_95_PLUS_A1, header=None, names=["pair_name", "is_duplicate"]
                ),
                pd.read_csv(
                    FASTDUP_95_PLUS_A2, header=None, names=["pair_name", "is_duplicate"]
                ),
                pd.read_csv(
                    FASTDUP_90_95_A1, header=None, names=["pair_name", "is_duplicate"]
                ),
            ]
        )
    else:
        duplicate_annotations_df = pd.concat(
            [
                pd.read_csv(
                    FASTDUP_95_PLUS_A1, header=None, names=["pair_name", "is_duplicate"]
                ),
                pd.read_csv(
                    FASTDUP_90_95_A1, header=None, names=["pair_name", "is_duplicate"]
                ),
            ]
        )

    duplicate_annotations_df = duplicate_annotations_df[
        duplicate_annotations_df["is_duplicate"] == "Duplicate"
    ]

    duplicate_annotations_pairs = [
        [
            "_".join(item.split("_")[:4]),
            "_".join(item.split("_")[4:]).replace(".jpg", ""),
        ]
        for item in duplicate_annotations_df["pair_name"].to_list()
    ]

    print(f"Number of duplicate pairs: {len(duplicate_annotations_pairs)}")
    print(
        f"Number of unique images in duplicate pairs: {len(set([item for pair in duplicate_annotations_pairs for item in pair]))}"
    )

    # Working with cleanlab duplicates

    cleanlab_duplicate_pairs = (
        pd.read_csv(CLEANLAB_DUPLICATE_PAIRS, header="infer").to_numpy().tolist()
    )
    cleanlab_duplicate_pairs = [
        [get_filenames(img) for img in pair] for pair in cleanlab_duplicate_pairs
    ]
    print(f"Number of cleanlab duplicate pairs: {len(cleanlab_duplicate_pairs)}")
    print(
        f"Number of unique images in cleanlab duplicate pairs: {len(set([item for pair in cleanlab_duplicate_pairs for item in pair]))}"
    )

    incomplete_clusters: List[List[str]] = deepcopy(selected_cc_filelist)
    incomplete_clusters.extend(duplicate_annotations_pairs)
    incomplete_clusters.extend(cleanlab_duplicate_pairs)

    complete_clusters: List[List[str]] = make_complete_clusters(incomplete_clusters)

    print(f"Number of complete clusters: {len(complete_clusters)}")
    print(
        f"Number of unique images in complete clusters: {len(set([item for cluster in complete_clusters for item in cluster]))}"
    )

    # Save the clusters to a file.
    with open(ANANLYSES_OUTPUT_PATH / "complete_clusters.txt", "w") as f:
        for cluster in complete_clusters:
            f.write(f"{cluster}\n")

    # Determine which, if any, images to keep from homogeneous clusters.
    # A homogeneous cluster is a cluster that contains images with the same "diag"
    # and "fst" labels.
    candidate_images_to_keep_from_homogeneous_clusters = (
        find_candidate_images_from_homogenous_clusters(
            list_of_clusters=complete_clusters, path_map=path_map
        )
    )
    # Save the candidate images to keep to a file.
    with open(
        ANANLYSES_OUTPUT_PATH
        / "candidate_images_to_keep_from_homogeneous_clusters.txt",
        "w",
    ) as f:
        for item in candidate_images_to_keep_from_homogeneous_clusters:
            f.write(f"{item}\n")

    if REMOVE_CC_DUPLICATES_LEVEL == "All":
        return set([item for cluster in complete_clusters for item in cluster])

    elif REMOVE_CC_DUPLICATES_LEVEL == "KeepOne":
        return set(
            [item for cluster in complete_clusters for item in cluster]
        ).difference(set(candidate_images_to_keep_from_homogeneous_clusters))
    else:
        raise ValueError(
            f"REMOVE_CC_DUPLICATES_LEVEL must be either 'KeepOne' (default) or 'All'. Got '{REMOVE_CC_DUPLICATES_LEVEL}'."
        )
