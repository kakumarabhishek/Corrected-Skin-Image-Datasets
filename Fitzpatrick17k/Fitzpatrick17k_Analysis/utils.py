import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Set, Dict, Optional

from definitions import *
from config import *


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Check if a matrix is symmetric.
    Source: # https://stackoverflow.com/a/42913743

    Parameters:
    a (numpy.ndarray): The input matrix.
    rtol (float, optional): The relative tolerance parameter for the allclose function. Default is 1e-05.
    atol (float, optional): The absolute tolerance parameter for the allclose function. Default is 1e-08.

    Returns:
    bool: True if the matrix is symmetric, False otherwise.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def filelist_to_df(filelist: List[str]) -> pd.DataFrame:
    """
    Convert a list of file paths to a pandas DataFrame.

    Args:
        filelist (list): A list of file paths.

    Returns:
        df (pandas.DataFrame): A DataFrame with columns 'filepath', 'filename', 'diag', and 'fst'.
    """
    df = pd.DataFrame({"filepath": filelist})
    df["filename"] = df.apply(
        lambda row: row.filepath.split("/")[-1].split(".")[0], axis=1
    )
    df["diag"] = df.apply(lambda row: row.filename.split("_")[0], axis=1)
    df["fst"] = df.apply(lambda row: row.filename.split("_")[1][1], axis=1)

    return df


def get_filename_with_dir(filepath: str) -> str:
    """
    Returns the filename with directory path.

    Args:
        filepath (str): The filepath of the file.

    Returns:
        str: The filename with directory path.
    """
    return (
        filepath
        if str(DATASET_PATH) not in filepath
        else "/".join(filepath.split("/")[-2:])
    )


def get_filenames(filepath: str) -> str:
    """
    Extracts the filename from the given filepath, and removes the extension.

    Args:
        filepath (str): The path of the file.

    Returns:
        str: The filename without the extension.
    """
    return (
        filepath.split(".")[0]
        if "/" not in filepath
        else filepath.split("/")[-1].split(".")[0]
    )


def get_disease_name(filepath: str) -> str:
    """
    Extracts the disease name (abbreviated code) from the given filepath.

    Args:
        filename (str): The path of the file.

    Returns:
        str: The disease code.
    """
    return get_filenames(filepath).split("_")[0]


def get_fst_label(filepath: str) -> str:
    """
    Extracts the FST label from the given filepath.

    Args:
        filepath (str): The path of the file.

    Returns:
        str: The FST label.
    """
    return get_filenames(filepath).split("_")[1][1]


def get_trunc_md5(filepath: str) -> str:
    """
    Extracts the truncated MD5 hash from the given filepath.

    Args:
        filepath (str): The path of the file.

    Returns:
        str: The truncated MD5 hash.
    """
    return get_filenames(filepath).split("_")[3]


def get_dataset_type_name() -> str:
    """
    Returns the name of the dataset type based on the parameters in the `config.py` file.
    For example: `SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv`
    denotes:
    - USE_SIMILARITY_THRESHOLD = True
    - USE_A2_ANNOTATIONS = True
    - SIMILARITY_THRESHOLD_UPPER = 0.99
    - SIMILARITY_THRESHOLD_LOWER = 0.70
    - REMOVE_CC_DUPLICATES_FASTDUP_CLEANLAB = True
    - REMOVE_CC_DUPLICATES_LEVEL = "KeepOne"
    - REMOVE_OUTLIERS = True
    - OUTLIER_SIMILARITY_THRESHOLD = None
    - REMOVE_MISSING_FST = False


    Returns:
        str: The name of the dataset type.
    """
    # Function to convert boolean to string flag (T/F).
    bool2str = lambda s: "T" if s is True else "F"

    fname: str = ""

    # Flag to use similarity threshold.
    fname += f"SimThresh_{bool2str(USE_SIMILARITY_THRESHOLD)}"
    # Flag to use annotator 2's labels.
    fname += f"_A2_{bool2str(USE_A2_ANNOTATIONS)}"
    # Upper and lower similarity thresholds.
    fname += f"_{SIMILARITY_THRESHOLD_UPPER:.2f}_{SIMILARITY_THRESHOLD_LOWER:.2f}"
    # Flag to remove connected components of duplicates, obtained from fastdup and cleanlab analyses.
    fname += f"_FC_{bool2str(REMOVE_CC_DUPLICATES_FASTDUP_CLEANLAB)}"
    # Flag to set policy for removing clusters of duplicates.
    fname += f"_{REMOVE_CC_DUPLICATES_LEVEL}"
    # Flag to remove outliers, obtained from fastdup analysis.
    fname += f"_Out_{bool2str(REMOVE_OUTLIERS)}"
    # Flag to set the outlier similarity threshold.
    fname += (
        f"_OutThresh_{OUTLIER_SIMILARITY_THRESHOLD:.2f}"
        if OUTLIER_SIMILARITY_THRESHOLD
        else f"_OutThresh_None"
    )
    # Flag to remove images with missing FST labels.
    fname += f"_0FST_{bool2str(REMOVE_MISSING_FST)}"
    # Add the file extension.
    fname += f".csv"

    return fname


#######################################################################################
#######################################################################################
# We are given a list of image clusters, where each cluster contains a list of images,
# and a list of duplicate image pairs. For each pair of images in the list of
# duplicates, check if either of the images in the pair is already present in any of
# the clusters. If yes, expand the appropriate cluster to include the pair of images,
# and remove the pair from the list of duplicates. Repeat this process until no more
# pairs can be added to any of the clusters.


def update_clusters_with_duplicates(pair: List[str], list_clusters: List[list]) -> bool:
    """
    Updates the clusters in the list with the given pair of images.

    Args:
        pair (list): A pair of images to be added to the clusters.
        list_clusters (list): A list of clusters.

    Returns:
        bool: True if the clusters were updated, False otherwise.
    """
    for cluster in list_clusters:
        if any(image in cluster for image in pair):
            cluster.extend(pair)
            return True
    return False


def process_duplicate_pairs(
    list_duplicates: List[tuple], list_clusters: List[list]
) -> None:
    """
    Process duplicate pairs by updating the clusters and removing the processed pairs from the list of duplicates.

    Args:
        list_duplicates (list): List of duplicate pairs.
        list_clusters (list): List of clusters.

    Returns:
        None
    """
    for pair in list_duplicates.copy():
        if update_clusters_with_duplicates(pair, list_clusters):
            list_duplicates.remove(pair)


#######################################################################################
#######################################################################################


#######################################################################################
#######################################################################################

# We have a list of image clusters, that are not completely clustered. This means that
# a cluster has been broken into more than 1 part. For example, if we have a list of
# incomplete image clusters such as

# [
#   ["A", "B"],
#   ["C", "D"],
#   ["E", "F"],
#   ["G", "H"],
#   ["A", "C"],
#   ["E", "D", "H"],
#   ["E", "I"],
#   ["L", "K"],
#   ["K", "N", "M", "J", "N"],
#   ["P", "T"]
# ],

# then the complete clustering would result in:

# [
#   ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
#   ['L', 'K', 'N', 'M', 'J']
#   ['T', 'P']
# ].

# In this example, ("A", "B"), ("C", "D"), ("C", "A"), and ("D", "G") are duplicate
# pairs. But since "A" and "B" are duplicates and "C" and "A" are duplicates, all "A",
# "B", and "C" are duplicates of each other. Similar logic applies to other duplicates.


class UnionFind:
    """
    The `UnionFind` class is a data structure that keeps track of a partition of a set
    into disjoint subsets. It provides two main operations: `find()` and `union()`.
    """

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        """
        The `find` method takes an element `x` as input and returns the root of the
        subset that `x` belongs to. If `x` is not already in the `parent` dictionary,
        it adds `x` to the dictionary with `x` as its own parent and returns `x`. If
        `x` is its own parent (meaning it's the root of its subset), it returns `x`.
        Otherwise, it recursively calls `find` on the parent of `x` and updates the
        parent of `x` to be the root of its subset. This process is known as path
        compression and helps to keep the tree flat, improving the efficiency of
        future operations.
        """
        if x not in self.parent:
            self.parent[x] = x
            return x
        elif self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """
        The `union` method takes two elements `x` and `y`, finds their roots using the
        `find` method, and if they are not already in the same subset, it makes the
        root of `y`'s subset the parent of the root of `x`'s subset, effectively
        merging the two subsets.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def make_complete_clusters(incomplete_clusters: List[List[str]]) -> List[List[str]]:
    """
    The `make_complete_clusters` function takes a list of incomplete clusters as input,
    where each cluster is a list of elements. It initializes a `UnionFind` object and
    for each pair of consecutive elements in each cluster, it calls the `union`
    method to group them into the same subset. Then it creates a dictionary `clusters`
    where each key is a root of a subset and its value is a list of all elements in
    that subset. Finally, it returns a list of all clusters, where each cluster is a
    list of its elements in the same order as they appear in the clusters. This
    function essentially completes the clusters by finding all connected components in
    the graph where each cluster represents a connected component.
    """
    uf = UnionFind()

    for cluster in incomplete_clusters:
        for i in range(1, len(cluster)):
            uf.union(cluster[i - 1], cluster[i])

    clusters: Dict[str, List[str]] = {}
    for item in uf.parent:
        root = uf.find(item)
        if root not in clusters:
            clusters[root] = [item]
        else:
            clusters[root].append(item)

    return [cluster for cluster in clusters.values()]


#######################################################################################
#######################################################################################


#######################################################################################
#######################################################################################

# We have a list of image clusters, where each cluster contains a list of images. All
# the image files are named as `<diagnosis>_<fst>_<image_number>_<md5>.jpg`. For each
# cluster, we want to determine if all the images in the cluster have the same
# diagnosis and fst. If no, do nothing. But if yes, append the image in that cluster
# which has the largest spatial resolution to the list of images to be retained.


def is_same_diag_fst(cluster: List[str]) -> bool:
    """
    Check if all images in the cluster have the same values for "diag" and "fst".

    Args:
        cluster (List[str]): List of image paths.

    Returns:
        bool: True if "diag" and "fst" are the same for all images, False otherwise.
    """

    if not cluster:
        return False

    # Extract attributes from the first image in the cluster
    img1 = cluster[0]
    diag1, fst1, _, _ = img1.split("_")

    # Check if all images in the cluster have the same values for diag and fst
    for img2 in cluster:
        diag2, fst2, _, _ = img2.split("_")
        if diag2 != diag1 or fst2 != fst1:
            return False

    return True


def find_largest_resolution_image(
    cluster: List[str], path_map: Dict[str, str]
) -> Optional[str]:
    """
    Find the image with the largest spatial resolution within a cluster.

    Args:
        cluster (List[str]): List of image paths.

    Returns:
        Optional[str]: Image path with the largest resolution, or None if the cluster is empty.
    """
    if not cluster:
        return None

    return max(cluster, key=lambda img: Image.open(path_map[img]).size)


def find_candidate_images_from_homogenous_clusters(
    list_of_clusters: List[List[str]], path_map: Dict[str, str]
) -> List[Optional[str]]:
    """
    Process a list of image clusters and return a list of images with the largest
    resolution for homogenous clusters. A homogeneous cluster is a cluster that
    contains images with the same "diag" and "fst" labels.

    Args:
        list_of_clusters (List[List[str]]): List of image clusters.

    Returns:
        List[Optional[str]]: List of images with the largest resolution for each relevant cluster.
    """
    images_to_keep: List[Optional[str]] = []

    for cluster in list_of_clusters:
        if is_same_diag_fst(cluster):
            largest_resolution_image = find_largest_resolution_image(cluster, path_map)
            if largest_resolution_image:
                images_to_keep.append(largest_resolution_image)

    return images_to_keep
