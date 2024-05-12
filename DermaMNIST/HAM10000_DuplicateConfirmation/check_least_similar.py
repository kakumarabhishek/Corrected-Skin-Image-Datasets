import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import fastdup
from sklearn.metrics.pairwise import cosine_similarity

# HAM10000 related variables.
HAM10000_METADATA_CSV = Path("../HAM10000_metadata.csv")
HAM10000_DATA_DIR = Path(
    "/local-scratch2/Datasets/ISBI_ISIC/2018/Task3/ISIC2018_Task3_Training_Input/"
)

# Fastdup related variables.
FEAT_DIM = 960
# pathlib Path object is not used because fastdup does not support it.
FEAT_DIR = "./fastdup_features"
VIS_DIR = "./Visualizations/LeastSimilarPairs"


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Check if a matrix is symmetric.
    # https://stackoverflow.com/a/42913743

    Args:
        a (np.ndarray): The matrix to check.
        rtol (float, optional): Tolerance when comparing relative differences.
                                Defaults to 1e-05.
        atol (float, optional): Tolerance when comparing absolute differences.
                                Defaults to 1e-08.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    # Check if the matrix is square
    if a.shape[0] != a.shape[1]:
        return False

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def main(top_K_least_similar: int = 5) -> None:
    """
    Main function for computing and visualizing the least similar pairs of images.

    This function loads the features and filenames from a binary file and a CSV file,
    computes the pairwise cosine similarity between the features of all images,
    filters the metadata based on cluster size, and finds the least similar pairs of images
    within each cluster. It then visualizes the top K least similar pairs of images.

    Args:
        None

    Returns:
        None
    """
    # Load the features and the filenames.
    # The features are stored in a binary file, and the filenames have to be stored in
    # a CSV file in the same location as the binary file, and the CSV file must be
    # named same as the binary file with ".csv" appended.
    file_list, feats = fastdup.load_binary_feature(
        filename=f"{FEAT_DIR}/atrain_features.dat", d=FEAT_DIM
    )
    print(
        f"Successfully loaded {feats.shape[1]}-d features from {len(file_list)} images."
    )

    # Compute the pairwise cosine similarity between the features of all images.
    pairwise_similarity_vals = cosine_similarity(feats)
    # Check if the similarity matrix is symmetric.
    assert check_symmetric(
        pairwise_similarity_vals
    ), "Similarity matrix is not symmetric."

    # Create a dictionary to map filenames to their index in the feature matrix.
    filenames = dict([(f.split("/")[-1], idx) for (idx, f) in enumerate(file_list)])

    # Load the HAM10000 metadata.
    HAM10000_METADATA = pd.read_csv(HAM10000_METADATA_CSV, header="infer", sep=",")

    # Iterate through different cluster sizes.
    for cluster_size in range(2, 7):
        # Filter the metadata to only include lesions with the specified cluster size.
        HAM10000_METADATA_filtered = HAM10000_METADATA.groupby("lesion_id").filter(
            lambda group: len(group) == cluster_size
        )
        print("\n\nCluster size:", cluster_size)

        # Create an empty list to store the least similar pairs.
        least_similar_pairs = []

        # Iterate through each lesion group.
        for lesion_id, group in HAM10000_METADATA_filtered.groupby("lesion_id"):
            # This is to skip the lesions with only one image.
            # This check is not necessary if the metadata is filtered correctly.
            if len(group) < 2:
                continue
            # Reset the index of the group.
            group = group.reset_index(drop=True)
            # Convert the image IDs to a list.
            images = group["image_id"].tolist()

            # Find the least similar pair of images in the group.
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    sim = pairwise_similarity_vals[
                        filenames[images[i] + ".jpg"],
                        filenames[images[j] + ".jpg"],
                    ]
                    # For all pairs, add the similarity value to the list.
                    least_similar_pairs.append((lesion_id, images[i], images[j], sim))

        # Sort by similarity and select the top K least similar pairs based on the
        # similarity value. The similarity value is the 4th element in the tuple.
        least_similar_pairs.sort(key=lambda x: x[3])

        # Select the top K least similar pairs.
        try:
            top_5_least_similar_pairs = least_similar_pairs[:top_K_least_similar]
        # If there are less than K pairs, select all pairs.
        except IndexError:
            top_5_least_similar_pairs = least_similar_pairs

        print("Top 5 least similar pairs:")
        for idx, pair in enumerate(top_5_least_similar_pairs):
            print(pair)

            # Create a directory to store the visualizations.
            tmp_vis_dir = Path(VIS_DIR) / f"least_similar_{cluster_size}"
            tmp_vis_dir.mkdir(parents=True, exist_ok=True)

            # Load the images and visualize the pair.
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            img1 = plt.imread(HAM10000_DATA_DIR / f"{pair[1]}.jpg")
            img2 = plt.imread(HAM10000_DATA_DIR / f"{pair[2]}.jpg")
            axs[0].imshow(img1)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_xlabel(pair[1])
            axs[0].set_ylabel(pair[0])
            axs[1].imshow(img2)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xlabel(pair[2])

            fig.suptitle(f"Similarity: {pair[3]:.4f}", size=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            # Save the figure.
            plt.savefig(tmp_vis_dir / f"{idx}.png", dpi=300, bbox_inches="tight")
            plt.close()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the least similar pairs of images."
    )
    parser.add_argument(
        "--top_K_least_similar",
        type=int,
        default=5,
        help="Number of least similar pairs to visualize.",
    )
    args = parser.parse_args()
    main(top_K_least_similar=args.top_K_least_similar)
