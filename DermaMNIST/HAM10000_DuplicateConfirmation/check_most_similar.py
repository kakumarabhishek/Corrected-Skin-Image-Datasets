import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

DUPLICATE_PAIRS_CSV = Path("./fastdup_outputs/duplicates_1000.csv")
HAM10000_METADATA_CSV = Path("../HAM10000_metadata.csv")

# HAM10000_DATA_DIR = Path(
#     "/local-scratch2/Datasets/ISBI_ISIC/2018/Task3/ISIC2018_Task3_Training_Input/"
# )
# Move to RAM disk for faster access.
HAM10000_DATA_DIR = Path("/dev/shm/myramdisk/HAM10000")
DUPLICATES_VIS_DIR = Path("./Visualizations/MostSimilarPairs/new_duplicates_vis")


def main() -> None:
    # Read the CSV files.
    duplicate_pairs_df = pd.read_csv(DUPLICATE_PAIRS_CSV, header="infer", sep=",")
    ham10000_metadata_df = pd.read_csv(HAM10000_METADATA_CSV, header="infer", sep=",")

    # Iterate over the duplicate pairs and check if the lesion IDs match.
    for idx, row in tqdm(duplicate_pairs_df.iterrows(), total=len(duplicate_pairs_df)):
        # Read each row to get the image IDs and the similarity score.
        sim, image_1, image_2 = (
            row["similarity"],
            row["from_img"].split(".")[0],
            row["to_img"].split(".")[0],
        )

        # Get the metadata for the 2 images.
        ham_row1 = ham10000_metadata_df[ham10000_metadata_df["image_id"] == image_1]
        ham_row2 = ham10000_metadata_df[ham10000_metadata_df["image_id"] == image_2]

        # Obtain the lesion IDs from the metadata.
        lesion_id_1, lesion_id_2 = (
            ham_row1.lesion_id.values[0],
            ham_row2.lesion_id.values[0],
        )

        # Check if the lesion IDs for the 2 images match.
        if lesion_id_1 != lesion_id_2:
            # If the lesion IDs do not match, print the details.
            print(f"\n[{sim:.4f}] Lesion IDs do not match for {image_1} and {image_2}.")
            print(f"{ham_row1.dataset.values[0]}, {ham_row2.dataset.values[0]}")

            # Further, check if the other attributes match. If not, print the details.
            for attribute in ["dx", "dx_type", "age", "sex", "localization", "dataset"]:
                if ham_row1[attribute].values[0] != ham_row2[attribute].values[0]:
                    print(
                        f"{attribute}: {ham_row1[attribute].values[0]} != {ham_row2[attribute].values[0]}"
                    )

            # Load the images and visualize the pair.
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            img1 = plt.imread(HAM10000_DATA_DIR / f"{row.from_img}")
            img2 = plt.imread(HAM10000_DATA_DIR / f"{row.to_img}")
            axs[0].imshow(img1)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_xlabel(row.from_img.split(".")[0])
            axs[0].set_ylabel(lesion_id_1)
            axs[1].imshow(img2)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xlabel(row.to_img.split(".")[0])
            axs[1].set_ylabel(lesion_id_2)

            fig.suptitle(f"Similarity: {row.similarity:.4f}", size=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            # Save the figure.
            # Create a directory to store the visualizations. We will create a new
            # directory for every 100 duplicate pairs analyzed.
            tmp_vis_dir = DUPLICATES_VIS_DIR / f"{(idx//100)}01-{(idx//100) + 1}00/"
            tmp_vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                tmp_vis_dir / f"{idx:03d}_{row.similarity:.4f}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            continue


if __name__ == "__main__":
    # Run this as
    # python check_most_similar.py > ./Visualizations/MostSimilarPairs/new_duplicates_vis/log.txt
    # to save the output to a log file.
    main()
