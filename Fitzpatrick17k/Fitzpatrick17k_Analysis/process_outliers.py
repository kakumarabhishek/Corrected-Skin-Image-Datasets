import pandas as pd
from pathlib import Path

from definitions import *
from utils import *
from config import *


def do_process_outliers() -> Set[str]:
    assert Path(FASTDUP_SELECTED_OUTLIERS).exists()

    all_outliers_df = pd.read_csv(
        FASTDUP_SELECTED_OUTLIERS, header=None, names=["partial_path", "similarity"]
    )
    print(f"Number of outliers: {len(all_outliers_df)}")

    if OUTLIER_SIMILARITY_THRESHOLD:
        assert 0 <= OUTLIER_SIMILARITY_THRESHOLD <= 1
        all_outliers_df = all_outliers_df[
            all_outliers_df["similarity"] <= OUTLIER_SIMILARITY_THRESHOLD
        ]
        print(
            f"Number of outliers after applying similarity threshold: {len(all_outliers_df)}"
        )

    selected_outliers = [
        get_filenames(filepath)
        for filepath in all_outliers_df["partial_path"].to_list()
    ]

    # Save the list of selected outliers to a file.
    with open(ANANLYSES_OUTPUT_PATH / "selected_outliers.txt", "w") as f:
        for item in selected_outliers:
            f.write(f"{item}\n")

    return selected_outliers
