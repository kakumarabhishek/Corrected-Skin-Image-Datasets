import pandas as pd
from sklearn.metrics import cohen_kappa_score

from definitions import *

annotator_1_ratings = FASTDUP_95_PLUS_A1
annotator_2_ratings = FASTDUP_95_PLUS_A2


def main():
    # Read annotators ratings
    a1_df = pd.read_csv(FASTDUP_95_PLUS_A1, header=None, names=["img", "label"])
    a2_df = pd.read_csv(FASTDUP_95_PLUS_A2, header=None, names=["img", "label"])

    # Verify that both the annotators provided ratings for all the images
    assert a1_df["img"].equals(a2_df["img"])

    # Calculate the overlap between the two annotators
    overlap = (a1_df["label"] == a2_df["label"]).sum() / len(a1_df)

    # Calculate the Cohen's kappa
    kappa = cohen_kappa_score(a1_df["label"], a2_df["label"])

    return overlap, kappa


if __name__ == "__main__":
    overlap, kappa = main()
    print(f"Annotator overlap: {overlap:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
