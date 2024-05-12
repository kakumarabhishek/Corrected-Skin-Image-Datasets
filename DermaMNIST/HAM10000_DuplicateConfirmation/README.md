# HAM10000_DuplicateConfirmation

Directory containing the code and the results for our duplicate detection of the HAM10000 dataset.

## Directory Structure

* [`AnalysesOutputFiles/`](AnalysesOutputFiles/): Directory containing file lists (`.csv`) of duplicate images pairs ([`newly_discovered_duplicates.csv`](AnalysesOutputFiles/newly_discovered_duplicates.csv)) and near-duplicates-but-different-lesions ([`near_duplicates_different_times.csv`](AnalysesOutputFiles/near_duplicates_different_times.csv)).
* [`Visualizations/`](Visualizations/): Directory containing all the visualizations of image pairs: [most similar](Visualizations/MostSimilarPairs/new_duplicates_vis/) and [least similar](Visualizations/LeastSimilarPairs/).
* [`fastdup_features/`](fastdup_features/): Directory containing the image embeddings for HAM10000 images calculated using `fastdup`.
* [`fastdup_outputs/`](fastdup_outputs/): Directory containing the list of 1,000 most similar image pairs in HAM10000 as a `.csv` file.
* [`check_least_similar.py`](check_least_similar.py): Script to visualize the 5 least similar image pairs in HAM10000's clusters of 2, 3, 4, 5, and 6.
* [`check_most_similar.py`](check_most_similar.py): Script to visualize the 1,000 most similar image pairs in HAM10000, and store them in directories in intervals of 100.