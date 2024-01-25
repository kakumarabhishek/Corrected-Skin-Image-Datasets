# DermaMNIST_Analysis

Directory containing the code for all our analysis of the DermaMNIST dataset, including the preparation of DermaMNIST-C and DermaMNIST-E datasets.

## Directory Structure

* [`CSV_files/`](CSV_files/): Directory containing metadata files (`.csv`).
* [`NPZ_files/`](NPZ_files/): Directory containing datasets files as NumPy arrays (`.npz`) for `28 × 28` spatial resolution. The `.npz` files for `224 × 224` resolution are too large for GitHub (> 1 GB).
* [`DermaMNIST-Corrected.ipynb`](DermaMNIST-Corrected.ipynb): Notebook containing code to generate the DermaMNIST-C dataset.
* [`DermaMNIST-Extended.ipynb`](DermaMNIST-Extended.ipynb): Notebook containing code to generate the DermaMNIST-E dataset.
* [`DermaMNIST_plots.ipynb`](DermaMNIST_plots.ipynb): Notebook containing code to reproduce the plots in our paper.
* [`PartitionStats.ipynb`](PartitionStats.ipynb): Notebook containing code to plot dataset statistics for DermaMNIST and DermaMNIST-C datasets.
* [`dermamnist_split_info.csv`](dermamnist_split_info.csv): CSV file containing `image_id` to `split` (dataset partition) mapping for DermaMNIST.
* [`HAM10000_metadata.csv`](HAM10000_metadata.csv): CSV file containing HAM10000 metadata, which includes the `lesion_id` to `image_id` mapping.