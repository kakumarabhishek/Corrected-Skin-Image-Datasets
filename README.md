# Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets

This repository contains the code accompanying our paper titled "Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets".

## Repository Structure

The repository is structured as:
* [`DermaMNIST/`](DermaMNIST/): Parent directory for DermaMNIST analysis and benchmarking experiments.
    * [`DermaMNIST_Analysis/`](DermaMNIST/DermaMNIST_Analysis/): Directory containing the code for all our analysis of the DermaMNIST dataset, including the preparation of DermaMNIST-C and DermaMNIST-E datasets.
    * [`DermaMNIST_Training/`](DermaMNIST/DermaMNIST_Training/): Directory containing the code for reproducing our benchmark experiment results, including the new [`medmnist_corrected`](DermaMNIST/DermaMNIST_Training/medmnist_corrected/).
* [`Fitzpatrick17k/`](Fitzpatrick17k/): Parent directory for Fitzpatrick17k analysis and benchmarking experiments.
    * [`Fitzpatrick17k_Analysis/`](Fitzpatrick17k/Fitzpatrick17k_Analysis/): Directory containing the code for our cleaning pipeline for the Fitzpatrick17k dataset, including the preparation of the Fitzpatrick17k-C dataset.
    * [`Fitzpatrick17k_Training/`](Fitzpatrick17k/Fitzpatrick17k_Training/): Directory containing the code for reproducing our benchmark experiment results.

## Online Resource

Additional visualizations and links to the new datasets (DermaMNIST-C, DermaMNIST-E, and Fitzpatrick17k-C) are available on [the project website](https://critique.kabhishe.com/).