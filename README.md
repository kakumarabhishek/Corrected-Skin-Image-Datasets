[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11101337.svg)](https://doi.org/10.5281/zenodo.11101337)
 [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets

This repository contains the code accompanying our paper titled "[Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets](https://arxiv.org/abs/2401.14497)".

## Repository Structure

The repository is structured as:
* [`DermaMNIST/`](DermaMNIST/): Parent directory for DermaMNIST analysis and benchmarking experiments.
    * [`DermaMNIST_Analysis/`](DermaMNIST/DermaMNIST_Analysis/): Directory containing the code for all our analysis of the DermaMNIST dataset, including the preparation of DermaMNIST-C and DermaMNIST-E datasets.
    * [`DermaMNIST_Training/`](DermaMNIST/DermaMNIST_Training/): Directory containing the code for reproducing our benchmark experiment results, including the new [`medmnist_corrected`](DermaMNIST/DermaMNIST_Training/medmnist_corrected/).
* [`Fitzpatrick17k/`](Fitzpatrick17k/): Parent directory for Fitzpatrick17k analysis and benchmarking experiments.
    * [`Fitzpatrick17k_Analysis/`](Fitzpatrick17k/Fitzpatrick17k_Analysis/): Directory containing the code for our cleaning pipeline for the Fitzpatrick17k dataset, including the preparation of the Fitzpatrick17k-C dataset.
    * [`Fitzpatrick17k_Training/`](Fitzpatrick17k/Fitzpatrick17k_Training/): Directory containing the code for reproducing our benchmark experiment results.

## Metadata Files

The metadata files for the datasets released with this work are listed below:

- [DermaMNIST-C](DermaMNIST/DermaMNIST_Analysis/CSV_files/combined_metadata_corrected-HAM10000_corrected.csv)
- [DermaMNIST-E](DermaMNIST/DermaMNIST_Analysis/CSV_files/combined_extended.csv)
- [Fitzpatrick17k-C](Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv)

These files are also available on this project's [Zenodo repository](https://doi.org/10.5281/zenodo.11101337).

## Online Resource

Additional visualizations and links to the new datasets: DermaMNIST-C, DermaMNIST-E, and Fitzpatrick17k-C are available on [the project website](https://derm.cs.sfu.ca/critique/).

## Zenodo Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11101337.svg)](https://doi.org/10.5281/zenodo.11101337)

The datasets released with this work: DermaMNIST-E, DermaMNIST-C, and Fitzpatrick17k-C are available on [Zenodo](https://doi.org/10.5281/zenodo.11101337).

## License and Citation

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The datasets released on [Zenodo](https://doi.org/10.5281/zenodo.11101337) are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


If you use our newly proposed datasets or our analyses, please cite [our paper](https://arxiv.org/abs/2401.14497) and [our Zenodo](https://doi.org/10.5281/zenodo.11101337) repository. The corresponding BibTeX entries are:

```
@article{abhishek2024investigating,
  title = {Investigating the Quality of {DermaMNIST} and {Fitzpatrick17k} Dermatological Image Datasets},
  author = {Abhishek, Kumar and Jain, Aditi and Hamarneh, Ghassan},
  journal = {arXiv preprint arXiv:2401.14497},
  doi = {10.48550/ARXIV.2401.14497},
  url = {https://arxiv.org/abs/2401.14497},
  year={2024}
}

@dataset{abhishek_2024_11101337,
  title = {{Investigating the Quality of {DermaMNIST} and {Fitzpatrick17k} Dermatological Image Datasets}},
  month = May,
  year = 2024,
  author = {Abhishek,  Kumar and Jain,  Aditi and Hamarneh,  Ghassan},
  language = {en},
  publisher = {Zenodo},
  doi = {10.5281/ZENODO.11101337},
  url = {https://zenodo.org/doi/10.5281/zenodo.11101337},
}
```
