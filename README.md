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
    * [`HAM10000_DuplicateConfirmation/`](DermaMNIST/HAM10000_DuplicateConfirmation/): Directory containing the code and results for detecting new duplicate image pairs in the HAM10000 dataset.
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

The datasets released with this work: DermaMNIST-C, DermaMNIST-E, and Fitzpatrick17k-C are available on [Zenodo](https://doi.org/10.5281/zenodo.11101337).

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
  year = {2024}
}

@dataset{abhishek_2024_11101337,
  title = {{Investigating the Quality of {DermaMNIST} and {Fitzpatrick17k} Dermatological Image Datasets}},
  month = May,
  year = 2024,
  author = {Abhishek, Kumar and Jain, Aditi and Hamarneh, Ghassan},
  language = {en},
  publisher = {Zenodo},
  doi = {10.5281/ZENODO.11101337},
  url = {https://zenodo.org/doi/10.5281/zenodo.11101337},
}
```

## Dataset Acknowledgements

We would like to thank the authors of the original papers: DermaMNIST ([_ISBI_ 2021](https://doi.org/10.1109/ISBI48211.2021.9434062), [_Nat Sci Data_ 2023](https://doi.org/10.1038/s41597-022-01721-8)), HAM10000 ([_Nat Sci Data_ 2018](https://doi.org/10.1038/sdata.2018.161)), and Fitzpatrick17k ([_CVPR ISIC_ 2021](https://doi.org/10.1109/CVPRW53098.2021.00201)), for making their datasets publicly available. We would request the users of our datasets to also cite the original datasets in their work. The corresponding BibTeX entries for the original datasets are:

<details>

<summary>DermaMNIST:</summary>

```
@inproceedings{yang2021medmnist,
  title = {{MedMNIST} Classification Decathlon: A Lightweight {AutoML} Benchmark for Medical Image Analysis},
  author = {Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
  booktitle = {2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages = {191--195},
  year = {2021},
  organization = {IEEE},
  doi = {10.1109/ISBI48211.2021.9434062}
}

@article{yang2023medmnist,
  title = {{MedMNIST} v2 - A large-scale lightweight benchmark for {2D} and {3D} biomedical image classification},
  author = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal = {Scientific Data},
  volume = {10},
  number = {1},
  pages = {41},
  year = {2023},
  publisher = {Nature Publishing Group UK London},
  doi = {10.1038/s41597-022-01721-8}
}
```

</details>

<details>

<summary>HAM10000:</summary>

```
@article{tschandl2018ham10000,
  title = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific Data},
  volume = {5},
  number = {1},
  pages = {1--9},
  year = {2018},
  publisher = {Nature Publishing Group},
  doi = {10.1038/sdata.2018.161}
}
```

</details>

<details>

<summary>Fitzpatrick17k:</summary>

```
@inproceedings{groh2021evaluating,
  title = {Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the {Fitzpatrick 17k} Dataset},
  author = {Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {1820--1828},
  year = {2021}.
  doi = {10.1109/CVPRW53098.2021.00201}
}
```

</details>