# Fitzpatrick17k_Training

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Directory containing the code for reproducing our benchmark experiment results.

## Directory Structure

* Model training and evaluation code:

    * [`run.sh`](run.sh): Run script for all the benchmark experiments on Fitzpatrick17k-C.
    * [`train.py`](train.py): The training and evaluation script, adopted from [the original file](https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py), with modifications as listed in the paper and detailed below.
    * [`utils.py`](utils.py): Utility functions for various training-related tasks.

* Hyperparameter search code:

    * [`hparamsearch_train.py`](hparamsearch_train.py): The script for hyperparameter search over the choices of number of training epochs, optimizer, and learning rate, over all the seven holdout sets, each repeated 3 times.
    * [`hparamsearch_run.sh`](hparamsearch_run.sh): SLURM script for the hyperparameter search.
    * [`hparamsearch_logs.csv`](hparamsearch_logs.csv): The log file containing the results of the hyperparameter search.
    * [`hparamsearch_analyze_logs.py`](hparamsearch_analyze_logs.py): The script for analyzing the results of the hyperparameter search and returning the best hyperparameters and their corresponding results.

## Modifications to [`train.py`](train.py)

Our training code is built upon [the original `train.py` by Groh et al.](https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py) with the following modifications:
* We use **separate partitions** for **validation** (i.e., picking the best performing model across training epochs based on the highest validation accuracy ) and for **testing** (i.e., for reporting the final Fitzpatrick17k-C benchmarks).
* We change the **optimizer** to SGD (from Adam in the original code), and train for 100 epochs with a learning rate of 0.001, weight decay of 0.001, and momentum of 0.9, and mini-batch size of 32.
* We use [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) for mixed-precision training.
    * This can easily be disabled by setting `mixed_precision` to `no` on L#22.

Additional edits to the code that do not affect the functionality:
* We use [`comet_ml`](https://www.comet.ml/) for experiment management and tracking.
* The [`flatten()` function](https://github.com/mattgroh/fitzpatrick17k/blob/26d50745348f82a76f872ed7924361d1dccd629e/train.py#L22) has been moved to [`utils.py`](utils.py) with other utility functions for better organization.