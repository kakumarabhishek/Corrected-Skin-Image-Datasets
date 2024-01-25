# DermaMNIST_Training

Directory containing the code for reproducing our benchmark experiment results.

## Directory Structure

* [`medmnist_corrected`](medmnist_corrected/): The new MedMNIST package, updated to include DermaMNIST-C and DermaMNIST-E at both `28 × 28` and `224 × 224` spatial resolutions.
* [`models.py`](models.py): Model definitions for ResNet-18 and ResNet-50 architectures. **Same as [the original MedMNIST implementation](https://github.com/MedMNIST/experiments/blob/8b0f553f95ea6b5f5517e49c539952cb21c79d89/MedMNIST2D/models.py).**
* [`train_and_eval_pytorch.py`](train_and_eval_pytorch.py): Training and evaluation script for MedMNIST datasets. **Same as [the original MedMNIST implementation](https://github.com/MedMNIST/experiments/blob/8b0f553f95ea6b5f5517e49c539952cb21c79d89/MedMNIST2D/train_and_eval_pytorch.py).**
* [`train_and_eval_pytorch_corrected.py`](train_and_eval_pytorch_corrected.py): Updated training and evaluation script to account for the new DermaMNIST-C and DermaMNIST-E datasets. This file is the same as [`train_and_eval_pytorch.py`](train_and_eval_pytorch.py) except for 3 changes:
    * L#7: `medmnist` changed to `medmnist_corrected`.
    * L#15: `medmnist` changed to `medmnist_corrected`.
    * L#54: Commented out.
* [`run_28.sh`](run_28.sh): Run script for all the benchmark experiments on DermaMNIST, DermaMNIST-C, and DermaMNIST-E datasets at `28 × 28` spatial resolution.
* [`run_224.sh`](run_224.sh): Run script for all the benchmark experiments on DermaMNIST, DermaMNIST-C, and DermaMNIST-E datasets at `224 × 224` spatial resolution.

## From [`medmnist`](https://github.com/MedMNIST/MedMNIST/tree/18a7564bc1fc3c68adbfeac7590d3949fe91467b) to [`medmnist_corrected`](medmnist_corrected/)

This code is built on `medmnist` v2.2.2. The only changes from `medmnist` to `medmnist_corrected` are as follows:
* [`__init__.py`](medmnist_corrected/__init__.py): Added L#6-L#7.
* [`__main__.py`](medmnist_corrected/__main__.py): Replaced all references of `medmnist` to `medmnist_corrected`.
* [`dataset.py`](medmnist_corrected/dataset.py): Added L#225-L#235.
* [`evaluator.py`](medmnist_corrected/evaluator.py): Changed `medmnist` to `medmnist_corrected` on L#9.
* [`info.py`](medmnist_corrected/info.py): Changed URL on L#25 and added L#466-L#565.

To use the DermaMNIST-C and DermaMNIST-E datasets, please copy the corresponding `.npz` files from OneDrive ([DermaMNIST-C](https://1sfu-my.sharepoint.com/:f:/g/personal/kabhishe_sfu_ca/EsEOH0QZwx5Ev779jIk5B2oBlq3IDLcMfm9zUlh80CFM6Q?e=yYEMq3), [DermaMNIST-E](https://1sfu-my.sharepoint.com/:f:/g/personal/kabhishe_sfu_ca/EhTDYQsbSNpBu3Njc9bD6NgB9xGPj8J4PtcqSrYAUl6gGw?e=PtXCpv)) into your local `~/.medmnist/` directory. This directory should have been created when you [installed `medmnist`](https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file#installation-and-requirements).