#!/bin/bash

# This is the training script for the 224x224 images.
# The only differences between this and run_28.sh are:
# - the --resize flag is not used for the original dataset.
# - the data_flag is changed from dermamnist_{corrected,extended}_28 to dermamnist_{corrected,extended}_224

LOG_DIR=<INSERT LOG DIRECTORY HERE>

for iter in {1..3}
do
        echo -e "\n\nRunning iteration ${iter}\n\n"
        python train_and_eval_pytorch.py --data_flag dermamnist --download --model_flag resnet18 --resize > "$LOG_DIR"/original_224/r18_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_corrected_224 --model_flag resnet18 > "$LOG_DIR"/corrected_224/r18_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_extended_224 --model_flag resnet18 > "$LOG_DIR"/extended_224/r18_run_${iter}.log
        
        python train_and_eval_pytorch.py --data_flag dermamnist --download --model_flag resnet50 --resize > "$LOG_DIR"/original_224/r50_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_corrected_224 --model_flag resnet50 > "$LOG_DIR"/corrected_224/r50_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_extended_224 --model_flag resnet50 > "$LOG_DIR"/extended_224/r50_run_${iter}.log
done
