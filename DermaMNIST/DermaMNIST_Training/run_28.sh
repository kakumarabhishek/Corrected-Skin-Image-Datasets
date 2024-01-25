#!/bin/bash

# This is the training script for the 28x28 images.

LOG_DIR=<INSERT LOG DIRECTORY HERE>

for iter in {1..3}
do
        echo -e "\n\nRunning iteration ${iter}\n\n"
        python train_and_eval_pytorch.py --data_flag dermamnist --download --model_flag resnet18 > "$LOG_DIR"/original_28/r18_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_corrected_28 --model_flag resnet18 > "$LOG_DIR"/corrected_28/r18_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_extended_28 --model_flag resnet18 > "$LOG_DIR"/extended_28/r18_run_${iter}.log
        
        python train_and_eval_pytorch.py --data_flag dermamnist --download --model_flag resnet50 > "$LOG_DIR"/original_28/r50_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_corrected_28 --model_flag resnet50 > "$LOG_DIR"/corrected_28/r50_run_${iter}.log
        python train_and_eval_pytorch_corrected.py --data_flag dermamnist_extended_28 --model_flag resnet50 > "$LOG_DIR"/extended_28/r50_run_${iter}.log
done
