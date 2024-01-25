#!/bin/bash


# The original file directory is /localhome/kabhishe/WorkingDir/F17k_original/.
# To speed up data loading, we create a RAM disk and copy the original files there.
# mkdir -p /dev/shm/myramdisk
# cp -r /localhome/kabhishe/WorkingDir/F17k_original/ /dev/shm/myramdisk/


# Declare an array of all holdout sets.
# https://stackoverflow.com/a/8880633
declare -a holdoutsets=(
    "expert_select"
    "random_holdout" 
    "a12" 
    "a34" 
    "a56" 
    "dermaamin" 
    "br"
)

LOGS_DIR=<INSERT LOG DIRECTORY HERE>
OUTPUT_DIR=<INSERT OUTPUT DIRECTORY HERE>

# Iterate over all the holdout sets.
for HOLDOUT in "${holdoutsets[@]}"
do
    #  Iterate over multiple seeds.
    for SEED in {8887..8889}
    do
        echo "Experiment for holdout set $HOLDOUT and seed $SEED" > "$LOGS_DIR"/groh_"$HOLDOUT"_"$SEED".log
        # Run the training script.
        python train_SGD.py \
        --n_epochs 100 \
        --dev_mode full \
        --data_list_file ../Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv \
        --images_dir /dev/shm/myramdisk/F17k_original/ \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --holdout_set $HOLDOUT >> "$LOGS_DIR"/"$HOLDOUT"_"$SEED".log
    done
done