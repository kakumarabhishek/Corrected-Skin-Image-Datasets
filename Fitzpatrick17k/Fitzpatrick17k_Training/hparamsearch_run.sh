#!/bin/bash
#SBATCH -J f17k_hparam       # Name that will show up in squeue
#SBATCH --gres=gpu:1         # Request 1 GPUs
#SBATCH --time=7-00:00       # Max job time is 7 days
#SBATCH --cpus-per-task=16   # Number of CPU cores per task
#SBATCH --mem=32G            # Max memory (CPU)
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --error=%N-%j.err    # Terminal output to file named (hostname)-(jobid).err
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --mail-user=kabhishe@sfu.ca
#SBATCH --mail-type=all
#SBATCH --nodelist=cs-venus-09
#SBATCH --qos=overcap
#SBATCH --propagate=STACK

ulimit -Su unlimited
ulimit -Sv unlimited

source /home/kabhishe/.bashrc
conda activate <CONDA ENV NAME>

set -euf -o pipefail    # https://sipb.mit.edu/doc/safe-shell/

cd <PATH TO CODE DIRECTORY>


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

# Iterate over all the epoch values.
for EPOCHS in {20,50,100,200}
do
    # Iterate over both the optimizers.
    for OPTIMIZER in {"SGD","Adam"}
    do
        # Iterate over all the learning rates.
        for LR in {0.01,0.001,0.0001}
        do
            # Iterate over all the holdout sets.
            for HOLDOUT in "${holdoutsets[@]}"
            do
                #  Iterate over multiple seeds.
                for SEED in {8887..8889}
                do
                    # Create a log file for the current experiment in a new directory.
                    mkdir -p "$LOGS_DIR"/"$EPOCHS"_"$OPTIMIZER"_"$LR"
                    echo "Experiment for holdout set $HOLDOUT and seed $SEED" > "$LOGS_DIR"/"$EPOCHS"_"$OPTIMIZER"_"$LR"/"$HOLDOUT"_"$SEED".log
                    # Run the training script.
                    python hparamsearch_train.py \
                    --n_epochs $EPOCHS \
                    --optimizer $OPTIMIZER \
                    --base_lr $LR \
                    --dev_mode full \
                    --data_list_file ./SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv \
                    --images_dir /dev/shm/myramdisk/F17k_original/ \
                    --output_dir $OUTPUT_DIR \
                    --seed $SEED \
                    --holdout_set $HOLDOUT >> "$LOGS_DIR"/"$EPOCHS"_"$OPTIMIZER"_"$LR"/"$HOLDOUT"_"$SEED".log
                done
            done
        done
    done
done
