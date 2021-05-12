#!/bin/bash
#SBATCH --job-name=g50_t500_n0
#SBATCH --output=g50_t500_n0.out
#SBATCH --error=g50_t500_n0_error.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch torchvision

cd $HOME/GitRepos/GOKU/

python3 goku_train.py \
        --model double_pendulum \
        --grounding \
        --grounding-epoch 0 \
        --data-path $HOME/scratch/data/setup_g50_t500 \
        --run-id g50_t500_n0 \
        --save-path $HOME/scratch/grounding_project/results/ \
        --checkpoints-dir $HOME/scratch/grounding_project/results/

python3 goku_train.py \
    --model double_pendulum \
    --grounding \
    --grounding-epoch 0 \
    --data-path /hdd/data/grounding_exp/Setup_g50_t500 \
    --run-id g50_t500_n0 \
    --save-path $HOME/scratch/grounding_project/results/ 