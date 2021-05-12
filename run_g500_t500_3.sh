#!/bin/bash
#SBATCH --job-name=Spurious_sweep
#SBATCH --output=Spurious_sweep.out
#SBATCH --error=Spurious_sweep_error.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=50Gb

# Load Modules and environements
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch torchvision

cd $HOME/GitRepos/Spurious-Learning/

python3 train_models.py \
        --data-path $HOME/scratch/data/ \
        --save-path $HOME/scratch/spurious_project/