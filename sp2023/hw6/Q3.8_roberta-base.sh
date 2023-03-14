#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"

source venv/bin/activate 
module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies

# runs your code
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 1e-4 --num_epochs 5
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 1e-4 --num_epochs 7
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 1e-4 --num_epochs 9

srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 5e-4 --num_epochs 5
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 5e-4 --num_epochs 7
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 32 --lr 5e-4 --num_epochs 9

srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 64 --lr 1e-3 --num_epochs 5
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 64 --lr 5e-4 --num_epochs 7
srun python classification.py --experiment roberta-base_grid_search --device cuda --model roberta-base --batch_size 64 --lr 5e-4 --num_epochs 9