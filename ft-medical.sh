#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ftt
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./out/%j.out
#SBATCH --error=./out/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane.ghebreziabiher@ku.ac.ae
#SBATCH --nodelist=gpu-10-3
#SBATCH --exclusive
 
module load miniconda/3
conda activate torch20
echo "Finally - out of queue" 
nvidia-smi

accelerate launch ft.py --run_id 240724111548
# CUDA_VISIBLE_DEVICES=0 python ft.py --run_id 240724111548