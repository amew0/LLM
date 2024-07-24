#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ftt
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./out/%j.out
#SBATCH --error=./out/%j.err
# SBATCH --nodelist=gpu-11-3
 
module load miniconda/3
conda activate torch20
echo "Finally - out of queue" 
nvidia-smi

# CUDA_DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
accelerate launch ft.py