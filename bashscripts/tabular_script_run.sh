#!/bin/bash -l

#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch run_tabular_job.sh <dataset_name>"
    exit 1
fi

cd /home/hpc/iwbn/iwbn128h/project/thesis/project2

source .venv/bin/activate

echo "======================================"
echo "Dataset: $DATASET"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "======================================"

python src/experiments/run_tabular_experiment.py \
    --dataset "$DATASET" \
    --bottleneck-range 1 2 4 8 16 32 \
    --hpo-latent 8 \
    --seeds 42 43 44 \
    --results-path results