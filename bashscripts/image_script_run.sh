#!/bin/bash -l

#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch run_image_job.sh <dataset_name>"
    exit 1
fi

cd /home/hpc/iwbn/iwbn128h/project/thesis/project2

source .venv/bin/activate

echo "Dataset: $DATASET"
echo "Job ID: $SLURM_JOB_ID"

python src/experiments/run_image_experiment.py \
    --dataset "$DATASET" \
    --transform \
    --bottleneck-range 1 2 4 8 \
    --hpo-latent 3 \
    --seeds 42 43 44 \
    --results-path results