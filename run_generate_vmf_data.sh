#!/bin/bash
#SBATCH --job-name=generate_vmf_data
#CommentSBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/aplesner/net_scratch/jobs/cusf/%j.out
#SBATCH --error=/itet-stor/aplesner/net_scratch/jobs/cusf/%j.err
#SBATCH --mem-per-cpu=200G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=tikgpu[04-10]

ETH_USERNAME=aplesner
PROJECT_NAME=code/cusf
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=cusf_env

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python3 generate_vmf_data.py

echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
