# VLAb Examples

This directory contains example scripts demonstrating how to use VLAb on the cluster.

For general usage and installation instructions, see the main [README.md](../README.md).

## Structure

- `scripts/` - Example training scripts (SLURM scripts for cluster training)
- `all_datasets_relative.txt` - List of pretraining datasets from Community_Datasets_v1 and Community_Datasets_v2 that are stored locally

## Training Scripts

The `scripts/` directory contains SLURM scripts for multi-GPU training on compute clusters:

### Fresh Training (`reproduce_smolvla.slurm`)

This script starts training from scratch with a new output directory. It uses all datasets from `all_datasets_relative.txt` and configures a 2-GPU training setup with optimized settings.

**Key Features:**
- 2 GPUs with global batch size of 16 (8 per GPU)
- Mixed precision training (AMP) enabled
- Parallel data loading with 4 workers per GPU
- 100,000 training steps
- Automatic output directory creation with timestamp

**Usage:**
```bash
sbatch VLAb/examples/scripts/reproduce_smolvla.slurm
```

### Resume Training (`reproduce_smolvla_resume.slurm`)

This script resumes training from a previous checkpoint. It loads the configuration from the checkpoint and continues training with the same settings as fresh training.

**Usage:**
```bash
# First, edit the script to set the checkpoint path:
# Update line 41: export RESUME_FROM_CHECKPOINT="/path/to/your/checkpoint"

# Then submit:
sbatch VLAb/examples/scripts/reproduce_smolvla_resume.slurm
```

## Dataset List Generation

You can generate a dataset list from local lerobot datasets using the following script:

```bash
# Generate dataset list dynamically
DATASET_LIST=$(python -c "
import os
datasets = []
for root, dirs, files in os.walk('/path/to/local/datasets'):
    if 'data.parquet' in files:
        rel_path = os.path.relpath(root, '/path/to/local/datasets')
        datasets.append(rel_path.replace('/', '/'))
print(','.join(datasets))
")

# Train with generated list
accelerate launch --config_file VLAb/accelerate_configs/multi_gpu.yaml \
    VLAb/src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --dataset.repo_id="$DATASET_LIST" \
    --dataset.root="/path/to/local/datasets" \
    --policy.repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
    --dataset.video_backend=pyav \
    --output_dir="./outputs/multi_gpu_training" \
    --batch_size=8 \
    --steps=100000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```

**Note:** Update `/path/to/local/datasets` to point to your actual dataset directory. This script will automatically discover all datasets that contain a `data.parquet` file.

## Quick Start

For basic usage, see the main [README.md](../README.md).

For cluster training, use the SLURM scripts in this directory. Make sure to:
1. Update all paths to match your environment (especially conda path and output directories)
2. Adjust resource requests if needed (currently configured for 2 GPUs, 44 CPUs, 2 hours)
3. Configure your dataset paths - scripts use `all_datasets_relative.txt` for local datasets
4. Set up your WandB credentials if using logging (scripts have WandB enabled by default)
5. For resume training, update the `RESUME_FROM_CHECKPOINT` variable in the resume script

**Note:** The scripts are configured for the hopper-prod partition. Adjust the partition name if using a different cluster.
