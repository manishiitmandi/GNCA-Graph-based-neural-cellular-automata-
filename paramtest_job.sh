#!/bin/bash
#SBATCH --job-name=mednca_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu              # Use the appropriate partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00              # Set max runtime (24 hours)

timestamp() {
    date +"%T"
}

echo "=== Job started at $(timestamp) ===" >> checkpoints.log

# Load Python module
module load python/3.9
echo "[$(timestamp)] Loaded Python module" >> checkpoints.log

# Activate virtual environment
source /scratch/arnavk.scee.iitmandi/mednca/.venv/bin/activate
echo "[$(timestamp)] Activated .venv" >> checkpoints.log

# Navigate to project directory
cd /scratch/arnavk.scee.iitmandi/mednca
echo "[$(timestamp)] Changed to project directory" >> checkpoints.log

# Check PyTorch installation
echo "[$(timestamp)] PyTorch version check:" >> checkpoints.log
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" >> checkpoints.log 2>&1

# Check for corrupted images before training
echo "[$(timestamp)] Checking for corrupted images..." >> checkpoints.log
python -c "
import os
import glob
from PIL import Image

corrupted_count = 0
total_count = 0

# Check common image extensions
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
    for img_path in glob.glob(f'data/**/{ext}', recursive=True):
        total_count += 1
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f'Corrupted: {img_path}')
            corrupted_count += 1

print(f'Found {corrupted_count} corrupted images out of {total_count} total images')
" >> checkpoints.log 2>&1

# Function to run training with error handling
run_training() {
    local attempt=1
    local max_attempts=3
    
    while [ $attempt -le $max_attempts ]; do
        echo "[$(timestamp)] Starting train.py (attempt $attempt/$max_attempts)" >> checkpoints.log

        # Log system resources before training
        echo "[$(timestamp)] System resources before training:" >> checkpoints.log
        echo "GPU status:" >> checkpoints.log
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader >> checkpoints.log 2>/dev/null || echo "nvidia-smi failed" >> checkpoints.log
        
        echo "Memory usage:" >> checkpoints.log
        free -h >> checkpoints.log
        
        echo "CPU usage:" >> checkpoints.log
        top -b -n 1 | head -5 >> checkpoints.log

        # Run training script with timeout
        timeout 23h python train.py >> train_output.log 2>> train_error.log
        exit_code=$?

        echo "[$(timestamp)] train.py finished with exit code: $exit_code" >> checkpoints.log

        # Check exit status
        if [ $exit_code -eq 0 ]; then
            echo "[$(timestamp)] Training completed successfully" >> checkpoints.log
            return 0
        elif [ $exit_code -eq 124 ]; then
            echo "[$(timestamp)] Training timed out after 23 hours" >> checkpoints.log
            return 0  # Timeout is expected, not an error
        else
            echo "[$(timestamp)] Training failed with exit code: $exit_code" >> checkpoints.log
            
            # Log last few lines of error for debugging
            echo "Last 10 lines of train_error.log:" >> checkpoints.log
            tail -10 train_error.log >> checkpoints.log 2>/dev/null || echo "No error log found" >> checkpoints.log
            
            if [ $attempt -lt $max_attempts ]; then
                echo "[$(timestamp)] Retrying in 60 seconds..." >> checkpoints.log
                sleep 60
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "[$(timestamp)] All training attempts failed" >> checkpoints.log
    return 1
}

# Run training once (remove infinite loop for HPC best practices)
run_training

# Final system status
echo "[$(timestamp)] Final system status:" >> checkpoints.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> checkpoints.log 2>/dev/null || echo "nvidia-smi failed" >> checkpoints.log

echo "=== Job ended at $(timestamp) ===" >> checkpoints.log