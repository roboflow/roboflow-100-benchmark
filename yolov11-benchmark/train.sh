#!/bin/bash
# set -euo pipefail

# Store the current directory
SCRIPT_DIR="$(pwd)"

# Set the datasets directory to ./rf100
datasets_dir="$SCRIPT_DIR/rf100"
echo "Datasets directory: $datasets_dir"

# Download datasets if not present
if [ -z "$(ls -A "$datasets_dir")" ]; then
    echo "Downloading datasets..."
    chmod +x "$SCRIPT_DIR/scripts/download_datasets.sh"
    "$SCRIPT_DIR/scripts/download_datasets.sh" -l "$datasets_dir" -f yolov11
fi

# Prepare the results directory
dir="$SCRIPT_DIR/runs/yolo-v11"
mkdir -p "$dir"
if [ ! -f "$dir/final_eval.txt" ]; then
    touch "$dir/final_eval.txt"
fi

cd "$SCRIPT_DIR/yolov11-benchmark/"

# Install dependencies if needed
echo "Installing dependencies..."
# Comment out apt-get commands if you don't have root permissions
# sudo apt-get update && sudo apt-get install -y libfreetype6-dev

pip install --user git+https://github.com/ultralytics/ultralytics.git
echo "Dependencies installed."

# Download the model file
echo "Downloading model file..."
wget -nc https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
echo "Model file downloaded."

# Set the model path
model_path="$SCRIPT_DIR/yolov11-benchmark/yolo11s.pt"

# Verify that the model file exists
if [ ! -f "$model_path" ]; then
    echo "Model file not found at $model_path. Exiting."
    exit 1
fi

# Get list of datasets
datasets=("$datasets_dir"/*)
num_datasets=${#datasets[@]}
echo "Number of datasets found: $num_datasets"

if [ "$num_datasets" -eq 0 ]; then
    echo "No datasets found in $datasets_dir. Exiting."
    exit 1
fi

num_gpus=8  # Number of GPUs

# Function to train on a single dataset
train_dataset() {
    local dataset="$1"

    # Find an available GPU using lock files
    while true; do
        for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
            lock_file="/tmp/gpu_lock_$gpu_id"
            exec {lock_fd}>$lock_file || continue
            if flock -n "$lock_fd"; then
                # Acquired lock for this GPU
                echo "Assigned GPU $gpu_id to dataset $dataset"
                # Start training
                dataset_name=$(basename "$dataset")
                results_dir="$dir/$dataset_name"

                if [ ! -f "$results_dir/train/weights/best.pt" ]; then
                    yolo detect train data="$dataset/data.yaml" model="$model_path" epochs=100 batch=-1 device="$gpu_id" project="$results_dir" name=train

                    yolo detect val data="$dataset/data.yaml" model="$results_dir/train/weights/best.pt" device="$gpu_id" project="$results_dir" name=val

                    python3 "$SCRIPT_DIR/yolov11-benchmark/parse_eval.py" -d "$dataset_name" -r "$results_dir/train" -o "$dir/final_eval.txt"
                else
                    echo "Results for $dataset already exist. Skipping training."
                fi

                # Release the lock
                flock -u "$lock_fd"
                exec {lock_fd}>&-
                rm -f "$lock_file"

                return 0
            else
                # Could not acquire lock; GPU is in use
                exec {lock_fd}>&-
            fi
        done
        # Wait before trying again
        sleep 5
    done
}

export -f train_dataset
export model_path
export dir
export SCRIPT_DIR

# Start training datasets with parallel execution
for dataset in "${datasets[@]}"; do
    train_dataset "$dataset" &
done

# Wait for all remaining jobs to finish
wait

echo "Done training all the datasets with YOLOv11!"
