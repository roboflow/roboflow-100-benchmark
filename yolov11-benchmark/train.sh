#!/bin/bash
set -euo pipefail

dir=$(pwd)/runs/yolo-v11
echo $dir
datasets_dir=$dir/rf100

if [ ! -d "$datasets_dir" ] ; then
    "$(pwd)/scripts/download_datasets.sh" -l "$datasets_dir" -f yolov11
fi

if [ ! -f "$dir/final_eval.txt" ] ; then
    touch "$dir/final_eval.txt"
fi

cd "$(pwd)/yolov11-benchmark/"

# Install dependencies if needed
apt-get install -y libfreetype6-dev
pip install git+https://github.com/ultralytics/ultralytics.git@7a6c76d16c01f3e4ce9ed20eedc6ed27421b3268
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

# Get list of datasets
datasets=("$datasets_dir"/*)
num_datasets=${#datasets[@]}
num_gpus=8

# Function to train on a single dataset
train_dataset() {
    local dataset=$1
    local gpu_id=$2
    echo "Training on $dataset using GPU $gpu_id"

    if [ ! -d "$dataset/results" ] ; then
        yolo detect train data="$dataset/data.yaml" model=yolov11s.pt epochs=100 batch=-1 device="$gpu_id" project="$dataset" name=train

        yolo detect val data="$dataset/data.yaml" model="$dataset/train/weights/best.pt" device="$gpu_id" project="$dataset" name=val
    fi
}

# Keep track of running jobs
pids=()
for ((i=0; i<num_datasets; i++)); do
    dataset=${datasets[$i]}
    gpu_id=$((i % num_gpus))
    train_dataset "$dataset" "$gpu_id" &
    pids+=($!)

    # When number of background jobs reaches num_gpus, wait for any to finish
    if (( ${#pids[@]} >= num_gpus )); then
        wait -n
        # Clean up finished jobs from the pids array
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                pids=("${pids[@]/$pid/}")
            fi
        done
    fi
done

# Wait for all remaining jobs to finish
wait

echo "Done training all the datasets with YOLOv11!"
