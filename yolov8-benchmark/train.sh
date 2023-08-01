#!/bin/bash
set -euo pipefail

dir=$(pwd)/runs/yolo-v8
echo $dir
datasets=$dir/rf100

if [ ! -d $datasets ] ; then
    $(pwd)/scripts/download_datasets.sh -l $datasets -f yolov5
fi

if [ ! -f "$dir/final_eval.txt" ] ; then
    touch "$dir/final_eval.txt"
fi


cd $(pwd)/yolov8-benchmark/

# fo rhttps://stackoverflow.com/questions/4011705/python-the-imagingft-c-module-is-not-installed
apt-get install -y libfreetype6-dev
# setting up yolov8 - specific version, 20.01.2023
pip install git+https://github.com/ultralytics/ultralytics.git@fix_shape_mismatch
# for AttributeError: partially initialized module ‘cv2’ has no attribute ‘gapi_wip_gst_GStreamerPipeline’ (most likely due to a circular import)
pip install --force --no-cache-dir opencv-contrib-python==4.5.5.62 
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt



for dataset in $(ls $datasets)
do
    dataset=$datasets/$dataset
    echo "Training on $dataset"
    if [ ! -d "$dataset/results" ] ;
    then
        yolo detect train data=$dataset/data.yaml model=yolov8s.pt epochs=1 batch=16

        yolo detect val data=$dataset/data.yaml  model=/workspace/yolov8-benchmark/runs/detect/train/weights/best.pt
        # yolov8 doesn't have a param for outdir LoL and save it in /runs without any reason
        cp -r /workspace/yolov8-benchmark/runs/detect/ $dataset/results 
        rm -rf /workspace/yolov8-benchmark/runs/detect/
        # python train.py --img 640 --batch 16 --epochs 100 --name $dataset/results --data $dataset/data.yaml  --weights ./yolov5s.pt
        # python val.py --data $dataset/data.yaml --img 640 --batch 16 --weights $dataset/results/weights/best.pt --name  $dataset --exist-ok --verbose |& tee $dataset/val_eval.txt 
        # python ../parse_eval.py -i $dataset/val_eval.txt -l $dataset -o $dir/final_eval.txt
    fi
done

echo "Done training all the datasets with YOLOv5!"




