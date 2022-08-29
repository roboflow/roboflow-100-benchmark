#!/bin/bash
set -euo pipefail


#git reset --hard commit c14ba0c297b3b5fc0374c917db798c88f9dd226c
#pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com -qr requirements.txt

dir=$(pwd)/runs/yolo-v5
echo $dir
datasets=$dir/rf100

if [ ! -d $datasets ] ; then
    $(pwd)/scripts/download_datasets.sh -l $datasets -f yolov5
fi

if [ ! -f "$dir/final_eval.txt" ] ; then
    touch "$dir/final_eval.txt"
fi


cd $(pwd)/yolov5-benchmark/

if [ ! -f "yolov5s.pt" ] ; then
    wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
fi

cd yolov5/

for dataset in $(ls $datasets)
do
    dataset=$datasets/$dataset
    echo "Training on $dataset"
    python train.py --img 640 --batch 16 --epochs 100 --name $dataset/results --data $dataset/data.yaml  --weights ./yolov5s.pt
    python val.py --data $dataset/data.yaml --img 640 --batch 16 --weights $dataset/results/weights/best.pt --name  $dataset --exist-ok --verbose |& tee $dataset/val_eval.txt 
    python ../parse_eval.py -i $dataset/val_eval.txt -l $dataset -o $dir/final_eval.txt
done

echo "Done training all the datasets with YOLOv5!"




