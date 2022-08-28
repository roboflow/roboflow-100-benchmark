#!/bin/bash
set -euo pipefail


#git reset --hard commit c14ba0c297b3b5fc0374c917db798c88f9dd226c
#pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com -qr requirements.txt

dir=$(pwd)/runs/yolo-v7/train
data=$dir/rf100

rm -rf $dir
mkdir -p $dir

$(pwd)/scripts/download_datasets.sh -l $data -f yolov5

cd $(pwd)/yolov7-benchmark/

file="mAP_v7.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi
touch "$file"

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

for dataset in $(ls $location)
do
    echo "Training on $dataset"
    python train.py --img 640 --batch 8 --epochs 100 --name $dataset --data $dataset/data.yaml  --weights ./yolov7.pt --cache
done

#     ############### 2 RUN TRAINING ################# yolov5/
#     python3 train.py --img 640 --batch 8 --epochs 10000 --name roboflow-100/$project/$version --data $loc/data.yaml --weights yolov7.pt --cache # train the model on loaded dataset
#     echo "here"
#     ############### 3 RUN EVALUATION ################# yolov5/    
#     python3 test.py --data $loc/data.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/roboflow-100/$project/$version/weights/best.pt --name yolov7_640_val |& tee val_eval.txt # evaluate the model

#     python3 ../parse_eval.py -l $loc # parse through evaluation 

#     echo "All the work has been completed. Removing the dataset folder..."
#     rm -rf $loc

#     echo "Onto the next one!";
#     echo " ";
    

# done < "$input"

echo "Done training all the datasets with YOLOv7!"




