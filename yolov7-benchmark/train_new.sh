#!/bin/bash
set -euo pipefail

while getopts f:l: flag
do
    case "${flag}" in
        l) location=${OPTARG};;
    esac
done
# default values
location=${location:-$(pwd)/rf100}

file="mAP_v7.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi
touch "$file"


cd $(pwd)/yolov7-benchmark/
#git reset --hard commit c14ba0c297b3b5fc0374c917db798c88f9dd226c
#pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com -qr requirements.txt

dir=$(pwd)/runs/yolo-v7/train/rf-100/

rm -rf dir
mkdir dir


for dataset in $(ls $location)
do
    echo $dataset
done

# while IFS= read -r line
# do

#     python3 ../../parse_url.py -u $line
#     str=`cat attributes.txt`
    
#     project=$(echo $str | cut -d' ' -f 2)
#     version=$(echo $str | cut -d' ' -f 3)


#     ############ 1 DOWLOAD DATASET ####################
#     # set_up_dataset.py imports the dataset from the Universe and loads it on our machine
#     python3 ../../set_up_dataset.py --p $project --v $version --d yolov5
#     loc=`cat ../loc.txt`  # file with the dataset path stored
   
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




