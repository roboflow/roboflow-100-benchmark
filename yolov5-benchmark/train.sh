#!/bin/bash
set -euo pipefail
input="$(pwd)/datasets.txt"


#check if yolov5 exists, if not 
### Check if a directory does not exist ###
if [ ! -d "yolov5/" ] 
then
    echo "Cloning new repository." 
    #git clone yolov5 repo
    git clone https://github.com/ultralytics/yolov5.git
fi

file="mAP_v5.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi
touch "$file"


cd yolov5/
pip install -qr requirements.txt
git reset --hard commit 2dd3db0050cd228e7a7ca3ff72ab7e3f34ea64d7

rm -rf runs/train/roboflow-100/

while IFS= read -r line
do

    python3 ../../parse_url.py -u $line
    str=`cat attributes.txt`
    
    project=$(echo $str | cut -d' ' -f 2)
    version=$(echo $str | cut -d' ' -f 3)

    echo "$project"
    ############ 1 DOWLOAD DATASET ####################
    # set_up_dataset.py imports the dataset from the Universe and loads it on our machine
    python3 ../../set_up_dataset.py --p $project --v $version --d yolov5
    loc=`cat ../loc.txt`  # file with the dataset path stored
    echo "$loc"
    ############### 2 RUN TRAINING ################# 
    python3 train.py --img 640 --batch 16 --patience 40 --epochs 10000 --name roboflow-100/$project/$version --data $loc/data.yaml --weights yolov5s.pt --cache # train the model on loaded dataset
 
    ############### 3 RUN EVALUATION #################
    python3 val.py --weights runs/train/roboflow-100/$project/$version/weights/best.pt --data $loc/data.yaml --img 640 --iou 0.65 --verbose |& tee val_eval.txt # evaluate the model

    python3 ../parse_eval.py -l $loc # parse through evaluation 

    echo "All the work has been completed. Removing the dataset folder..."
    rm -rf $loc

    echo "Onto the next one!";
    echo " ";
    

done < "$input"

echo "Done training all the datasets with YOLOv5!"

