#!/bin/bash

input="../../url_list.txt"


#check if yolov5 exists, if not 
### Check if a directory does not exist ###
if [ ! -d "yolov5/" ] 
then
    echo "Cloning new repository." 
    #git clone yolov5 repo
    git clone https://github.com/ultralytics/yolov5.git
fi

cd yolov5/
pip install -qr requirements.txt
git reset --hard commit 2dd3db0050cd228e7a7ca3ff72ab7e3f34ea64d7

while IFS= read -r line
do

    python3 ../parse_url.py -u $line
    str=`cat ../attributes.txt`
    
    workspace=$(echo $str | cut -d' ' -f 1)
    project=$(echo $str | cut -d' ' -f 2)
    version=$(echo $str | cut -d' ' -f 3)


    ############ 1 DOWLOAD DATASET####################
    # set_up_dataset.py imports the dataset from the Universe and loads it on our machine
    python3 ../../set_up_dataset.py --workspace $workspace --project $project --version $version --download yolov5
    loc=`cat ../loc.txt`  # file with the dataset path stored
    
    ############### 2 RUN TRAINING################# yolov5/
    python3 train.py --img 416 --batch 16 --patience 40 --epochs 1 --name $workspace/$project/$version --data $loc/data.yaml --weights yolov5s.pt --cache # train the model on loaded dataset
 
    ############### 3 RUN EVALUATION#################yolov5/
    python3 val.py --weights runs/train/$workspace/$project/$version/weights/best.pt --data $loc/data.yaml --img 416 --iou 0.65 --verbose |& tee val_eval.txt # evaluate the model
    
    python3 ../parse_eval.py -l $loc # parse through evaluation 

    echo "All the work has been completed. Removing the dataset folder..."
    rm -rf $loc

    echo "Onto the next one!";
    echo " ";
    

done < "$input"

echo "Done training all the datasets with YOLOv5!"

