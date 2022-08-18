#!/bin/bash

#input="../../url_list.txt"
input="../../yolov7_url_list.txt"

# input="../test2.txt"


#check if yolov5 exists, if not 
### Check if a directory does not exist ###
# if [ ! -d "yolov5/" ] 
# then
#     echo "Cloning new repository." 
#     #git clone yolov5 repo
#     git clone https://github.com/ultralytics/yolov5.git
# fi




file="mAP_v7_a10g.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi
touch "$file"


cd yolov7/
#git reset --hard commit c14ba0c297b3b5fc0374c917db798c88f9dd226c
#pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com -qr requirements.txt


rm -rf runs/train/roboflow-100/



while IFS= read -r line
do

    # python3 ../../parse_url.py -u $line
    # str=`cat ../attributes.txt`
    echo "$line"
    # workspace=$(echo $str | cut -d' ' -f 1)
    # project=$(echo $str | cut -d' ' -f 2)
    # version=$(echo $str | cut -d' ' -f 3)


    ############ 1 DOWLOAD DATASET ####################
    # set_up_dataset.py imports the dataset from the Universe and loads it on our machine
    python3 ../../set_up_dataset.py --project $line
    loc=`cat ../loc.txt`  # file with the dataset path stored
    
    ############### 2 RUN TRAINING ################# yolov5/
    python3 train.py --img 640 --batch 8 --epochs 10000 --name roboflow-100/$line/1 --data $loc/data.yaml --weights yolov7.pt --cache # train the model on loaded dataset
 
    ############### 3 RUN EVALUATION ################# yolov5/
    #python3 val.py --weights runs/train/roboflow-100/$line/1/weights/best.pt --data $loc/data.yaml --img 640 --iou 0.65 --verbose |& tee val_eval.txt # evaluate the model
    
    python3 test.py --data $loc/data.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/roboflow-100/$line/1/weights/best.pt --name yolov7_640_val |& tee val_eval.txt # evaluate the model

    python3 ../parse_eval.py -l $loc # parse through evaluation 

    echo "All the work has been completed. Removing the dataset folder..."
    rm -rf $loc

    echo "Onto the next one!";
    echo " ";
    

done < "$input"

echo "Done training all the datasets with YOLOv5!"

#tmux new -s sess1
#tumux attach -t sess1
#run training in here


#leave tmux [ctrl + b + d ]


#you can always come back
#tmux attach -t sess1


