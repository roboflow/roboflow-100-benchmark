#!/bin/bash

#installs
# pip install -q transformers
# pip install -q pytorch-lightning
# pip install jupyter


input="../url_list.txt"


### Check if a directory does not exist ###
if [ ! -d "detr/" ] 
then
    echo "Cloning new repository..." 
    #git clone detr/ repo
    git clone https://github.com/facebookresearch/detr.git
fi

file="mAP_s.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi
touch "$file"

cp train_and_eval.py detr/
# cd detr/
# pip install -qr requirements.txt

while IFS= read -r line
do

    python3 ../parse_url.py -u $line
    str=`cat ../attributes.txt`
    
    workspace=$(echo $str | cut -d' ' -f 1)
    project=$(echo $str | cut -d' ' -f 2)
    version=$(echo $str | cut -d' ' -f 3)

    echo " -- " $workspace $project $version


    python3 ../set_up_dataset.py --workspace $workspace --project $project --version $version --download coco
    loc=`cat ../loc.txt`  # file with the dataset path stored
    echo $loc
 
    python3 detr/train_and_eval.py --loc $loc

    python3 parse_res.py --loc $loc

    echo "All the work has been completed. Removing the dataset folder..."
    rm -rf $loc

    echo "Onto the next one!";
    echo " ";

done < "$input"

echo "Done training all the datasets with YOLOS!"



