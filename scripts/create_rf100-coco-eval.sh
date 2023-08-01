#!/bin/bash
root=$1
dest=$(pwd)/rf100-coco-eval

mkdir -p $dest

for dataset in $(ls $root); do
    echo "$dataset"
    mkdir -p $dest/$dataset
    cp $root/$dataset/test/_annotations.coco.json $dest/$dataset/
done

echo "zipping ..."
zip -r rf100-coco-eval.zip rf100-coco-eval/