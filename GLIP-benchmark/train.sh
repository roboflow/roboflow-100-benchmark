#!/bin/bash
set -euo pipefail

dir=$(pwd)/runs/glip
echo $dir
datasets=$dir/rf100

if [ ! -d $datasets ] ; then
    $(pwd)/scripts/download_datasets.sh -l $datasets -f coco
fi

if [ ! -f "$dir/final_eval.txt" ] ; then
    touch "$dir/final_eval.txt"
fi


cd $(pwd)/GLIP-benchmark/

if [ ! -f "$dir/glip_tiny_model_o365_goldg_cc_sbu.pth" ] ; then
    wget -P $dir https://github.com/microsoft/GLIP/blob/main/configs/pretrain/glip_Swin_T_O365_GoldG.yaml
fi

cd GLIP/

python setup.py build develop --user

mkdir -p ./DATASET/coco/annotations

if [ -L "./DATASET/coco/val2017/test" ] ; then
    rm ./DATASET/coco/val2017/test
    rm ./DATASET/coco/annotations/instances_val2017.json
fi

for dataset in $(ls $datasets)
do
    dataset=$datasets/$dataset
    echo "Training on $dataset"
    if [ ! -d "$dataset/results" ] ;
    then
        ln -s $dataset/test ./DATASET/coco/val2017
        ln -s $dataset/test/_annotations.coco.json ./DATASET/coco/annotations/instances_val2017.json

        PYTHONPATH=""
        # interesting, we need to append this module to the PYTHONPATH
        export PYTHONPATH="${PYTHONPATH}:${PWD}/"  && python tools/test_grounding_net.py --config-file ../glip_Swin_T_O365_GoldG.yaml --weight $dir/glip_tiny_model_o365_goldg_cc_sbu.pth \
            TEST.IMS_PER_BATCH 1 \
            MODEL.DYHEAD.SCORE_AGG "MEAN" \
            TEST.EVAL_TASK detection \
            MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
            OUTPUT_DIR $dataset/results
        rm ./DATASET/coco/val2017
        rm ./DATASET/coco/annotations/instances_val2017.json
    fi
done

echo "Done evaluating all the datasets with GLIP!"




