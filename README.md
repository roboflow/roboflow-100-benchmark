# Roboflow 100: A Rich, Multi-Task Object Detection Benchmark

This repository implements the Roboflow 100 benchmark. It contains code to reproduce 
mAP values for YOLOv5 and YOLOv7 Fine-Tuning and GLIP Evaluation on 100 of Roboflow Universe
datasets. 

## Reproduce YOLOv5 Fine-Tuning
Execute the following commands to reproduce YOLOv5 Fine Tuning values. 

1) ```cd yolov5-benchmark```
2) ```bash train.sh```

The output will be stored in the ```mAP_v5.txt``` file. 

## Reproduce YOLOv7 Fine-Tuning
Execute the following commands to reproduce YOLOv7 Fine Tuning values. 

1) ```cd yolov7-benchmark```
2) ```bash train.sh```

The output will be stored in the ```mAP_v7.txt``` file. 

## Reproduce GLIP Evaluation
Execute the following commands to reproduce GLIP Evaluation values. 

1) ```cd GLIP-benchmark/GLIP/```
2) ```bash GLIP_eval.sh ```

The output will be stored in the ```mAP_output.txt``` file.

