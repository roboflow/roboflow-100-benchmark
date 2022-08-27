# Roboflow 100: A Rich, Multi-Task Object Detection Benchmark

This repository implements the Roboflow 100 benchmark developed by [Roboflow](https://roboflow.com/). It contains code to download the dataset and reproduce 
mAP values for YOLOv5 and YOLOv7 Fine-Tuning and GLIP Evaluation on 100 of Roboflow Universe
datasets. 

## RF100

`RF100` contains the following datasets, carefully chosen from more than 90'000 datasets hosted on our [universe hub](https://universe.roboflow.com/).

**TODO** create a table

## Getting Started

First, clone this repo and go inside it.


```bash
git clone https://github.com/roboflow-ai/roboflow-100-benchmark.git
cd roboflow-100-benchmark
```

You will need an API key. `RF100` can be accessed with any key from Roboflow, head over [our doc](https://docs.roboflow.com/rest-api.') to learn how to get one.

Then, export the key to your current shell

```bash
export ROBOFLOW_API_KEY=<YOUR_API_KEY>
```

**Note**: The datasets are taken from `datasets_links.txt`, you can modify that file to add/remove datasets.

### Docker

The easiest and faster way to download `RF100` is using [docker](https://docs.docker.com/engine/install/) and our [Dockerfile](Dockerfile.rf100.download).

First, build the container

```bash
docker build -t rf100-download -f Dockerfile.rf100.download .
```

Be sure to have the `ROBOFLOW_API_KEY` in your env, then run it

```bash
docker run --rm -it -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -v $(pwd)/rf100:/app/rf100 rf100-download
```

Internally, `RF100` will downloaded to `/app/rf100`. You can also specify the format with the `-f` flag, by default `coco` is used.

```bash
 docker run --rm -it -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -v $(pwd)/rf100:/app/rf100 rf100-download -f yolov5
```


### Local Env

To download `RF100` in your local enviroment (python `>=3.6`), you need to install roboflow

```bash
pip install roboflow
```

Then,

```bash
chmod +x ./scripts/download_datasets.sh
./scripts/download_datasets.sh 
./scripts/download_datasets.sh -f yolov5 $ change format
./scripts/download_datasets.sh -l <path_to_my_location> change download location
```

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

