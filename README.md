# Roboflow 100 📸: A Rich, Multi-Task Object Detection Benchmark

This repository implements the Roboflow 100 benchmark developed by [Roboflow](https://roboflow.com/). It contains code to download the dataset and reproduce
mAP values for YOLOv5 and YOLOv7 Fine-Tuning and GLIP Evaluation on 100 of Roboflow Universe
datasets.

## RF100

`RF100` contains the following datasets, carefully chosen from more than 90'000 datasets hosted on our [universe hub](https://universe.roboflow.com/).

**TODO** create a table

We provide a [notebook to help you using RF100 with PyTorch](/notebooks/roboflow-100-pytorch.ipynb)

## Getting Started

First, clone this repo and go inside it.

```bash
git clone https://github.com/roboflow-ai/roboflow-100-benchmark.git
cd roboflow-100-benchmark
git submodule update --init --recursive
```

You will need an API key. `RF100` can be accessed with any key from Roboflow, head over [our doc](https://docs.roboflow.com/rest-api.') to learn how to get one.

Then, export the key to your current shell

```bash
export ROBOFLOW_API_KEY=<YOUR_API_KEY>
```

**Note**: The datasets are taken from `datasets_links.txt`, you can modify that file to add/remove datasets.

### Docker

The easiest and faster way to download `RF100` is using [docker](https://docs.docker.com/engine/install/) and our [Dockerfile](Dockerfile.rf100.download).

**NOTE** Be sure to do the [post process steps](https://docs.docker.com/engine/install/linux-postinstall/) after you installed docker, we will read and write to shared volumes so your user should also be in the docker group.

If you have an NVIDIA GPU, be sure to also install [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

First, build the container

```bash
docker build -t rf100-download -f Dockerfile.rf100.download .
```

Be sure to have the `ROBOFLOW_API_KEY` in your env, then run it

```bash
docker run --rm -it \
    -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
    -v $(pwd)/rf100:/workspace/rf100 \
    rf100-download
```

Internally, `RF100` will downloaded to `/app/rf100`. You can also specify the format with the `-f` flag, by default `coco` is used.

```bash
docker run --rm -it \
    -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
    -v ${PWD}/rf100:/workspace/rf100 \
    rf100-download -f yolov5
```

### Local Env

To download `RF100` in your local enviroment (python `>=3.6`), you need to install roboflow

```bash
pip install roboflow
```

Then,

```bash
chmod 770 ./scripts/download_datasets.sh
./scripts/download_datasets.sh
./scripts/download_datasets.sh -f yolov5 $ change format
./scripts/download_datasets.sh -l <path_to_my_location> change download location
```

### Formats

Supported formats are

- `coco`
- `yolov5` (used by YOLOv7 as well)

## Reproduce Results

We will use docker to ensure the same enviroment is used.

First, build the container

```
docker build -t rf100-benchmark -f Dockerfile.rf100.benchmark .
```

Then, follow the guide for each model.

All results are stored inside `./runs`.

### [YOLOv5](https://github.com/ultralytics/yolov5) Fine-Tuning

**Note**, we will map the current folder to the container file system to persist data

```bash
nvidia-docker run --gpus all --rm -it --ipc host --network host --shm-size 64g \
    -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
    -v /etc/group:/etc/group:ro \
    -u "$(id -u):$(id -g)" \
    -v ${PWD}/runs:/workspace/runs \
    -v ${PWD}/datasets_links_640.txt:/workspace/datasets_links_640.txt \
    rf100-benchmark ./yolov5-benchmark/train.sh
```

### [YOLOv7](https://github.com/WongKinYiu/yolov7) Fine-Tuning

**Note**, we will map the current folder to the container file system to persist data

```bash
nvidia-docker run --gpus all --rm -d --ipc host --network host --shm-size 64g \
    -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
    -v /etc/group:/etc/group:ro \
    -u "$(id -u):$(id -g)" \
    -v ${PWD}/runs:/workspace/runs \
    -v ${PWD}/datasets_links_640.txt:/workspace/datasets_links_640.txt \
    rf100-benchmark ./yolov7-benchmark/train.sh
```

### [GLIP](https://github.com/microsoft/GLIP)

```bash
nvidia-docker run --gpus all --rm -it --ipc host --network host --shm-size 64g \
    -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
    -v /etc/group:/etc/group:ro \
    -u "$(id -u):$(id -g)" \
    -v ${PWD}/runs:/workspace/runs \
    -v ${PWD}/datasets_links_640.txt:/workspace/datasets_links_640.txt \
    rf100-benchmark ./GLIP-benchmark/train.sh
```
