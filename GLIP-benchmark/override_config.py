import yaml
import json
import argparse

# This file contains helper functions to setup the GLIP repo for Evaluation


def write_yaml(data):
    filename = "dataset.yaml"

    file_data = dict(
        DATALOADER=dict(ASPECT_RATIO_GROUPING=False, SIZE_DIVISIBILITY=32),
        DATASETS=dict(
            GENERAL_COPY=16,
            OVERRIDE_CATEGORY=data,
            REGISTER=dict(
                test=dict(
                    ann_file="odinw/dataset/test/_annotations.coco.json",
                    img_dir="odinw/dataset/test",
                ),
                train=dict(
                    ann_file="odinw/dataset/train/_annotations.coco.json",
                    img_dir="odinw/dataset/train",
                ),
                val=dict(
                    ann_file="odinw/dataset/valid/_annotations.coco.json",
                    img_dir="odinw/dataset/valid",
                ),
            ),
            TEST='("val",)',
            TRAIN='("train",)',
        ),
        INPUT=dict(
            MAX_SIZE_TEST=1333,
            MAX_SIZE_TRAIN=1333,
            MIN_SIZE_TEST=800,
            MIN_SIZE_TRAIN=800,
        ),
        MODEL=dict(
            ATSS=dict(NUM_CLASSES=8),
            DYHEAD=dict(NUM_CLASSES=8),
            FCOS=dict(NUM_CLASSES=8),
            ROI_BOX_HEAD=dict(NUM_CLASSES=8),
        ),
        SOLVER=dict(CHECKPOINT_PERIOD=100, MAX_EPOCH=12, WARMUP_ITERS=0),
        TEST=dict(IMS_PER_BATCH=8),
    )

    with open(filename, "w") as outfile:
        yaml.dump(file_data, outfile, default_flow_style=False)


def delete_first_entry():
    print("Deleting first JSON entry...")
    folders = ["dataset/test/", "dataset/train/", "dataset/valid/"]

    for folder in folders:
        with open(folder + "_annotations.coco.json") as f:
            data = json.load(f)

        for entry in list(data["categories"]):
            if entry["id"] == 0:
                data["categories"].remove(entry)

        with open(folder + "_annotations.coco.json", "w") as f:
            json.dump(data, f)


def gen_data():
    f = open("dataset/test/_annotations.coco.json")
    data = json.load(f)

    data_list = []
    g = open("../custom_prompts.json")
    custom_data = json.load(g)

    for i in data["categories"]:
        # data_list.append(i) <-- use this line for normal evaluation
        dataset_name = i["supercategory"]
        break

    for dataset in custom_data:
        if dataset["dataset_name"] == dataset_name:
            data_list = dataset["classes"]

    if len(data_list) == 0:
        data_list = data["categories"]

    return data_list


def main():

    delete_first_entry()  # remove the zeroth entry in JSON file
    data = gen_data()  # generate the dictionary data to pass to yaml file
    write_yaml(str(data))  # generate yaml file for evaluation


if __name__ == "__main__":
    main()
