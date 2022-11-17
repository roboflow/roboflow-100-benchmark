"""
Poor man script to convert the table from an old version of the paper. This was the only way since we didn't have access to the original data source from the table
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument("-f", "--file", type=Path)

args = parser.parse_args()
filepath = args.file
filepath = "./file.csv"

src_df = pd.read_csv(
    filepath,
    header=None,
    names=[
        "name",
        "classes",
        "labelling hours",
        "train",
        "valid",
        "test",
        "yolov5",
        "yolov7",
        "glip",
        "url",
    ],
)

src_df["dataset"] = src_df["url"].apply(lambda x: Path(x).parent.stem)
glip_df = src_df[["dataset", "glip"]].set_index("dataset")
src_df = src_df[["dataset", "labelling hours"]].set_index("dataset")


src_df.to_csv("./metadata/labeling_hours.csv")
glip_df.to_csv("./metadata/results_glip.csv")
