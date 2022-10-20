"""
This script merge the model's results in one single csv fie
"""
import pandas as pd

yolov5_df = pd.read_csv(
    "./yolov5-benchmark/final_eval.txt",
    sep=" ",
    index_col=0,
    header=None,
    names=["dataset", "yolov5"],
)
yolov7_df = pd.read_csv(
    "./yolov7-benchmark/final_eval.txt",
    sep=" ",
    index_col=0,
    header=None,
    names=["dataset", "yolov7"],
)

df = yolov5_df.join(yolov7_df)
# let's add glip as well!
glip_df = pd.read_csv("./metadata/results_glip.csv", index_col=0)
df = df.join(glip_df)
df.to_csv("./results.csv")
