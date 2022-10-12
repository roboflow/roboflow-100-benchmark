"""
This script merge the model's results in one single csv fie
"""
import pandas as pd

yolov5_df = pd.read_csv(
    "/home/ubuntu/jacob/roboflow-100-benchmark/yolov5-benchmark/final_eval.txt",
    sep=" ",
    index_col=0,
)
yolov7_df = pd.read_csv(
    "/home/ubuntu/jacob/roboflow-100-benchmark/yolov7-benchmark/final_eval.txt",
    sep=" ",
    index_col=0,
)

df = yolov5_df.join(yolov7_df)
print(df)

df = df.reset_index()
df.columns = ["name", "yolov5", "yolov7"]
df.to_csv("./results.csv", index=None)
