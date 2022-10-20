"""
This script create the result table in the paper.

1) get all links from `dataset_links.txt`
2) get all the stats from RF100 (name, number of images in train/val/test) + labeling hours
3) merge the stats with the result dataframe (created by `merge_final_eval.py`)
4) print the latex code for the table
"""
from pathlib import Path
from textwrap import indent
from unicodedata import name
import pandas as pd

RF100_BASE_URL = "https://app.roboflow.com/roboflow-100"


def get_rf100_version_and_links(root: Path) -> pd.DataFrame:
    filepath = root / "datasets_links_640.txt"
    records = []
    with filepath.open("r") as f:
        for line in f.readlines():
            link = Path(line.strip())
            name, version = link.parent.stem, link.stem
            records.append(
                {
                    "dataset": name,
                    "version": version,
                    "link": f"{RF100_BASE_URL}/{name}/{version}",
                }
            )
    return pd.DataFrame.from_records(records).set_index("dataset")


def get_rf100_results(
    rf100_version_and_links: pd.DataFrame,
    rf100_stats: pd.DataFrame,
    rf100_labeling_info: pd.DataFrame,
) -> pd.DataFrame:

    df = rf100_stats.join(rf100_version_and_links).join(rf100_labeling_info)
    df = df.fillna(-1)
    df = df.astype({"labelling hours": int, "num_classes": int})
    return df


rf100_version_and_links = get_rf100_version_and_links(Path("."))
rf100_stats = pd.read_csv("./metadata/datasets_stats.csv", index_col=0)
rf100_train_results = pd.read_csv("./results.csv", index_col=0)
# let's also add labeling hours on the fly
rf100_labeling_info = pd.read_csv("./metadata/labeling_hours.csv", index_col=0)
results = get_rf100_results(rf100_version_and_links, rf100_stats, rf100_labeling_info)
results = results.reset_index()
# formatting for latex
format_name = lambda x: " ".join(x.split("-")[:2])
add_link = lambda link, name: "\href{" + link + "}{" + name + "}"
results["name"] = results.apply(
    lambda x: add_link(x["link"], format_name(x["dataset"])), axis=1
)
del results["version"]
del results["link"]
results = results.set_index("name", drop=True)
del results["dataset"]
del results["num_datasets"]
del results["size"]
labelling_hours_col = results.pop("labelling hours")
results.insert(4, "labelling hours", labelling_hours_col)
# # move "name" to beginning
# name_col = results.pop("name")
# results.insert(0, "name", name_col)
print(results.head(5))
# s = results.style.highlight_max(
#     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
# )
# https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
table_style = results.style
table_style.clear()
table_style.table_styles = []
table_style.caption = None
table_style = table_style.format(
    {
        "train": "{}",
        "val": "{}",
        "test": "{}",
        "yolov5": "{:.3f}",
        "yolov7": "{:.3f}",
        "glip": "{:.3f}",
    }
)
print(
    table_style.to_latex(
        "table.tex", hrules=True, clines="all;data", position_float="centering"
    )
)
#
