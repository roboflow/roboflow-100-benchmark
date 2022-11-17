"""Code to generate the markdown table in the README.md
"""
import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import make_grid

ROOT = Path("./rf100")


def read_and_transform(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    return pil_to_tensor(image)


def make_grid_from_dataset(dataset_dir: Path, n=5) -> Tuple[str, Image.Image]:
    # get random 5 images
    images_paths = list((dataset_dir / "train" / "images").glob("*"))
    sampled_images_paths = random.choices(images_paths, k=n)
    tensors = [read_and_transform(image_path) for image_path in sampled_images_paths]
    grid = make_grid(tensors, nrow=n)
    return dataset_dir.stem, to_pil_image(grid)


def save(dataset_name: str, grid: Image.Image) -> Tuple[str, str]:
    grid_path = f"doc/images/grid/{dataset_name}.jpg"
    # grid.save(grid_path, quality=70)
    return dataset_name, grid_path


def make_grids(root: Path):
    out = list(map(make_grid_from_dataset, root.iterdir()))
    out = list(map(lambda x: save(*x), out))
    return out


def make_table(root: Path) -> str:
    out = make_grids(root)
    dataset_names, grid_paths = zip(*out)
    category_df = pd.read_csv("metadata/categories.csv", index_col=0)
    df = pd.DataFrame(
        data=dict(dataset=dataset_names, samples=grid_paths),
        columns=["dataset", "samples"],
    )
    df.samples = df.samples.apply(lambda x: f"![alt]({x})")
    df = df.set_index("dataset", drop=True)
    df = category_df.join(df)
    del df["category"]
    return df.to_markdown()


print(make_table(ROOT))
