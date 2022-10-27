"""
This script will split the reduced-* pickle file into folders with the right dimension and metadata for the collage.
"""

from pathlib import Path
from typing import Iterable, Tuple, List
from tqdm import tqdm
from pickle import load, dump
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import json

"""
1) read the data
2) divide the data in chuck, each chunk is the amount of images we can fit
    a) read images with a thread pool
    b) resize all the images with PIL
    c) save all the images with a thread pool
    d) create a json with { x, y, z } 
    c) the json is named from 0 ... (number of chunks) to link json <-> montage
"""


def get_data(input_path: Path) -> dict:
    with input_path.open("rb") as f:
        data = load(f)
    return data


def make_json_and_image_for_split(
    split_id: int, x: np.array, image_paths: np.array, size: Tuple, output_dir: Path
):
    split_output_dir_images = output_dir / "images" / str(split_id)

    split_output_dir_points = output_dir / "points" / str(split_id)
    split_output_dir_points.mkdir(exist_ok=True, parents=True)
    images_ids = range(len(image_paths))

    if not split_output_dir_images.exists():
        split_output_dir_images.mkdir(exist_ok=True, parents=True)
        # open all the images
        images: List[Image.Image] = list(
            map(lambda x: Image.open(x).convert("RGB"), image_paths)
        )
        # resize images
        images: List[Image.Image] = list(map(lambda x: x.resize(size), images))
        # store them to disk
        for image_id in images_ids:
            image = images[image_id]
            image.save(
                split_output_dir_images / f"{str(image_id)}.jpeg",
                format="JPEG",
                quality=70,
            )
    # create the json
    points_json = []
    for points, image_id in zip(x, images_ids):
        x, y, z = points.tolist()
        points_json.append({"x": x, "y": y, "z": z, "image_id": image_id})

    with (split_output_dir_points / "points.json").open("w") as f:
        json.dump(points_json, f)


def get_data_chunks(
    data: dict, chunk_size: int
) -> Iterable[Tuple[int, dict, List[Path]]]:
    num_splits = data["x"].shape[0] // chunk_size
    for i in range(num_splits):
        x = data["x"][i * chunk_size : (i + 1) * chunk_size]
        image_paths = data["image_paths"][i * chunk_size : (i + 1) * chunk_size]
        yield i, x, image_paths


def make_split(input_path: Path, output_dir: Path, montage_size: str, image_size: str):
    data = get_data(input_path)
    montage_width, montage_height = [int(el) for el in montage_size.split("x")]
    image_width, image_height = [int(el) for el in image_size.split("x")]

    chunk_size = (montage_width * montage_height) // (image_width * image_height)

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda x: make_json_and_image_for_split(
                        *x, (image_width, image_height), output_dir
                    ),
                    get_data_chunks(data, chunk_size),
                ),
                total=data["x"].shape[0] // chunk_size,
            )
        )


if __name__ == "__main__":
    # from argparse import ArgumentParser

    # parser = ArgumentParser()

    # parser.add_argument("-i", type=Path, help="Input pkl file/")
    # parser.add_argument(
    #     "-o", type=Path, help="output dir."
    # )
    # parser.add_argument(
    #     "--montage-size", type=str, default="2048x2048", help="size of the montage in pixels."
    # )
    # parser.add_argument(
    #     "--image-size", type=str, default="64x64", help="new size for the images in pixels."
    # )

    # args = parser.parse_args()

    # input_path, output_dir, k, method = args.i, args.o, args.k, args.method
    input_path, output_dir, montage_size, image_size = (
        Path("./reduced-pca-k=3.pk"),
        Path("./montages"),
        "2048x2048",
        "64x64",
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    make_split(input_path, output_dir, montage_size, image_size)
