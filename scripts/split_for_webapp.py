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

    data["image_paths"] = np.array(data["image_paths"])
    return data


def make_json_and_image_for_split(
    split_id: int,
    x: np.array,
    image_paths: np.array,
    size: Tuple,
    output_dir: Path,
    points_file_prefix: str = "",
):
    # split_output_dir = output_dir /  str(split_id)
    split_output_dir = output_dir
    split_output_dir_images = split_output_dir / "images"

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
        if points.shape[0] == 3:
            x, y, z = points.tolist()
        elif points.shape[0] == 2:
            x, y = points.tolist()
            z = 1
        else:
            raise ValueError(
                f"points have shape={points.shape}, only 2D and 3D are supported."
            )
        points_json.append({"x": x, "y": y, "z": z, "image_id": image_id})

    with (split_output_dir / f"{points_file_prefix}-points.json").open("w") as f:
        json.dump(points_json, f)


def get_data_chunks(
    data: dict, chunk_size: int
) -> Iterable[Tuple[int, dict, List[Path]]]:
    num_splits = max(1, data["x"].shape[0] // chunk_size)
    for i in range(num_splits):
        x = data["x"][i * chunk_size : (i + 1) * chunk_size]
        image_paths = data["image_paths"][i * chunk_size : (i + 1) * chunk_size]
        yield i, x, image_paths


def make_split(
    data: dict, output_dir: Path, montage_size: str, image_size: str, *args, **kwargs
):  
    """
    Takes `data`, writes to disk all the images and the embeddings in json format
    """
    montage_width, montage_height = [int(el) for el in montage_size.split("x")]
    image_width, image_height = [int(el) for el in image_size.split("x")]

    chunk_size = (montage_width * montage_height) // (image_width * image_height)

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda x: make_json_and_image_for_split(
                        *x, (image_width, image_height), output_dir, *args, **kwargs
                    ),
                    get_data_chunks(data, chunk_size),
                ),
                total=data["x"].shape[0] // chunk_size,
            )
        )



def make_splits_by_dataset(
    data: dict, output_dir: Path, montage_size: str, image_size: str, **kwargs
) -> List[dict]:
    # better way?
    datasets = np.array(
        [Path(x).parent.parent.parent.stem for x in data["image_paths"]]
    )
    buckets = {}
    for dataset in np.unique(datasets):
        indices = datasets == dataset
        buckets[str(dataset)] = {
            "x": data["x"][indices],
            "image_paths": data["image_paths"][indices],
        }

    for key, bucket in tqdm(buckets.items()):
        make_split(bucket, output_dir / key, montage_size, image_size, **kwargs)


if __name__ == "__main__":
    #     # from argparse import ArgumentParser

    #     # parser = ArgumentParser()

    #     # parser.add_argument("-i", type=Path, help="Input pkl file/")
    #     # parser.add_argument(
    #     #     "-o", type=Path, help="output dir."
    #     # )
    #     # parser.add_argument(
    #     #     "--montage-size", type=str, default="2048x2048", help="size of the montage in pixels."
    #     # )
    #     # parser.add_argument(
    #     #     "--image-size", type=str, default="64x64", help="new size for the images in pixels."
    #     # )

    #     # args = parser.parse_args()

    #     # input_path, output_dir, k, method = args.i, args.o, args.k, args.method
    input_path, output_dir, montage_size, image_size = (
        Path("./reduced-tsne-k=3.pk"),
        Path("./montages"),
        "2048x2048",
        "64x64",
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    make_splits_by_dataset(
        get_data(input_path),
        output_dir,
        montage_size,
        image_size,
        points_file_prefix=input_path.stem,
    )

#     make_split(input_path, output_dir, montage_size, image_size, points_file_prefix=input_path.stem)
