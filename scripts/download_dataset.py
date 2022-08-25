from argparse import ArgumentParser

from roboflow import Roboflow


def main():
    # construct the argument parser and parse the arguments
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--project",
        required=True,
        type=str,
        help="The project ID of the dataset found in the dataset URL.",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        type=int,
        help="The version the dataset you want to use",
    )
    parser.add_argument(
        "-d",
        "--download",
        required=True,
        type=str,
        help="The format of the export you want to use (i.e. coco or yolov5)",
    )
    parser.add_argument(
        "-k",
        "--api_key",
        required=True,
        type=str,
        help="The api key to access Roboflow, see https://docs.roboflow.com/rest-api.",
    )

    # parses command line arguments
    args = vars(parser.parse_args())

    rf = Roboflow(api_key=args["api_key"])  # change this to parameter
    project = rf.workspace("roboflow-100").project(args["project"])
    dataset = project.version(args["version"]).download(
        args["download"], location="dataset"
    )

    with open("../loc.txt", "w") as f:
        f.write(dataset.location)


if __name__ == "__main__":
    main()
