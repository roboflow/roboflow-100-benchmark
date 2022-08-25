from roboflow import Roboflow
import os 
import argparse

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-p", "--project", required=True, help="The project ID of the dataset found in the dataset URL.")
    ap.add_argument("-v", "--version", required=True, help="The version the dataset you want to use")
    ap.add_argument("-d", "--download", required=True, help="The format of the export you want to use (i.e. coco or yolov5)")

    # parses command line arguments 
    args = vars(ap.parse_args())

    project = args["project"]
    version = args["version"]
    download = args["download"]


    rf = Roboflow(api_key="ddkQGNIW95EfGbudHOeh") #change this to parameter 
    project = rf.workspace("roboflow-100").project(project)
    dataset = project.version(int(version)).download(download, location="dataset")

    with open('../loc.txt', 'w') as f:
        f.write(dataset.location)

    return 


if __name__ == "__main__":
    main()