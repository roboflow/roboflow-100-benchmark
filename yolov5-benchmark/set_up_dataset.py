from roboflow import Roboflow
import os 
import argparse

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # ap.add_argument("-k", "--api_key", required=True, help="api key") 
    ap.add_argument("-w", "--workspace", required=True, help="workspace")
    ap.add_argument("-p", "--project", required=True, help="project")
    ap.add_argument("-v", "--version", required=True, help="version")
    ap.add_argument("-d", "--download", required=True, help="download")

    # parses command line arguments 
    args = vars(ap.parse_args())

    # api_key = args["api_key"] 
    workspace = args["workspace"] 
    project = args["project"]
    version = args["version"]
    download = args["download"]


    rf = Roboflow(api_key="ddkQGNIW95EfGbudHOeh")
    project = rf.workspace(workspace).project(project)
    dataset = project.version(int(version)).download(download)

    with open('../loc.txt', 'w') as f:
        f.write(dataset.location)


    return 


if __name__ == "__main__":
    main()