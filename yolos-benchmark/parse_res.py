import os
import argparse
from os import path

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--loc", required=True, help="location")
    args = vars(ap.parse_args())
    loc = args["loc"] 

    # read the file with stdout data
    res_file = open("yolos_res.txt", "r")
    # get list of lines
    lines = res_file.readlines()
    res_file.close()

    # grad the line with the value we are interested in
    res_file = open("yolos_res.txt", "w")
    for line in lines:
        if "IoU=0.50 " in line.strip("\n"):
            # Delete "line2" from new_file
            res_file.write(line)
            res = loc, ": ", line[-6:] # value that we want to save to res file

    res_file.close()
    
    with open('mAP_s.txt', 'a') as f:
        f.write(''.join(res))
        f.write("\n")
    





if __name__ == "__main__":
    main()