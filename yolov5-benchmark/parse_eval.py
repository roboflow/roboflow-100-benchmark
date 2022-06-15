import json
import yaml
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-l", "--loc", required=True, help="data file location")
args = vars(ap.parse_args())
loc = args["loc"] 


with open(loc + "/data.yaml", 'r') as stream:
    class_names = yaml.safe_load(stream)['names']

with open("val_eval.txt", "r") as f:
    lines = f.readlines()
    eval_lines = []
    for line in lines:
        entries = line.split(" ")
        entries = list(filter(lambda val: val !=  "", entries))
        entries = [e.strip("\n") for e in entries]
        start_class = False
        for e in entries:
            if e == "all":
                if "(AP)" not in entries:
                    if "(AR)" not in entries:
                        #parse all
                        eval = {}
                        eval["class"] = entries[0]
                        eval["images"] = entries[1]
                        eval["targets"] = entries[2]
                        eval["precision"] = entries[3]
                        eval["recall"] = entries[4]
                        eval["map50"] = entries[5]
                        eval["map95"] = entries[6]
                        eval_lines.append(eval)

            if e in class_names:
                eval = {}
                eval["class"] = entries[0]
                eval["images"] = entries[1]
                eval["targets"] = entries[2]
                eval["precision"] = entries[3]
                eval["recall"] = entries[4]
                eval["map50"] = entries[5]
                eval["map95"] = entries[6]
                eval_lines.append(eval)

print("-- This is the result:")
print(eval_lines)
