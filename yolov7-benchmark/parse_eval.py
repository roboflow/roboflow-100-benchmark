import argparse

import yaml

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="Input file to parse")
ap.add_argument(
    "-l", "--location", required=True, help="Dataset location (where data.yaml is)"
)
ap.add_argument("-o", "--output", required=True, help="Output file to write")

args = vars(ap.parse_args())
location = args["location"]


def should_remove_line(line, stop_words):
    return any([word in line for word in stop_words])


with open(location + "/data.yaml", "r") as stream:
    class_names = yaml.safe_load(stream)["names"]

with open(args["input"], "r") as f:
    lines = f.readlines()
    eval_lines = []
    for line in lines:
        entries = line.split(" ")
        entries = list(filter(lambda val: val != "", entries))
        entries = [e.strip("\n") for e in entries]
        start_class = False
        for e in entries:
            if e == "all":
                if "(AP)" not in entries:
                    if "(AR)" not in entries:
                        # parse all
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

# get the mAP50 value
if len(eval_lines) > 1:
    print("There's more dicts")
    for lst in eval_lines:
        if lst["class"] == "all":
            map_val = lst["map50"]

else:
    print("There's only one dict res")
    map_val = [res["map50"] for res in eval_lines][0]

res = location, ": ", map_val


with open(args["output"], "a") as f:
    f.write("".join(res))
    f.write("\n")
