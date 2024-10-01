#!/bin/bash
set -euo pipefail
input="$(pwd)/datasets_links_640.txt"

while getopts f:l: flag
do
    case "${flag}" in
        f) format=${OPTARG};;
        l) location=${OPTARG};;
    esac
done

# Default values
format=${format:-coco}
location=${location:-$(pwd)/rf100}

echo "Starting downloading RF100 in parallel..."

# Create the datasets directory if it doesn't exist
mkdir -p "$location"

# Function to download a single dataset
download_single_dataset() {
    local link=$1
    local format=$2
    local location=$3

    attributes=$(python3 "$(pwd)/scripts/parse_dataset_link.py" -l "$link")
    project=$(echo "$attributes" | cut -d' ' -f 3)
    version=$(echo "$attributes" | cut -d' ' -f 4)

    if [ ! -d  "$location/$project" ] ; then
        echo "Downloading dataset $project..."
        python3 "$(pwd)/scripts/download_dataset.py" -p "$project" -v "$version" -l "$location" -f "$format"
    else
        echo "Dataset $project already exists. Skipping download."
    fi
}

export -f download_single_dataset

# Read the dataset links and download in parallel
cat "$input" | xargs -P20 -I{} bash -c 'download_single_dataset "{}" "'"$format"'" "'"$location"'"'

echo "Done downloading datasets!"
