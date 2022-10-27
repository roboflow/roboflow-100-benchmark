#!/bin/bash
set -euo pipefail

dir=$(pwd)
echo $dir
root=$dir/montages

mkdir -p $root/montages
image_dirs=$root/images
for image_dir in $(ls $image_dirs | sort -V)
do
    echo "Montage for $image_dir"
    montage_dir=$root/montages/$image_dir
    mkdir -p $montage_dir
    ls $image_dirs/$image_dir/* | sort -V > $montage_dir/images_to_montage.txt
    montage `cat $montage_dir/images_to_montage.txt` -geometry +0+0 -background none -tile 32x $montage_dir/2048-img-atlas.jpg
done

echo "Done!"
