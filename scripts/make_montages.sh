#!/bin/bash
set -euo pipefail

dir=$(pwd)
echo $dir
root=$dir/montages

for split_id in $(ls $root | sort -V)
do
    echo "Montage for $split_id"
    montage_dir=$root/$split_id/
    ls $root/$split_id/images/* | sort -V > $root/$split_id/images_to_montage.txt
    montage `cat $root/$split_id/images_to_montage.txt` -geometry +0+0 -background none -tile 32x $root/$split_id/2048-img-atlas.jpg
done

echo "Done!"
