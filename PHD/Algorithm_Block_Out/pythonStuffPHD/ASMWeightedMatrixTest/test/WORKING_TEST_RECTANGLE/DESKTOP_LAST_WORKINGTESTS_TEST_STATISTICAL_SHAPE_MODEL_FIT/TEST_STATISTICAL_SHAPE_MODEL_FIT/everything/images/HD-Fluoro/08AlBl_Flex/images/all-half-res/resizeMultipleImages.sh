#!/bin/bash
# resizeMultipleImages.sh
# basic image resizer

echo "\$1 extension of files to be renamed"


for f in *.$1;#specify which files to match based on extension
do

    #new=$(printf "%0${pad}d" "$COUNTER") #04 pad to length of 4
    #echo mv "$f"  ${2}$new"."${4}
    #echo convert "$f" -resize 512x512 "$f"
    convert "$f" -resize 512x512 "$f"

    #mv "$f"  ${2}$new"."${4}
    let COUNTER=COUNTER+1
done
echo "Done!"
