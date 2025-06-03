#!/bin/bash

SCRIPTPATH="$(cd "$(dirname "$0")" && pwd)"
DATAPATH="$(cd "$SCRIPTPATH"/../data/raw && pwd)"
LOGFILE="$LOGPATH"/collect.logs
SOURCE_URL="https://files.grouplens.org/datasets/movielens/"

DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATETIME_SAVE=${DATETIME:0:4}${DATETIME:5:2}${DATETIME:8:2}_${DATETIME:11:2}${DATETIME:14:2}

datasets=("ml-latest-small" "ml-latest" "ml-20m")

for ds in ${datasets[*]}; do
    echo $DATETIME
    ds_path=$DATAPATH/$ds
    fn=$ds_path.zip
    if [ -d $ds_path ]; then
        echo "Directory $ds_path aleady exists. No download necessary."
    else
        echo "Downloading $ds.zip ..."
        wget -P $DATAPATH $SOURCE_URL$ds.zip
        echo "Unzipping $fn ..."
        unzip $fn -d $DATAPATH
        echo "Removing $fn ..."
        rm $fn
    fi     
done
