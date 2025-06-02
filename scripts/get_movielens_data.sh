#!/bin/bash

SCRIPTPATH="$(cd "$(dirname "$0")" && pwd)"
DATAPATH="$(cd "$SCRIPTPATH"/../data/raw && pwd)"
LOGFILE="$LOGPATH"/collect.logs
SOURCE_URL="https://files.grouplens.org/datasets/movielens/"

DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATETIME_SAVE=${DATETIME:0:4}${DATETIME:5:2}${DATETIME:8:2}_${DATETIME:11:2}${DATETIME:14:2}

filenames=("ml-latest-small.zip" "ml-latest.zip" "ml-20m.zip")

for fn in ${filenames[*]}; do
    echo $DATETIME
    filepath=$DATAPATH/$fn
    if [ -f $filepath ]; then
        echo "File $fn aleady exists. No download necessary."
    else
        echo "Downloading $fn"
        wget -P $DATAPATH $SOURCE_URL$fn
        echo "Unzipping $fn"
        unzip $filepath -d $DATAPATH
    fi    
done
