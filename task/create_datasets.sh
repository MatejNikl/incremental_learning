#!/bin/bash

temp=`mktemp`

trap "echo 'Caught signal!'; rm $temp; exit 1" SIGQUIT SIGABRT SIGINT

function help() {
    echo "usage: $0 .../*train*/dir"
    rm $temp
    exit 1
}


for dir in "$@"; do
    if [ ! -d "$dir" ] ; then
        help
    fi

    echo -e "\n"
    echo th dataset_creator.lua --csv "$dir/labels.csv" --dir "$dir" "${dir}.t7" | tee $temp
    th dataset_creator.lua --csv "$dir/labels.csv" --dir "$dir" "${dir}.t7" | tee $temp

    mean=`sed -n 's/^mean: \([0-9]\+.[0-9]\+\)/\1/p' $temp`
    std=`sed -n 's/^std:  \([0-9]\+.[0-9]\+\)/\1/p' $temp`

    dir=`sed 's/train/test/' <<< "$dir"`

    if [ ! -d "$dir" ] ; then
        help
    fi

    echo
    echo th dataset_creator.lua --csv "$dir/labels.csv" --dir "$dir" --mean $mean --std $std "${dir}.t7"
    th dataset_creator.lua --csv "$dir/labels.csv" --dir "$dir" --mean $mean --std $std "${dir}.t7"
done

rm $temp

