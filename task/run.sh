#!/bin/bash

OPTS="--seed 1337 --lambda 1 --try-epochs 10 --weight-decay 0.003 --learning-rate 0.001 --net-dir net --test-dir test_data --train-dir train_data --visualize --finetune"
PREV=""

for i in $(seq 1 6); do
    TASK="SCT${i}"
    DIR="net${i}"

    echo "yes | th train.lua --task $TASK $OPTS $PREV"
    yes | th train.lua --task $TASK $OPTS $PREV
    mkdir $DIR
    cp net/* $DIR
    PREV="$PREV $TASK"
done

rm -r net
mkdir "$1"
mv net? "$1"
