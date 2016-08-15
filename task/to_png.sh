#!/bin/bash

for dir in "$@"; do
    echo "mogrify -format png ${dir}/*.bmp"
    mogrify -format png ${dir}/*.bmp

    echo "rm ${dir}/*.bmp"
    rm ${dir}/*.bmp
done

