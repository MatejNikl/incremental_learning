#!/bin/bash

DIR="$1"
NAME="SCT"

for n in $(seq 1 6); do
    TASK="${NAME}${n}"

    echo $TASK

    if [[ ${n} -gt 1 ]]; then
        for i in $(seq 1 $(( $n - 1 ))); do
            echo "${NAME}${i}"
        done
    fi

    for i in $(seq $n 6); do
        ACC=$(th train.lua --task $TASK --net-dir ${DIR}/net${i} --test-dir test_data | tail -n 1 | tr -s ' ' | cut -d ' ' -f5)

        echo "${NAME}${i} ${ACC}"
    done
    echo
    echo
done
