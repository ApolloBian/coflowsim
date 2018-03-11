#!/usr/bin/env bash

echo exp3.2:

for alg in sebf sebf_bg; do
    echo $alg
    ddl=2
    for bg in `seq 0 0.1 0.5`; do
        python exp3_2.py --algorithm $alg --bgflow $bg --ddl_ratio $ddl
    done
done
