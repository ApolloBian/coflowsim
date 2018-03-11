#!/usr/bin/env bash

echo exp3.1:

for alg in sebf sebf_bg; do
    echo $alg
    bg=0.1
    for ddl in `seq 0.5 0.5 0.5`; do
        python exp3.py --algorithm $alg --bgflow $bg --ddl_ratio $ddl
    done
done
