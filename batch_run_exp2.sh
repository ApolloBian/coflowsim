#!/usr/bin/env bash

for alg in wss fifo sebf sebf_rr; do
    echo $alg
    for bg in `seq 0 0.1 0.5`; do
        python exp2.py --algorithm $alg --bgflow $bg
    done
done
