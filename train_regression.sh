#!/bin/bash

python model.py \
    datasets/track1/middle01/driving_log.csv \
    datasets/track1/right01/recovery.csv \
    datasets/track1/left01/recovery.csv \
    datasets/track1/middle02/driving_log.csv \
    --architecture regression \
    --epochs 20
