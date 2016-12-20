#!/bin/bash

python model.py \
    datasets/track1/middle01/01_smooth_forward.csv \
    datasets/track1/middle01/02_turn_left.csv \
    datasets/track1/middle01/03_turn_right.csv \
    datasets/track1/right01/recovery.csv \
    datasets/track1/left01/recovery.csv \
    datasets/track1/middle02/driving_log.csv \
    --architecture classification
