#!/bin/bash

python model.py \
    datasets/track1/middle06/01_smooth_forward.csv \
    datasets/track1/middle06/02_turn_left.csv \
    datasets/track1/middle06/03_sharp_turn_left.csv \
    datasets/track1/middle06/04_turn_right.csv \
    datasets/track1/right01/recovery.csv \
    datasets/track1/left01/recovery.csv \
    --architecture classification
