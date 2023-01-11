#!/bin/sh
datasets=("epilepsy" "uci_har" "wisdm" "trajectories" "pedestrian" "temperature" "pavement" "haptics" "eog")

for dataset in ${datasets[@]}; do
    python3 train_gambler.py --dataset $dataset --window-size 20
done
