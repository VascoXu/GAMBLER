#!/bin/sh

datasets=("epilepsy" "uci_har" "wisdm" "trajectories" "pedestrian" "pavement" "haptics" "eog")

for dataset in ${datasets[@]};
do
    python3 fit_thresholds.py --dataset $dataset --policy adaptive_heuristic --collect tiny --should-print
done
