#!/bin/sh
datasets=("epilepsy" "uci_har" "wisdm" "trajectories" "pedestrian" "temperature" "pavement" "haptics" "eog")
windows=(5 10 20 30 40 50 60 70 80 90 100)

for window in ${windows[@]};
do
    python3 train_gambler.py --datasets epilepsy uci_har wisdm trajectories pedestrian temperature pavement haptics eog --should-print --window-size $window
done
