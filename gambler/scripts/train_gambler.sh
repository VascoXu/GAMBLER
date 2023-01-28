#!/bin/sh
# datasets=("epilepsy" "uci_har" "wisdm" "trajectories" "pedestrian" "pavement" "haptics" "eog")
windows=(5 10 20 30 40 50 60 70 80 90 100)

for window in ${windows[@]};
do
    # python3 train_gambler.py --datasets epilepsy uci_har wisdm trajectories pedestrian pavement haptics eog --should-print --should-retrain --window-size $window
    python3 train_gambler.py --datasets haptics --should-print --should-retrain --window-size $window
done
