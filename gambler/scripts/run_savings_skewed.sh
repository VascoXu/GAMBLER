#!/bin/sh

python3 analysis/energy_savings.py --datasets epilepsy uci_har wisdm trajectories pedestrian temperature pavement haptics eog --randomize skewed --num-runs 100
