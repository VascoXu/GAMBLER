#!/bin/sh

python3 analysis/experiments.py --randomize skewed --rand-amount 1 --num-runs 100 --datasets epilepsy uci_har wisdm trajectories pedestrian temperature pavement haptics eog 
