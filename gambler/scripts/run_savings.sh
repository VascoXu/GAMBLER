#!/bin/sh

python3 analysis/budget_reduction2.py --datasets epilepsy &
python3 analysis/budget_reduction2.py --datasets uci_har &
python3 analysis/budget_reduction2.py --datasets wisdm & 
python3 analysis/budget_reduction2.py --datasets trajectories & 
python3 analysis/budget_reduction2.py --datasets pedestrian &
python3 analysis/budget_reduction2.py --datasets pavement & 
python3 analysis/budget_reduction2.py --datasets haptics &
python3 analysis/budget_reduction2.py --datasets eog &
