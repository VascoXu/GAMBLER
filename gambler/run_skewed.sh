python3 analysis/plot_policy_error.py --dataset wisdm --window-size 20 --should-enforce-budget --num-runs 100 > wisdm_100.txt &
python3 analysis/plot_policy_error.py --dataset trajectories --window-size 20 --should-enforce-budget --num-runs 100 > trajectories_100.txt &
python3 analysis/plot_policy_error.py --dataset pedestrian --window-size 20 --should-enforce-budget --num-runs 100 > pedestrian_100.txt &
python3 analysis/plot_policy_error.py --dataset temperature --window-size 20 --should-enforce-budget --num-runs 100 > temperature_100.txt &
python3 analysis/plot_policy_error.py --dataset pavement --window-size 20 --should-enforce-budget --num-runs 100 > pavement_100.txt &
python3 analysis/plot_policy_error.py --dataset haptics --window-size 20 --should-enforce-budget --num-runs 100 > haptics_100.txt &
python3 analysis/plot_policy_error.py --dataset eog --window-size 20 --should-enforce-budget --num-runs 100 > eog_100.txt &
python3 analysis/plot_policy_error.py --dataset uci_har --window-size 20 --should-enforce-budget --num-runs 100 > uci_har_100.txt &

