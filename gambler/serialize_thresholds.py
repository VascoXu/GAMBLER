from gc import collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import h5py
import random
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import List, Tuple
from collections import Counter

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    threshold_path = os.path.join(base, 'saved_models', args.dataset, 'thresholds_stream.json.gz')
    
    thresholds_list = read_json_gz(threshold_path)
    thresholds = thresholds_list[args.policy]['tiny']
    thresholds = list(thresholds.values())[:9]

    thresholds_fp = [(idx+3, threshold*(2 << 5)) for idx,threshold in enumerate(thresholds)] 
    print(thresholds_fp)
