    #!/bin/python3

import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
import os
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools
import random
import statistics
from collections import Counter

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


def remove_dataset(dataset):
    dist_num = 1
    distribution = 'expected'
    output_file = os.path.join('results', f'{distribution}_distribution_{dist_num}.json.gz')
    results_dict = read_json_gz(output_file) if os.path.exists(output_file) else dict()
    results_dict.pop(dataset, None)

    save_json_gz(results_dict, output_file)


if __name__ == '__main__':
    remove_dataset("temperature")
