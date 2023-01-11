
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

# Goal: I want to holdout one class of data? Can it still learn?
# - How many classes of data do I want to holdout? 
# Steps:
# 1. Train a threshold with one class missing 
# 2. Train a gambler with that threshold

# Test:
# 1. Test on expected data with all the classes across all budgets
# 2. Test on skewed data with all the classes across all budgets


