import numpy as np
import os.path
from pickle import NONE
import pexpect
import csv 
import matplotlib.pyplot as plt
import pandas as pd

from file_utils import read_json, read_pickle_gz, read_json_gz

def run_cmd(cmd):
    try:
        policy_obj = NONE
        policy_obj = pexpect.spawn(cmd)
        policy_obj.expect(pexpect.EOF)
        response = policy_obj.before.decode("utf-8").strip()
        return response
    finally:
        policy_obj.close()   

STATIC_POLICY_CMD = 'python run_policy.py --dataset {0} --policy {1} --collection-rate {2} --window {3}'

def fit():
    df = pd.read_csv('results/rates.csv')
    thresholds = df['threshold']
    ratios = df['sampling_ratio']

    thresholds = (thresholds/thresholds[0]).tolist()
    ratios = (ratios/ratios[0]).tolist()

    slope, intercept = np.polyfit(thresholds,ratios,1)
    print(f'Alpha: {slope} Intercept: {intercept}')


def fit2():
    for i in range(4):
        thresholds = read_json_gz(f'saved_models/epilepsy/thresholds_{i}_block.json.gz')
        
        ratios = []
        collection_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for rate in collection_rates:
            static_cmd = STATIC_POLICY_CMD.format('epilepsy', 'adaptive_deviation', rate, 500)
            static_cmd += f' --labels {i}'
            response = run_cmd(static_cmd)
            error, num_collected, total_samples = response.split(',')
            ratios.append(int(num_collected)/int(total_samples))
            
        thresholds = list(thresholds['adaptive_deviation']['tiny'].values())

        slope, intercept = np.polyfit(thresholds,ratios,1)
        print(f'Label {i} | Alpha: {slope} Intercept: {intercept}')

    thresholds = read_json_gz(f'saved_models/epilepsy/thresholds_block.json.gz')
    collection_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ratios = []
    for rate in collection_rates:
        static_cmd = STATIC_POLICY_CMD.format('epilepsy', 'adaptive_deviation', rate, 500)
        response = run_cmd(static_cmd)
        error, num_collected, total_samples = response.split(',')
        ratios.append(int(num_collected)/int(total_samples))

    thresholds = list(thresholds['adaptive_deviation']['tiny'].values())

    slope, intercept = np.polyfit(thresholds,ratios,1)
    print(f'ALL | Alpha: {slope} Intercept: {intercept}')

fit2()