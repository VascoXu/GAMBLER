import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.signal import savgol_filter

from gambler.utils.loading import load_data

def group_data(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    df = pd.read_csv('eps.csv', delim_whitespace=True)
    td_errors = df.iloc[:,0]
    eps = df.iloc[:,1]

    fig, ax = plt.subplots()

    ax.plot(td_errors, color='red', label='TD Error')
    ax.tick_params(axis='y', labelcolor='red')

    ax2 = ax.twinx()

    ax2.plot(eps, color='green', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.legend()
    plt.show()
