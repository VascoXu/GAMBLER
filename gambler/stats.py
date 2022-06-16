import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.signal import savgol_filter

from gambler.utils.loading import load_data

def group_data(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    fold = 'test'
    dataset = 'epilepsy'
    dist = ''
    inputs, labels = load_data(dataset_name=dataset, fold=fold, dist=dist)

    num_seq, seq_length, num_features = inputs.shape
    
    # Parameters
    _alpha = 0.7
    _beta = 0.1
    
    data = []
    count = 0

    for idx, (sequence, label) in enumerate(zip(inputs, labels)):
        _mean = np.zeros(shape=(num_features, ))  # [D]
        _dev = np.zeros(shape=(num_features, ))
        for seq in sequence:
            _mean = (1.0 - _alpha) * _mean + _alpha * seq
            _dev = (1.0 - _beta) * _dev + _beta * np.abs(_mean - seq)
            if count == 200:
                data.append(sum(_dev))
                count = 0
            count += 1
    
    with open('moving_dist.txt', 'w') as f:
        for v in data:
            f.write("%f\n" % (v))

    plt.plot(data, label='Moving Deviation')
    plt.legend()
    plt.show()


