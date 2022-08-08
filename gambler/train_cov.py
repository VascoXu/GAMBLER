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
    fold = 'train'
    dataset = 'epilepsy'
    dist = ''
    inputs, labels = load_data(dataset_name=dataset, fold=fold, dist=dist)

    num_seq, seq_length, num_features = inputs.shape
    
    # Parameters
    _alpha = 0.7
    _beta = 0.1
    
    data = []
    count = 0

    deviations = []
    means = []

    for idx, (sequence, label) in enumerate(zip(inputs, labels)):
        _mean = 0
        _dev = 0
        for seq in sequence:
            # pack sequence into total accel
            measurement = (seq[0]**2 + seq[1]**2 + seq[2]**2)**0.5

            _mean = (1.0 - _alpha) * _mean + _alpha * measurement
            _dev = (1.0 - _beta) * _dev + _beta * np.abs(_mean - measurement)

            if count == 100:
                # calculate deviations
                _devs = sum(deviations)/len(deviations)
                _means = sum(means)/len(means)
                _cov = _devs/_means

                data.append(_cov)
                
                # reset
                deviations = []
                means = []
                count = 0
            else:
                deviations.append(_dev)
                means.append(_mean)

            count += 1

    print("Max Cov: ", max(data))

    plt.plot(data, label='Moving Deviation')
    plt.legend()
    plt.show()
