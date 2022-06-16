import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from gambler.utils.loading import load_data

def group_data(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]

fold = 'test'
dataset = 'epilepsy'
dist = ''
inputs, labels = load_data(dataset_name=dataset, fold=fold, dist=dist)

num_seq, seq_length, num_features = inputs.shape
_alpha = 0.7
_beta = 0.09
var = []
data = []
data2 = []

count = 0

for idx, (sequence, label) in enumerate(zip(inputs, labels)):
    _mean = np.zeros(shape=(num_features, ))  # [D]
    _dev = np.zeros(shape=(num_features, ))
    for seq in sequence:
        _mean = (1.0 - _alpha) * _mean + _alpha * seq
        _dev = (1.0 - _beta) * _dev + _beta * np.abs(_mean - seq)
        if count == 200:
            # print(sum(_dev))
            count = 0
            data.append(sum(_dev))

        count += 1
        # x = seq[0]
        # y = seq[1]
        # z = seq[2]
        # accel = (x**2 + y**2 + z**2)**0.5
        # data.append(accel)
        # data2.append((accel, label))

fold = 'test'
dataset = 'epilepsy'
dist = ''
inputs, labels = load_data(dataset_name=dataset, fold=fold, dist=dist)
_alpha = 0.7
_beta = 0.7
devs = []
for idx, (sequence, label) in enumerate(zip(inputs, labels)):
    _mean = np.zeros(shape=(num_features, ))  # [D]
    _dev = np.zeros(shape=(num_features, ))
    for seq in sequence:
        _mean = (1.0 - _alpha) * _mean + _alpha * seq
        _dev = (1.0 - _beta) * _dev + _beta * np.abs(_mean - seq)
        if count == 200:
            # print(sum(_dev))
            count = 0
            avg_data = []
            # data2.append(sum(_dev))
            data2.append(sum(devs)/len(devs))
            devs = []
        else:
            devs.append(sum(_dev))

        count += 1
        # x = seq[0]
        # y = seq[1]
        # z = seq[2]
        # accel = (x**2 + y**2 + z**2)**0.5
        # data.append(accel)
        # data2.append((accel, label))

plt.plot(data, label='Normal')
# data2 = savgol_filter(data2, 7, 2)
plt.plot(data2, label='Smoothed')
plt.legend()
plt.show()

# mean = np.mean(data)
# std = np.std(data)

# print("Mean: ", mean, " Std: ", std)

# data = group_data(data, 100)
# for seq in data:
#     x = seq[0]
#     y = seq[1]
#     z = seq[2]
#     x_i = (x**2 + y**2 + z**2)**0.5    
    
#     var_i = (sum((x_i - mean)**2 for x_i in seq) / len(seq))**0.5
#     var.append(var_i)

# with open('dist.txt', 'w') as f:
#     for v in var:
#         f.write("%f\n" % (v))


