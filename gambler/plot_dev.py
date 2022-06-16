import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
dataset_file = 'dist.txt'
data_file = os.path.join(dirname, '.', dataset_file)

real_dist = []
with open(data_file) as f:
    lines = f.read().splitlines()
    real_dist = list(map(float, lines))


dirname = os.path.dirname(__file__)
dataset_file = 'moving_dist.txt'
data_file = os.path.join(dirname, '.', dataset_file)

moving_dist = []
with open(data_file) as f:
    lines = f.read().splitlines()
    moving_dist = list(map(float, lines))


plt.plot(moving_dist, label='Moving Deviation')
plt.legend()
plt.show()
