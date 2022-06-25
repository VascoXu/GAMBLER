import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
dataset_file = 'hi.txt'
data_file = os.path.join(dirname, '.', dataset_file)

dist = []
with open(data_file) as f:
    lines = f.read().splitlines()
    dist = list(map(float, lines))

plt.plot(dist, label='Coefficient of Variance')
plt.legend()
plt.show()
