import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
dataset_file = 'hi.txt'
data_file = os.path.join(dirname, '.', dataset_file)

dist = []
with open(data_file) as f:
    lines = f.read().splitlines()
    del lines[-1]
    del lines[-1]
    dist = list(map(float, lines))

plt.plot(dist, label='Collection Rate')
plt.legend()
plt.show()
