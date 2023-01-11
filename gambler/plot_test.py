import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, default='hi.txt')
    args = parser.parse_args()

dirname = os.path.dirname(__file__)
dataset_file = args.filename
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
