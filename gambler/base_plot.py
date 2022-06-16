import matplotlib.pyplot as plt

with open('hi.txt') as f:
    lines = f.read().splitlines()
    lines = list(map(float, lines))
    plt.plot(lines)
    plt.show()
