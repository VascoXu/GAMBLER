import json
import numpy as np
import matplotlib.pyplot as plt

from gambler.analysis.plot_utils import bar_plot
from gambler.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

running_filename = 'running.txt'
walking_filename = 'walking.txt'

running = []
walking= []
running_error = 0
walking_error = 0

with open(running_filename) as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]

    # read collection rates
    running = json.loads(lines[0])

    # error
    running_error = lines[1].split(',')[0]


with open(walking_filename) as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]

    # read collection rates
    walking = json.loads(lines[0])

    # error
    walking_error = lines[1].split(',')[0]


# cap 
length = min(len(running), len(walking))
running = running[:length]
walking = walking[:length]

with plt.style.context('seaborn-ticks'):
    fig, ax1 = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * 0.75))

    ax1.plot(list(range(len(running))), running, label='Early Exhaustion', linewidth=LINE_WIDTH)
    ax1.plot(list(range(len(walking))), walking, label='Energy Waste', linewidth=LINE_WIDTH)

    ax1.set_xlabel('Window Number', fontsize=AXIS_FONT)
    ax1.set_ylabel('Collection Rate', fontsize=AXIS_FONT)

    ax1.set_title('Collection Rate per Window', fontsize=TITLE_FONT)

    # ax1.text(0.56, 0.05, s='Adapt #: {0}'.format('ABC'), fontsize=LEGEND_FONT, transform=ax1.transAxes)
    # ax1.text(0.78, 0.05, s='Adapt Error: {0:.3f}'.format(0.123), fontsize=LEGEND_FONT, transform=ax1.transAxes)

    # ax1.text(0.56, 0.1, s='Uniform #: {0}'.format('ABC'), fontsize=LEGEND_FONT, transform=ax1.transAxes)
    # ax1.text(0.78, 0.1, s='Uniform Error: {0:.3f}'.format(0.123), fontsize=LEGEND_FONT, transform=ax1.transAxes)

    ax1.legend()
    plt.show()