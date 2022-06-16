import numpy as np
import math
import random
import matplotlib.pyplot as plt

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.constants import ALPHA, P
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution


distribution = load_distribution('moving_dist.txt')
interp = np.interp(distribution, [min(distribution), max(distribution)], [0, 100])
dev = np.interp(0.9, [min(distribution), max(distribution)], [0, 100])
percentile = np.percentile(interp, dev)

plt.plot(distribution)
plt.show()

