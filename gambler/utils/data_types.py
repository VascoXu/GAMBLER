from enum import Enum, auto
from collections import namedtuple


class PolicyType(Enum):
    ADAPTIVE_HEURISTIC = auto()
    ADAPTIVE_LITESENSE = auto()
    ADAPTIVE_DEVIATION = auto()
    ADAPTIVE_CONTROLLER = auto()
    ADAPTIVE_GREEDY = auto()
    ADAPTIVE_UNIFORM = auto()
    ADAPTIVE_BANDIT = auto()
    UNIFORM = auto()
    RANDOM = auto()


class CollectMode(Enum):
    TINY = auto()  # Data in FRAM
    LOW = auto()
    MED = auto()
    HIGH = auto()


PolicyResult = namedtuple('PolicyResult', ['measurements', 'collected_indices', 'collection_ratios', 'num_collected', 'collected_within_window', 'curr_window_size'])