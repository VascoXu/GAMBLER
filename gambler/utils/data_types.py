from enum import Enum, auto
from collections import namedtuple


class PolicyType(Enum):
    ADAPTIVE_HEURISTIC = auto()
    ADAPTIVE_LITESENSE = auto()
    ADAPTIVE_ELITESENSE = auto()
    ADAPTIVE_DEVIATION = auto()
    ADAPTIVE_BUDGET = auto()
    ADAPTIVE_UNIFORM = auto()
    ADAPTIVE_PROB = auto()
    ADAPTIVE_GAMBLER = auto()
    ADAPTIVE_GAMBLERV2 = auto()
    ADAPTIVE_GAMBLERV3 = auto()
    ADAPTIVE_GAMBLERV4 = auto()
    ADAPTIVE_TRAIN = auto()
    UNIFORM = auto()
    RANDOM = auto()


class CollectMode(Enum):
    TINY = auto()  # Data in FRAM
    LOW = auto()
    MED = auto()
    HIGH = auto()


PolicyResult = namedtuple('PolicyResult', ['measurements', 'estimate_list', 'collected_indices', 'window_indices', 'collection_ratios', 'num_collected', 'errors'])