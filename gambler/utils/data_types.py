from enum import Enum, auto
from collections import namedtuple


class PolicyType(Enum):
    ADAPTIVE_HEURISTIC = auto()
    ADAPTIVE_LITESENSE = auto()
    ADAPTIVE_ELITESENSE = auto()
    ADAPTIVE_DEVIATION = auto()
    ADAPTIVE_BUDGET = auto()
    ADAPTIVE_UNIFORM = auto()
    ADAPTIVE_BANDIT = auto()
    ADAPTIVE_GAMBLER = auto()
    ADAPTIVE_TRAIN = auto()
    UNIFORM = auto()
    RANDOM = auto()


class CollectMode(Enum):
    TINY = auto()  # Data in FRAM
    LOW = auto()
    MED = auto()
    HIGH = auto()


PolicyResult = namedtuple('PolicyResult', ['measurements', 'collected_indices', 'collection_ratios', 'num_collected', 'training_data'])