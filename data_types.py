from enum import Enum, auto
from collections import namedtuple


class PolicyType(Enum):
    ADAPTIVE_HEURISTIC = auto()
    ADAPTIVE_DEVIATION = auto()
    SKIP_RNN = auto()
    UNIFORM = auto()
    RANDOM = auto()


class CollectMode(Enum):
    TINY = auto()  # Data in FRAM
    LOW = auto()
    MED = auto()
    HIGH = auto()


PolicyResult = namedtuple('PolicyResult', ['measurements', 'collected_indices', 'collected_batches', 'batch_indices', 'num_collected'])