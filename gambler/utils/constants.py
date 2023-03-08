DEBUG = True
ADAPTIVE = False

SMALL_NUMBER = 1e-7
BIG_NUMBER = 1e7

BITS_PER_BYTE = 8
MIN_PRECISION = 1
MAX_PRECISION = 16
MIN_WIDTH = 5
SHIFT_BITS = 4

MAX_SHIFT_GROUPS = 6
MIN_SHIFT_GROUPS = 4
MAX_SHIFT_GROUPS_FACTOR = 0.05

LENGTH_SIZE = 2
LENGTH_ORDER = 'little'

PERIOD = 4
BT_FRAME_SIZE = 20

POLICIES = ['random', 'uniform', 'adaptive_heuristic', 'adaptive_deviation', 'skip_rnn']
ENCODING = ['standard', 'group', 'group_unshifted', 'single_group', 'padded', 'pruned']
ENCRYPTION = ['stream', 'block']
COLLECTION = ['tiny', 'low', 'med', 'high']

WINDOW_SIZES = {
    'epilepsy': 10,
    'uci_har': 50, 
    'wisdm': 20,
    'trajectories': 50,
    'pedestrian': 20, 
    'pavement': 30, 
    'haptics': 5,
    'eog': 50
}

DATASETS = [
    'epilepsy',
    'uci_har',
    'wisdm',
    'trajectories',
    'pedestrian',
    'pavement',
    'haptics',
    'eog'
]

COLLECTION_RATES = [
    0.2, 
    0.3, 
    0.4, 
    0.5, 
    0.6, 
    0.7, 
    0.8,
    0.9
]