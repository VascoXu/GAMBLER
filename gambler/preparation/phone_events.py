import os
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, List
import h5py


WINDOW_SIZE = 50
CHUNK_SIZE = 500
SEQ_LENGTH = 100


# Data folder names
TRAIN = 'train'
VALID = 'validation'
TEST = 'test'

# Data format constants
SAMPLE_ID = 'sample_id'
USER_ID = 'user_id'
DATA_FIELD_FORMAT = '{0}-{1}'
INPUTS = 'inputs'
OUTPUT = 'output'
TIMESTAMP = 'timestamp'
DATA_FIELDS = [INPUTS, OUTPUT]
INDEX_FILE = 'index.pkl.gz'

# Remaining are for training
VALID_USER_IDS = set([3, 5, 11, 14, 19, 29, 1])
TEST_USER_IDS = set([2, 4, 9, 10, 12, 13, 18, 20, 24, 30, 33])

LABEL_MAP = {
    'WALKING': 0,
    'JOGGING': 1,
    'SITTING': 2,
    'STANDING': 3,
    'PING_PONG': 4,
    'DRIVING': 5,
    'COOKING': 6
}

TRAIN_FRAC = 1


def iterate_dataset(path: str) -> Iterable[Any]:
    with open(path, 'r') as fin:
        is_header = True
        num_features = 0
        num_samples = 0

        next(fin)

        for line in fin:
            tokens = line.strip().split(',')

            label = LABEL_MAP[tokens[-1]]  # Map label into 0, 1, 2, 3
            features = list(map(float, tokens[1:-1]))

            yield features, label


def write_dataset(path: str, output_folder: str):
    # Create the data writers
    inputs = dict(test=[])
    output = dict(test=[])

    label_counters = {
        'test': Counter()
    }

    # Iterate over the dataset
    for index, sample in enumerate(iterate_dataset(path=path)):
        sample_inputs, sample_output = sample

        # inputs['test'].append(np.expand_dims(sample_inputs, axis=0))
        inputs['test'].append(sample_inputs)
        output['test'].append(sample_output)

        label_counters['test'][sample_output] += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(index + 1), end='\r')
    print()

    print(label_counters)
   
    for fold in inputs.keys():

        if len(inputs[fold]) == 0:
            print('WARNING: Empty fold {0}'.format(fold))
            continue

        partition_inputs = np.vstack(inputs[fold])  # [N, T, D]
        partition_inputs = np.expand_dims(partition_inputs, axis=1)
        partition_output = np.vstack(output[fold]).reshape(-1)  # [N]

        partition_folder = os.path.join(output_folder, fold)

        if not os.path.exists(partition_folder):
            os.mkdir(partition_folder)

        with h5py.File(os.path.join(partition_folder, 'data.h5'), 'w') as fout:
            input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
            input_dataset.write_direct(partition_inputs)

            output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
            output_dataset.write_direct(partition_output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    # Set the random seed for reproducible results
    random.seed(42)

    # Create the output folder
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    train_path = os.path.join(args.input_folder, 'cooking.csv')
    write_dataset(train_path, args.output_folder)