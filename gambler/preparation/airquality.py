import os
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, List
import h5py


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


TRAIN_FRAC = 0.50
TEST_FRAC = 0.90
CHUNK_SIZE = 250


def iterate_dataset(path: str) -> Iterable[Any]:
    with open(path, 'r') as fin:
        is_header = True
        num_features = 0
        num_samples = 0

        next(fin)

        for line in fin:
            tokens = line.strip().split(';')

            label = 0 # No label for timeseries data
            features = list(map(int, [tokens[3], tokens[6], tokens[8], tokens[10], tokens[11]]))
            
            yield features, label


def write_dataset(path: str, output_folder: str):
    # Create the data writers
    inputs = dict(train=[], validation=[], test=[])
    output = dict(train=[], validation=[], test=[])

    label_counters = {
        'train': Counter(),
        'validation': Counter(),
        'test': Counter()
    }

    # Iterate over the dataset
    for index, sample in enumerate(iterate_dataset(path=path)):

        # Select the partition
        rand = random.random()
        if rand < TRAIN_FRAC:
            partition = 'train'
        elif rand < TEST_FRAC :
            partition = 'test'
        else:
            partition = 'validation'

        sample_inputs, sample_output = sample

        # inputs[partition].append(np.expand_dims(sample_inputs, axis=0))
        inputs[partition].append(sample_inputs)
        output[partition].append(sample_output)

        label_counters[partition][sample_output] += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(index + 1), end='\r')
    print()

    print(label_counters)
   
    for fold in inputs.keys():
        print(fold)

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

    path = os.path.join(args.input_folder, 'AirQualityUCI.csv')
    write_dataset(path, args.output_folder)