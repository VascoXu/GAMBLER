import os
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, List
import h5py


WINDOW_SIZE = 20
STRIDE = 4
TRAIN_FRAC = 0.85
VALID_FRAC = 0.15
CHUNK_SIZE = 500
SEQ_LENGTH = 100


# Data folder names
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

# Data format constants
SAMPLE_ID = 'sample_id'
DATA_FIELD_FORMAT = '{0}-{1}'
INPUTS = 'inputs'
OUTPUT = 'output'
TIMESTAMP = 'timestamp'
DATA_FIELDS = [INPUTS, OUTPUT]
INDEX_FILE = 'index.pkl.gz'


def read_dataset(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        sample_id = 0

        for _ in range(4):
            next(input_file)

        seq = []
        for line in input_file:
            line = line.strip().lower()
            tokens = line.split()

            hum = float(tokens[2])
            temp = float(tokens[3])

            label = int(tokens[-1])

            features = [[temp]]

            yield {
                SAMPLE_ID: sample_id,
                INPUTS: features,
                OUTPUT: label
            }
            sample_id += 1            


def get_partition(partitions: List[str], fractions: List[float]) -> str:
    r = random.random()

    frac_sum = 0.0
    for frac, partition in zip(fractions, partitions):
        frac_sum += frac
        if r < frac_sum:
            return partition

    return partitions[-1]


def write_dataset(path: str, output_folder: str, series: str):
    # Create the data writers
    if series == 'train':
        inputs = dict(train=[], validation=[])
        output = dict(train=[], validation=[])

        label_counters = {
            'train': Counter(),
            'validation': Counter()
        }
    else:
        inputs = dict(test=[])
        output = dict(test=[])

        label_counters = {
            'test': Counter()
        }

    # Iterate over the dataset
    for index, sample in enumerate(read_dataset(input_path=path)):
        # Select the partition
        if series == 'train':
            if random.random() < TRAIN_FRAC:
                partition = 'train'
            else:
                partition = 'validation'
        else:
            partition = 'test'

        sample_inputs = sample['inputs']
        sample_output = sample['output']

        inputs[partition].append(np.expand_dims(sample_inputs, axis=0))
        output[partition].append(sample_output)

        label_counters[partition][sample_output] += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(index + 1), end='\r')
    print()

    print(label_counters)


    for fold in inputs.keys():
        partition_inputs = np.vstack(inputs[fold])  # [N, T, D]
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

    train_file = os.path.join(args.input_folder, 'singlehop-indoor-moteid1-data.txt')
    test_file = os.path.join(args.input_folder, 'singlehop-outdoor-moteid4-data.txt')

    # Set random seed for reproducible results
    random.seed(42)

    write_dataset(train_file, output_folder=args.output_folder,  series='train')
    write_dataset(test_file, output_folder=args.output_folder, series='test')