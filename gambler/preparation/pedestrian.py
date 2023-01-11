import os.path
import numpy as np
import random
from argparse import ArgumentParser
from collections import Counter
import h5py

# Data folder names
TRAIN = 'train'
VALID = 'validation'
TEST = 'test'

# Data format constants
SAMPLE_ID = 'sample_id'
DATA_FIELD_FORMAT = '{0}-{1}'
INPUTS = 'inputs'
OUTPUT = 'output'
TIMESTAMP = 'timestamp'
DATA_FIELDS = [INPUTS, OUTPUT]
INDEX_FILE = 'index.pkl.gz'

WINDOW_SIZE = 20
STRIDE = 4
TRAIN_FRAC = 0.85
VALID_FRAC = 0.15
CHUNK_SIZE = 5000


def write_dataset(data: np.ndarray, output_folder: str, series: str):
    # Create the data writers
    if series == TRAIN:
        inputs = dict(train=[], validation=[])
        output = dict(train=[], validation=[])

        label_counters = {
            TRAIN: Counter(),
            VALID: Counter()
        }
    else:
        inputs = dict(test=[])
        output = dict(test=[])

        label_counters = {
            TEST: Counter()
        }

    sample_id = 0
    for index, features in enumerate(data):
        label = int(features[0])
        input_features = features[1:].reshape(-1, 1).astype(float).tolist()

        # Get the data partition
        if series == TRAIN:
            if random.random() < TRAIN_FRAC:
                partition = TRAIN
            else:
                partition = VALID
        else:
            partition = TEST

        # Create the sample and add to corresponding data writer
        for i in range(0, len(input_features) - WINDOW_SIZE + 1, STRIDE):
            sample = {
                SAMPLE_ID: sample_id,
                OUTPUT: label,
                INPUTS: input_features[i:i+WINDOW_SIZE],
            }

            sample_inputs = sample['inputs']
            sample_output = sample['output']    

            inputs[partition].append(np.expand_dims(sample_inputs, axis=0))
            output[partition].append(sample_output)

            label_counters[partition][sample_output] += 1            
            sample_id += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} sample.'.format(index + 1), end='\r')

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

    # Set the random seed for reproducible results
    random.seed(42)

    train_path = os.path.join(args.input_folder, 'MelbournePedestrian_TRAIN.txt')
    train_data = np.loadtxt(train_path)  # [T, D + 1] array. Element 0 is the label.

    # make_dir(args.output_folder)
    write_dataset(train_data, output_folder=args.output_folder,  series='train')

    test_path = os.path.join(args.input_folder, 'MelbournePedestrian_TEST.txt')
    test_data = np.loadtxt(test_path)
    write_dataset(test_data, output_folder=args.output_folder, series='test')
