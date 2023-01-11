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
    'UPSTAIRS': 4,
    'DOWNSTAIRS': 5,
}

def read_dataset(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        sample_id = 0
        count = 0

        data_seq = []
        labels = []
        for line in input_file:
            line = line.strip().upper()
            tokens = line.split(',')

            if len(line) == 0:
                continue

            user_id, activity, timestamp = tokens[0], tokens[1], tokens[2]
            x_acc, y_acc, z_acc = tokens[3], tokens[4], tokens[5].split(';')[0]
            features = [float(x_acc), float(y_acc), float(z_acc)]
            
            data_seq.append(features)
            labels.append(activity)

            count += 1

            if count == WINDOW_SIZE:
                # Ensure all labels are equal for a sequence
                if labels.count(labels[0]) == len(labels):
                    yield {
                        SAMPLE_ID: sample_id,
                        USER_ID: int(user_id),
                        INPUTS: data_seq,
                        OUTPUT: LABEL_MAP[activity]
                    }
                    
                    sample_id += 1

                
                data_seq = []
                labels = []
                count = 0
            



def get_partition(user_id: int) -> str:
    if user_id in VALID_USER_IDS:
        return VALID
    elif user_id in TEST_USER_IDS:
        return TEST
    return TRAIN    


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
    for index, sample in enumerate(read_dataset(input_path=path)):
        # Select the partition
        partition = get_partition(user_id=sample[USER_ID])

        sample_inputs = sample[INPUTS]
        sample_output = sample[OUTPUT]

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

    input_file = os.path.join(args.input_folder, 'WISDM_ar_v1.1_raw.txt')

    # Set random seed for reproducible results
    random.seed(42)

    write_dataset(input_file, output_folder=args.output_folder)