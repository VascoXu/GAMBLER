from enum import unique
import numpy as np
import random
import h5py
from typing import List, Tuple
from math import ceil, floor
import os

from gambler.utils.loading import load_data
from argparse import ArgumentParser

PROPORTION = 0.6

def distribution(inputs, labels, idx, even=False):
    label_count = {}
    for label in labels:
        label_count[label] = label_count.get(label, 0) + 1
    
    max_count = label_count[max(label_count, key=label_count.get)]
    unique_labels = set(labels)
    num_seqs = max_count * len(unique_labels)

    # p = ceil(PROPORTION * num_seqs)
    p = floor((floor(PROPORTION*num_seqs)-max_count) / (1-PROPORTION))
    even_count = floor((num_seqs + p) / len(unique_labels))
    rest = max_count

    norm_inputs: List[np.ndarray] = []
    norm_labels: List[np.ndarray] = []  

    for i,label in enumerate(unique_labels):
        label_idx = np.where(labels == label)[0]

        if even:
            copy_amount = even_count - label_count[label]
        else:
            copy_amount = p - label_count[label] + max_count if i == idx else rest - label_count[label]
        
        # Add random data to fit desired proportion
        if copy_amount >= 0:
            _labels = labels[label_idx[0]:label_idx[len(label_idx)-1]+1].tolist()
            _inputs = [inputs[i] for i in label_idx]
            for _ in range(copy_amount):
                _inputs.append(random.choice(_inputs))
                _labels.append(_labels[0])

            _labels = np.asarray(_labels)
            _inputs = np.asarray(_inputs)

        # Trim data to fit desired proportion
        else:
            _labels = np.asarray(labels[label_idx[0]:label_idx[0]+rest])
            _inputs = np.asarray([inputs[i] for i in label_idx][0:rest])

        norm_inputs.append(_inputs)
        norm_labels.append(_labels)

    labels = np.concatenate(norm_labels).ravel().tolist()
    inputs = norm_inputs

    return (inputs, labels)
   

def write_dataset(inputs, labels, idx, output_folder):
    # Write new dataset to h5 file
    partition_folder = os.path.join('datasets', output_folder)
    if not os.path.exists(partition_folder):
        os.mkdir(partition_folder)

    partition_inputs = np.vstack(inputs)            # [N, T, D]
    partition_output = np.asarray(labels)           # [N]
    with h5py.File(os.path.join(partition_folder, f'data_{idx}.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
        input_dataset.write_direct(partition_inputs)

        output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
        output_dataset.write_direct(partition_output)


def prepare_dataset(dataset, output_folder):
    fold = 'validation'
    inputs, labels = load_data(dataset_name=dataset, fold=fold, dist='')

    # Seed random for reproducible results
    random.seed(42)

    unique_labels = set(labels)
    for i in range(len(unique_labels)):
        _inputs, _labels = distribution(inputs, labels, i)
        write_dataset(_inputs, _labels, i, output_folder)

    # Even distribution
    _inputs, _labels = distribution(inputs, labels, 0, even=True)
    write_dataset(_inputs, _labels, 'even', output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    prepare_dataset(args.dataset, args.output_folder)
