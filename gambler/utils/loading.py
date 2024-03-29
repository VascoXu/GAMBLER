import h5py
import os.path
import numpy as np
from typing import Tuple


def load_data(dataset_name: str, fold: str, filename: str='') -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the dataset inputs and labels.
    Args:
        dataset_name: The name of the dataset
        fold: The name of the fold to retrieve
    Returns:
        A tuple of (1) [N, T, D] inputs and (2) [N] labels.
    """
    dirname = os.path.dirname(__file__)
    
    name = 'data.h5'
    dataset_file = name if filename == '' else filename

    data_file = os.path.join(dirname, '..', 'datasets', dataset_name, fold, dataset_file)

    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    if len(inputs.shape) == 2:
        inputs = np.expand_dims(inputs, axis=-1)  # [N, T, 1]        
        
    output = output.reshape(-1).astype(int)

    return inputs, output