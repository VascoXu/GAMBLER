import numpy as np
import os.path
import h5py
import socket
import time
from argparse import ArgumentParser
from collections import namedtuple, Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Optional, List, Tuple


def reconstruct_sequence(measurements: np.ndarray, collected_indices: List[int], seq_length: int) -> np.ndarray:
    """
    Reconstructs a sequence using a linear interpolation.
    Args:
        measurements: A [K, D] array of sub-sampled features
        collected_indices: A list of [K] indices of the collected features
        seq_length: The length of the full sequence (T)
    Returns:
        A [T, D] array of reconstructed measurements.
    """
    feature_list: List[np.ndarray] = []
    seq_idx = list(range(seq_length))

    # Interpolate the unseen measurements using a linear function
    for feature_idx in range(measurements.shape[1]):
        collected_features = measurements[:, feature_idx]  # [K]
        reconstructed = np.interp(x=seq_idx,
                                  xp=collected_indices,
                                  fp=collected_features,
                                  left=collected_features[0],
                                  right=collected_features[-1])

        feature_list.append(np.expand_dims(reconstructed, axis=-1))

    return np.concatenate(feature_list, axis=-1)  # [T, D]