import numpy as np
import random

from gambler.utils.loading import load_data

def get_data(classes, inputs, labels):
    """
    Rearrange dataset for testing.
    Args:
        classes: The desired list of classes 
        inputs: The input data
        labels: The input labels
    Returns:
        A tuple of (1) [N, T, D] inputs and (2) [N] labels.
    """

    if len(classes) > 0:
        _inputs = []
        _labels = []

        for label in classes:
            # Find indices of the desired class
            label_idx = np.where(labels == label)[0]

            # Pick a random sequence of the desired class
            seq_idx = random.choice(label_idx)
            seq = inputs[seq_idx]
            label = labels[seq_idx]

            # if # TODO: fix for when data is one large sequence
            _inputs.append(np.asarray(seq))
            _labels.append(np.asarray(label))

        return (np.array(_inputs), np.array(_labels))

        # desired_label = int(classes[0])
        # label_idx = np.where(labels == desired_label)[0]
        # inputs = np.asarray([inputs[i] for i in label_idx])
        # labels = np.asarray([labels[i] for i in label_idx])

        # return (inputs, labels)
    
    return (inputs, labels)