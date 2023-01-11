import numpy as np
import random

from gambler.utils.loading import load_data

def get_data(args, inputs, labels):
    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    inputs, labels = load_data(dataset_name=args.dataset, fold=args.fold, dist='')
    num_seq = len(labels)

    if len(args.classes) > 0:
        """Specify classes"""
        label_parts = []
        input_parts = []

        for label in args.classes:
            label_idx = np.where(labels == label)[0]

            # pick random sequence
            random_idx = random.choice(label_idx)
            random_idx = label_idx[0]

            input_seq = inputs[random_idx]
            label_seq = labels[random_idx]
            if seq_length == 1:
                input_seq = input_seq[0]

            input_parts.append(np.asarray(input_seq))
            label_parts.append(np.asarray(label_seq))

            """
            # pick all sequences from label
            input_parts.append(np.asarray([inputs[i] for i in label_idx]))
            label_parts.append(np.asarray([labels[i] for i in label_idx]))

        input_parts = [item for chunk in input_parts for item in chunk] # flatten list
        label_parts = [item for chunk in label_parts for item in chunk] # flatten list
        """

        return (np.array(input_parts), np.array(label_parts))

    
    return (inputs, labels)