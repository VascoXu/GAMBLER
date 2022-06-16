import numpy as np
import random


def get_data(args, inputs, labels):
    return args_data(args, inputs, labels)

def args_data(args, inputs, labels):
    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    if args.labels != 'all':
        """Return inputs of desired label"""
        desired_label = int(args.labels)
        label_idx = np.where(labels == desired_label)[0]
        unique_labels = set(labels)

        inputs = np.asarray([inputs[i] for i in label_idx])
        labels = np.asarray([labels[i] for i in label_idx])

        return (inputs, labels)

    if args.randomize == 'group':
        """Group randomize"""
        unique_labels = set(labels)
        label_idx = []
        for label in unique_labels:
            label_idx.append([np.where(labels == label)[0]])

        chunks = []
        n = args.block_size
        for i in range(len(label_idx)):
            lst = label_idx[i][0]
            for j in range(0, len(lst), n):
                chunks.append(lst[j:j + n]) # split into even-sized chunks
        
        random.shuffle(chunks)
        parts = [item for chunk in chunks for item in chunk] # flatten list

        # Randomize chunks
        inputs = np.asarray([inputs[i] for i in parts])
        labels = np.asarray([labels[i] for i in parts])
        
        return (inputs, labels)

    elif args.randomize == 'full':
        """Full randomize"""
        rand_idx = list(range(num_seq))
        random.shuffle(rand_idx)
        inputs = np.asarray([inputs[i] for i in rand_idx])
        labels = np.asarray([labels[i] for i in rand_idx])
    
    return (inputs, labels)