import os.path
import numpy as np
from argparse import ArgumentParser
from typing import List

from gambler.utils.serialize_utils import array_to_fixed_point
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.file_utils import make_dir, load_h5_dataset, randomize_dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True, help='The name of the dataset to serialize.')
    parser.add_argument('--precision', type=int, required=True, help='The fixed point precision.')
    parser.add_argument('--fold', type=str, required=True, choices=['train', 'val', 'test'], help='The dataset fold to serialize.')
    parser.add_argument('--num-inputs', type=int, required=True, help='The maximum number of inputs to include.')
    parser.add_argument('--should-randomize', action='store_true', help='Randomize dataset before clipping num inputs.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to prepare the dataset for the MSP430 device.')
    args = parser.parse_args()

    assert (args.precision >= 0) and (args.precision <= 15), 'The precision must be in [0, 16).'

    # Load the inputs and labels
    """
    window_size = 20
    path = os.path.join('train', args.dataset_name, f'{args.fold}_{window_size}.csv')
    inputs, labels = load_h5_dataset(os.path.join('train', args.dataset_name, f'{args.fold}_{window_size}.csv'))
    """
    path = os.path.join('datasets', args.dataset_name, args.fold, 'data.h5')
    inputs, labels = load_data(dataset_name=args.dataset_name, fold=args.fold)

    inputs, labels = get_data(inputs=inputs, labels=labels, classes=[1, 1, 1, 2, 2, 2])

    #TODO: fix labels (currently match sequences not individual inputs)

    # Serialize the inputs into a static C array
    inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2])

    # inputs = inputs[0:args.num_inputs]
    # labels = labels[0:args.num_inputs]

    num_inputs, num_features = inputs.shape

    print(num_inputs)

    # Quantize the input features
    quantized_inputs = array_to_fixed_point(inputs, precision=args.precision, width=16)

    input_str = list(map(str, quantized_inputs.reshape(-1)))
    input_var = 'static int16_t DATASET[] = {{ {} }};\n'.format(','.join(input_str))

    label_str = list(map(str, labels))
    label_var = 'static uint8_t DATASET_LABELS[] = {{ {} }};\n'.format(','.join(label_str))

    with open('c_implementation/data.h', 'w') as fout:
        fout.write('#include <stdint.h>\n')

        if args.is_msp:
            fout.write('#include <msp430.h>\n')

        fout.write('#ifndef DATA_H_\n')
        fout.write('#define DATA_H_\n')

        fout.write('#define NUM_FEATURES {}\n'.format(num_features))
        fout.write('#define DATASET_LENGTH {}\n'.format(num_inputs*num_features))

        # For MSP430 implementations, place the data into FRAM
        if args.is_msp:
            fout.write('#pragma PERSISTENT(DATASET_INPUTS)\n')

        fout.write(input_var)

        # Do not save the labels for MSP430 versions. The labels are only there
        # for testing purposes.
        if not args.is_msp:
            fout.write(label_var)

        fout.write('#endif\n')
