import numpy as np
from argparse import ArgumentParser
from sklearn import tree
from typing import List, Tuple
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from gambler.utils.file_utils import read_pickle_gz
from gambler.utils.serialize_utils import serialize_int_array, serialize_float_array


def serialize_tree(clf: DecisionTreeClassifier, name: str, labels: List[int], precision: int, is_msp: bool):
    root = clf.tree_
    lines: List[str] = []
    num_nodes = len(root.feature)

    features_name = '{}_FEATURES'.format(name)
    features = serialize_int_array(var_name=features_name,
                                   array=root.feature,
                                   dtype='int16_t')

    if is_msp:
        lines.append('#pragma PERSISTENT({})'.format(features_name))
    
    lines.append(features)

    thresholds_name = '{}_THRESHOLDS'.format(name)
    thresholds = serialize_float_array(var_name=thresholds_name,
                                       array=root.threshold,
                                       width=16,
                                       precision=precision,
                                       dtype='int16_t')

    if is_msp:
        lines.append('#pragma PERSISTENT({})'.format(thresholds_name))
    
    lines.append(thresholds)

    left_name = '{}_CHILDREN_LEFT'.format(name)
    children_left = serialize_int_array(var_name=left_name,
                                        array=root.children_left,
                                        dtype='int16_t')

    if is_msp:
        lines.append('#pragma PERSISTENT({})'.format(left_name))
    
    lines.append(children_left)

    right_name = '{}_CHILDREN_RIGHT'.format(name)
    children_right = serialize_int_array(var_name=right_name,
                                         array=root.children_right,
                                         dtype='int16_t')

    if is_msp:
        lines.append('#pragma PERSISTENT({})'.format(right_name))
    
    lines.append(children_right)

    pred_name = '{}_PREDICTIONS'.format(name)
    predictions = serialize_int_array(var_name=pred_name,
                                      array=[labels[np.argmax(node[0])] for node in root.value],
                                      dtype='uint16_t')

    if is_msp:
        lines.append('#pragma PERSISTENT({})'.format(pred_name))
    
    lines.append(predictions)

    tree_var = 'static struct decision_tree {} = {{ {}, {}, {}, {}, {}, {} }};'.format(name, num_nodes, thresholds_name, features_name, pred_name, left_name, right_name)
    lines.append(tree_var)

    return '\n'.join(lines)


def serialize_ensemble(ensemble: AdaBoostClassifier, precision: int, is_msp: bool) -> str:
    lines: List[str] = []

    clfs = ensemble.estimators_
    num_estimators = len(clfs)
    boost_weights = ensemble.estimator_weights_
    num_labels = ensemble.n_classes_

    # Collect the information for all of the classifiers
    var_names: List[str] = []

    for idx, clf in enumerate(clfs):
        name = 'TREE_{}'.format(idx)
        lines.append(serialize_tree(clf, name=name, precision=precision, is_msp=is_msp))
        var_names.append(name)

    # Create the array of decision trees
    trees_var = 'static struct decision_tree *TREES[] = {{ {} }};'.format(','.join(('&{}'.format(n) for n in var_names)))
    lines.append(trees_var)

    # Create the array of boosting weights
    boost_weights_var = serialize_float_array(var_name='BOOST_WEIGHTS',
                                              array=boost_weights,
                                              precision=precision,
                                              width=16,
                                              dtype='int16_t')
    lines.append(boost_weights_var)

    # Create the ensemble structure
    ensemble_var = 'static struct adaboost_ensemble ENSEMBLE = {{ {}, {}, TREES, BOOST_WEIGHTS }};'.format(num_estimators, num_labels)
    lines.append(ensemble_var)

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model parameters. This should be a pickle gzip file.')
    parser.add_argument('--precision', type=int, required=True, help='The precision of fixed point values.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to serialize the system for the MSP430.')
    args = parser.parse_args()

    clf = pickle.load(open(args.model_path, 'rb'))

    num_labels = clf.n_classes_
    labels = clf.classes_
    num_input_features = clf.n_features_in_

    serialized_model = serialize_tree(clf, name='TREE', labels=labels, precision=args.precision, is_msp=args.is_msp)

    with open('c_implementation/parameters.h', 'w') as fout:
        fout.write('#include <stdint.h>\n')
        
        if args.is_msp:
            fout.write('#include <msp430.h>\n')

        fout.write('#include "decision_tree.h"\n')

        fout.write('#ifndef PARAMETERS_H_\n')
        fout.write('#define PARAMETERS_H_\n')

        fout.write('#define PRECISION {}\n'.format(args.precision))
        fout.write('#define NUM_LABELS {}\n'.format(num_labels))
        fout.write('#define NUM_TREE_FEATURES {}\n\n'.format(num_input_features))

        fout.write(serialized_model)

        fout.write('\n#endif\n')
