import numpy as np
from sklearn import tree, metrics
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier

from gambler.utils.loading import load_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--distribution', type=str, default='')
    args = parser.parse_args()

    fold = 'reconstructed'
    train_inputs, train_labels = load_data(dataset_name=args.dataset, fold='validation', dist='')
    train_labels = train_labels.reshape(-1)

    # Reshape train input vector
    num_seq, seq_length, num_features = train_inputs.shape
    train_inputs = train_inputs.reshape((num_seq, seq_length*num_features))

    # Reshape test input vector
    filename = f'reconstructed_{args.distribution}.h5'
    test_inputs, test_labels = load_data(dataset_name=args.dataset, fold=fold, dist='', filename=filename)
    test_labels = test_labels.reshape(-1)

    num_seq, seq_length, num_features = test_inputs.shape
    test_inputs = test_inputs.reshape((num_seq, seq_length*num_features))

    # Train Decision Tree Classifer
    clf = RandomForestClassifier(random_state=0)
    clf = clf.fit(train_inputs, train_labels)

    # Predict test dataset
    y_pred = clf.predict(test_inputs)

    # Model Accuracy
    print("Accuracy:", metrics.accuracy_score(test_labels, y_pred))
