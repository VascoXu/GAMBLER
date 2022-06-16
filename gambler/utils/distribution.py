import os


def load_distribution(dist_name):
    """Load distribution of variance values"""
    dirname = os.path.dirname(__file__)
    dataset_file = dist_name
    data_file = os.path.join(dirname, '..', dataset_file)

    with open(data_file) as f:
        lines = f.read().splitlines()
        lines = list(map(float, lines))
        return lines