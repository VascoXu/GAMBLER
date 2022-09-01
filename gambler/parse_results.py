import json
import numpy as np
import statistics
import matplotlib.pyplot as plt
from gambler.analysis.plot_utils import bar_plot

filename = "binomial.txt"


policies = ['uniform',
            'heuristic',
            'deviation',
            'adaptive_uniform',
            'adaptive_training',
            ]
budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 'Mean Error']

def rewrite_file():
    with open("binomial.txt", "rt") as fin:
        with open("binomial_2.txt", "wt") as fout:
            for line in fin:
                fout.write(line.replace("'", '"'))


def plot_normalized_error():
    with open(filename) as f:
        results = json.load(f)
        
        # find means
        mean_errors = {'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_training': []}
        
        # normalize to uniform error
        a = np.array(results['uniform'])
        uniform_error = np.mean(a, axis=0).tolist()

        for policy in policies:
            if policy != 'uniform':
                a = np.array(results[policy])
                mean_error = np.mean(a, axis=0).tolist()[:-1]
                # mean_error = np.mean(a, axis=0).tolist()
                mean_errors[policy] = [mean_error[i]/uniform_error[i] for i in range(len(mean_error))]
                mean_errors[policy].append(statistics.harmonic_mean(mean_errors[policy]))

        print(mean_errors)

        fig, ax = plt.subplots()
        bar_plot(ax, mean_errors, total_width=.8, single_width=.9, legend=['Adaptive Heuristic', 'Adaptive Deviation', 'Adaptive Uniform', 'Gambler'])
        plt.xticks(range(len(budgets)), budgets)
        plt.title('Epilepsy (Skewed Distribution): Normalized Error across Multiple Budgets')
        fig.tight_layout()
        plt.show()


def plot_reconstruction_error():
    with open(filename) as f:
        results = json.load(f)
        
        # find means
        mean_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_training': []}
        for policy in policies:
            a = np.array(results[policy])
            # mean_errors[policy] = np.mean(a, axis=0).tolist()
            mean_errors[policy] = np.mean(a, axis=0).tolist()[:-1]
            mean_errors[policy].append(sum(mean_errors[policy])/len(mean_errors[policy]))

        fig, ax = plt.subplots()
        bar_plot(ax, mean_errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Heuristic', 'Adaptive Deviation', 'Adaptive Uniform', 'Gambler'])
        plt.xticks(range(len(budgets)), budgets)
        plt.title('Epilepsy (Skewed Distribution): Reconstruction Error across Multiple Budgets')
        fig.tight_layout()
        plt.show()

# rewrite_file()
plot_normalized_error()