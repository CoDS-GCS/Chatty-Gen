import sys
sys.path.append('../')
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    baseline_file =  '../output2/dblp_e1_20_5.json'
    approach_file =  '../output2/dblp_e11_20_5.json'
    with open(baseline_file, 'r') as file:
        baseline = json.load(file)
        baseline_data = baseline['analysis']['types']

    with open(approach_file, 'r') as file:
        approach = json.load(file)
        approach_data = approach['analysis']['types']

    labels = list(set(baseline_data.keys()) | set(approach_data.keys()))

    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_baseline = [baseline_data.get(label, 0) for label in labels]
    values_approach = [approach_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.35

    # Create the grouped bar chart
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - bar_width / 2, values_baseline, bar_width, label='Baseline')
    bar2 = ax.bar(x + bar_width / 2, values_approach, bar_width, label='Approach')

    for i, value in enumerate(values_baseline):
        ax.text(i - bar_width / 2, value + 0.1, str(value), ha='center', va='bottom')

    for i, value in enumerate(values_approach):
        ax.text(i + bar_width / 2, value + 0.1, str(value), ha='center', va='bottom')

    # Set labels and title
    ax.set_xlabel('Question prefix for DBLP')
    ax.set_ylabel('Count')
    # ax.set_title('Comparison of Approaches')
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xticklabels(labels, rotation=90, ha='right')
    ax.legend()

    plt.savefig('dblp_complexity.pdf', bbox_inches='tight')
    # Show the plot
    plt.tight_layout()
    plt.show()
