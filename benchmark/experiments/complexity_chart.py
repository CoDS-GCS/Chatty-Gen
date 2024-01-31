import sys
sys.path.append('../')
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def bar_chart_for_two_question_words(baseline_data, approach_data):


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


def extract_first_word(data):
    updated_data = dict()
    for key, value in data.items():
        word = key.split(' ')[0]
        if word in updated_data:
            updated_data[word] += value
        else:
            updated_data[word] = value
    return updated_data

def bar_chart_for_one_question_word(baseline_data, approach_data, kg_name):
    baseline_data = extract_first_word(baseline_data)
    approach_data = extract_first_word(approach_data)

    baseline_data = OrderedDict(sorted(baseline_data.items(), key=lambda item: item[1], reverse=True))
    approach_data = OrderedDict(sorted(approach_data.items(), key=lambda item: item[1], reverse=True))

    labels = list(baseline_data.keys() | approach_data.keys())
    labels = list()
    for item in baseline_data.keys():
        if item not in labels:
            labels.append(item)

    for item in approach_data.keys():
        if item not in labels:
            labels.append(item)
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
    ax.set_xlabel('Question prefix for '+ kg_name.upper())
    ax.set_ylabel('Count')
    # ax.set_title('Comparison of Approaches')
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.legend()
    output_file = f"{kg_name}_complexity.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    # Show the plot
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    baseline_file =  '../output_26/yago_e1_20_5.json'
    approach_file =  '../output_26/yago_e11_20_5.json'
    kg_name = baseline_file.split('/')[-1].split('_')[0]
    with open(baseline_file, 'r') as file:
        baseline = json.load(file)
        baseline_data = baseline['analysis']['types']

    with open(approach_file, 'r') as file:
        approach = json.load(file)
        approach_data = approach['analysis']['types']
    # bar_chart_for_two_question_words()
    bar_chart_for_one_question_word(baseline_data, approach_data, kg_name)