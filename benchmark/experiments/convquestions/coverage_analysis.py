import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def plot_chart_one_bar(our_data):
    our_data = OrderedDict(sorted(our_data.items(), key=lambda item: item[1], reverse=True))

    labels = list()
    for item in our_data.keys():
        if item not in labels:
            labels.append(item)

    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_our_data = [our_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 25
    values_font_size = 18

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width, values_our_data, bar_width, label='ConvQuestions', color='orange')


    for i, value in enumerate(values_our_data):
        ax.text(i - bar_width, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    # Set labels and title
    ax.set_ylabel('Count', fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    ax.legend(loc='upper right', fontsize=values_font_size)
    fig.tight_layout()
    output_file = f"../Final_Benchmarks/conv_questions_all.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')

def group_and_get_percentage(input_dict):
    output_dict = dict()
    total = 0
    for key, value in input_dict.items():
        if key.lower() in ['is', 'are', 'does', 'was', 'did', 'has', 'can', 'do']:
            output_key = 'Boolean'
        elif key.lower() in ['butch', 'sean', 'through', 'it', 'in', 'to', 'the', 'by', 'under']:
            output_key = 'Noun'
        else:
            output_key = key

        if output_key in output_dict:
            output_dict[output_key] += value
        else:
            output_dict[output_key] = value

        total += value
    percentage_dist = dict()
    for key, value in output_dict.items():
        percentage_dist[key] = (value / total) * 100
    return percentage_dist

def process_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    distribution = dict()
    for inst in data:
        for question_obj in inst['questions']:
            question = question_obj['completed_question'] if "completed_question" in question_obj else question_obj['question']
            word = question.split(" ")[0]
            word = word.replace("'s","")
            word = word.lower()
            if word in distribution:
                distribution[word] += 1
            else:
                distribution[word] = 1
    return distribution


if __name__ == '__main__':
    file_name = 'test_set_ALL.json'
    distribution = process_file(file_name)
    distribution = group_and_get_percentage(distribution)
    plot_chart_one_bar(distribution)
    print(distribution)