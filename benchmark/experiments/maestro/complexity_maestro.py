import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

maestro_directories = {'dbpedia': "../Final_Benchmarks/Smart_dbpedia_1_LIKE_QALD_450_pruned.json",
                       "yago": "../Final_Benchmarks/Smart_Yago_1_LIKE_QALD_450_pruned.json",
                       "dblp": "../Final_Benchmarks/Smart_dblp_1_LIKE_QALD_450_pruned.json",
                       "makg": "../Final_Benchmarks/MAKG_5_LIKE_QALD_450_pruned.json"}
our_files = {"dbpedia": '../Final_Benchmarks/dbpedia_e11_20_5_original.json',
            "yago": '../Final_Benchmarks/yago_e11_20_5_original.json',
             'dblp': '../Final_Benchmarks/dblp_e11_20_5_original.json',
             'makg': '../Final_Benchmarks/Summarized_vs_Subgraph/makg_subgraph_summarized_20_5_original.json'}
def get_question_coverage_distribution(json_data, exisiting_dist):
    for instance in json_data:
        question = instance["questionString"]
        word = question.split(" ")[0]
        if word in exisiting_dist:
            exisiting_dist[word] += 1
        else:
            exisiting_dist[word] = 1
    return exisiting_dist

# def get_maestro_data(directory_path):
#     coverage_dist = dict()
#     total_num_questions = 0
#     for filename in os.listdir(directory_path):
#         if os.path.isfile(os.path.join(directory_path, filename)):
#             if filename == 'config.txt':
#                 continue
#             with open(f"{directory_path}/{filename}", 'r') as f:
#                 data = json.load(f)
#             total_num_questions += len(data)
#             coverage_dist = get_question_coverage_distribution(data, coverage_dist)
#     return coverage_dist

def get_maestro_data(filename):
    coverage_dist = dict()
    total_num_questions = 0
    with open(f"{filename}", 'r') as f:
        data = json.load(f)
    total_num_questions += len(data)
    coverage_dist = get_question_coverage_distribution(data, coverage_dist)
    return coverage_dist

def get_our_data(approach_file):
    with open(approach_file, 'r') as file:
        approach = json.load(file)
        approach_data = approach['analysis']['types']

    data = dict()

    for key, value in approach_data.items():
        word = key.split(' ')[0].replace("'","")
        if word in data:
            data[word] += value
        else:
            data[word] = value

    return data



def plot_chart(maestro_data, our_data, qald_data, kg_name):

    labels = list()

    for item in qald_data.keys():
        if item not in labels:
            labels.append(item)

    for item in our_data.keys():
        if item not in labels:
            labels.append(item)

    for item in maestro_data.keys():
        if item not in labels:
            labels.append(item)


    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_maestro = [maestro_data.get(label, 0) for label in labels]
    values_our_data = [our_data.get(label, 0) for label in labels]
    values_qald = [qald_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 15

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width, values_qald, bar_width, label='QALD', color='coral')
    bar2 = ax.bar(x, values_our_data, bar_width, label='Chatty-Gen', color='tab:green')
    bar3 = ax.bar(x + bar_width, values_maestro, bar_width, label='Maestro', color='grey')

    for i, value in enumerate(values_qald):
        ax.text(i - bar_width, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom', fontsize=10)

    for i, value in enumerate(values_our_data):
        ax.text(i, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=10)

    for i, value in enumerate(values_maestro):
        ax.text(i + bar_width, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=10)



    # Set labels and title
    # ax.set_xlabel('Question prefix for '+ kg_name.upper(),  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Count', fontsize=font_size)
    # ax.set_title('Comparison of Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    # ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.legend(loc='upper right', fontsize=13)
    fig.tight_layout()
    # output_file = f"../Figures/maestro_complexity_{kg_name}.pdf"
    output_file = f"../Final_Benchmarks/maestro_complexity_{kg_name}.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')


def plot_chart_two_bars(maestro_data, our_data, kg_name):
    maestro_data = OrderedDict(sorted(maestro_data.items(), key=lambda item: item[1], reverse=True))
    our_data = OrderedDict(sorted(our_data.items(), key=lambda item: item[1], reverse=True))

    labels = list()
    for item in our_data.keys():
        if item not in labels:
            labels.append(item)

    for item in maestro_data.keys():
        if item not in labels:
            labels.append(item)


    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_maestro = [maestro_data.get(label, 0) for label in labels]
    values_our_data = [our_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 25
    values_font_size = 18

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width / 2, values_our_data, bar_width, label='Chatty-Gen', color='tab:green')
    bar2 = ax.bar(x + bar_width / 2, values_maestro, bar_width, label='Maestro', color='grey')


    for i, value in enumerate(values_our_data):
        ax.text(i - bar_width / 2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    for i, value in enumerate(values_maestro):
        ax.text(i + bar_width /2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)



    # Set labels and title
    # ax.set_xlabel('Question prefix for '+ kg_name.upper(),  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Count', fontsize=font_size)
    # ax.set_title('Comparison of Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    # ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.legend(loc='upper right', fontsize=values_font_size)
    fig.tight_layout()
    # output_file = f"../Figures/maestro_complexity_{kg_name}.pdf"
    output_file = f"../Final_Benchmarks/maestro_complexity_{kg_name}.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')


def get_qald_data():
    qald_file = '../qald-9-test-multilingual.json'
    qald_data = dict()
    total = 0
    with open(qald_file) as f:
        qald9_testset = json.load(f)
        for question in qald9_testset['questions']:
            for language_variant_question in question['question']:
                if language_variant_question['language'] == 'en':
                    question_text = language_variant_question['string'].strip()
                    break
            word = question_text.split()[0]
            if word in qald_data:
                qald_data[word] += 1
            else:
                qald_data[word] = 1

    return qald_data

def group_and_get_percentage(input_dict):
    output_dict = dict()
    total = 0
    for key, value in input_dict.items():
        if key.lower() in ['is', 'are', 'does', 'was', 'did', 'has', 'can']:
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

def plot_chart_one_bar(our_data, kg_name):
    # maestro_data = OrderedDict(sorted(maestro_data.items(), key=lambda item: item[1], reverse=True))
    our_data = OrderedDict(sorted(our_data.items(), key=lambda item: item[1], reverse=True))

    labels = list()
    for item in our_data.keys():
        if item not in labels:
            labels.append(item)

    # for item in maestro_data.keys():
    #     if item not in labels:
    #         labels.append(item)


    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    # values_maestro = [maestro_data.get(label, 0) for label in labels]
    values_our_data = [our_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 25
    values_font_size = 18

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width, values_our_data, bar_width, label='Chatty-Gen', color='tab:green')
    # bar2 = ax.bar(x + bar_width / 2, values_maestro, bar_width, label='Maestro', color='grey')


    for i, value in enumerate(values_our_data):
        ax.text(i - bar_width, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    # for i, value in enumerate(values_maestro):
    #     ax.text(i + bar_width /2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)



    # Set labels and title
    # ax.set_xlabel('Question prefix for '+ kg_name.upper(),  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Count', fontsize=font_size)
    # ax.set_title('Comparison of Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    # ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.legend(loc='upper right', fontsize=values_font_size)
    fig.tight_layout()
    # output_file = f"../Figures/maestro_complexity_{kg_name}.pdf"
    output_file = f"../Final_Benchmarks/maestro_complexity_{kg_name}.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
    kg_name = "makg"
    maestro_data = get_maestro_data(maestro_directories[kg_name])
    maestro_data = group_and_get_percentage(maestro_data)
    our_data = get_our_data(our_files[kg_name])
    our_data = group_and_get_percentage(our_data)
    # # qald_data = get_qald_data()
    # # qald_data = group_and_get_percentage(qald_data)
    # # plot_chart(maestro_data, our_data, qald_data, kg_name)
    plot_chart_two_bars(maestro_data, our_data, kg_name)

    #MAG
    # kg_name = "makg"
    # our_data = get_our_data(our_files[kg_name])
    # our_data = group_and_get_percentage(our_data)
    # plot_chart_one_bar(our_data, kg_name)