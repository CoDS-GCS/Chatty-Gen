import json
import os
import sys

sys.path.append('../../../benchmark')
sys.path.append('../')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from seed_node_extractor import utils
from collections import OrderedDict
from seed_node_extractor.seed_node_selector import SeedNodeSelector

maestro_directories = {'dbpedia': "../Final_Benchmarks/Smart_dbpedia_1_LIKE_QALD_450_pruned.json",
                       "yago": "../Final_Benchmarks/Smart_Yago_1_LIKE_QALD_450_pruned.json",
                       "dblp": "../Final_Benchmarks/Smart_dblp_1_LIKE_QALD_450_pruned.json",
                       "makg": "../Final_Benchmarks/MAKG_5_LIKE_QALD_450_pruned.json"}
our_files = {"dbpedia": '../Final_Benchmarks/dbpedia_e11_20_5_original.json',
            "yago": '../Final_Benchmarks/yago_e11_20_5_original.json',
             'dblp': '../Final_Benchmarks/dblp_e11_20_5_original.json',
             'makg': '../Final_Benchmarks/Summarized_vs_Subgraph/makg_subgraph_summarized_20_5_original.json'}

def get_unique_types_per_file(json_data):
    unique_types = dict()
    for instance in json_data:
        type = instance["seedType_withPrefix"]
        key = utils.get_name(type)
        if key not in unique_types:
            unique_types[key] = 1
        else:
            unique_types[key] += 1
    return unique_types


def get_total_type_distribution(current_dist, exisiting_dist):
    for key, value in current_dist.items():
        if key in exisiting_dist:
            exisiting_dist[key] += value
        else:
            exisiting_dist[key] = value
    return exisiting_dist


def get_types_for_kg(kg_name):
    knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
    knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
    selector = SeedNodeSelector(kg_name)
    kg_type_distribution = utils.get_type_distrubution(knowledge_graph_uri, knowledge_graph_prefix)
    distribution = pd.DataFrame(kg_type_distribution)
    percentage_df = selector.calculate_class_importance(distribution)
    update_df = selector.remove_rare_types(percentage_df)
    cleaned_df = selector.eliminate_dominated_parents(update_df, knowledge_graph_uri)
    json_object = json.loads(cleaned_df)
    kg_data = dict()
    for obj in json_object:
        kg_data[utils.get_name(obj['Type']).strip()] = obj['percentage']
    return kg_data


def get_type_distribution_our_approach(kg_type_distribution, file):
    # get seed entities
    kg_name = file.split('/')[-1].split('_')[0]
    file = open(file, 'r')
    file_data = json.load(file)
    seed_nodes = list()
    for inst in file_data['data']:
        seed_nodes.append(inst['seed_entity'])

    # Eliminate rare types
    non_rare_types = list(kg_type_distribution.keys())
    non_rare_types = [x.strip() for x in non_rare_types]

    types_distribution = dict()
    endpoint = utils.knowledge_graph_to_uri[kg_name][0]
    for node in seed_nodes:
        query = ("Select ?type where {"
                 f"<{node}> rdf:type ?type"
                 "}")
        result = utils.send_sparql_query(endpoint, query)
        for binding in result["results"]["bindings"]:
            type = binding.get('type', {}).get('value', None)
            type = utils.get_name(type)
            if type not in non_rare_types:
                continue
            if type in types_distribution:
                types_distribution[type] += 1
            else:
                types_distribution[type] = 1

    json_for_df = []
    for key, value in types_distribution.items():
        json_for_df.append({"Type": key, "Count": value})
    df = pd.DataFrame(json_for_df)
    total_count = df['Count'].sum()
    percentage_df = df.copy()
    percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
    json_data = percentage_df.to_json(orient='records')
    json_object = json.loads(json_data)
    benchmark_data = dict()
    for obj in json_object:
        benchmark_data[obj['Type'].strip()] = obj['percentage']
    return benchmark_data

# def get_types_distribution_for_maestro_benchmark(directory_path):
#     type_dist = dict()
#     for filename in os.listdir(directory_path):
#         if os.path.isfile(os.path.join(directory_path, filename)):
#             if filename == 'config.txt':
#                 continue
#             with open(f"{directory_path}/{filename}", 'r') as f:
#                 data = json.load(f)
#             current_dist = get_unique_types_per_file(data)
#             type_dist = get_total_type_distribution(current_dist, type_dist)
#     total = sum(type_dist.values())
#     percentage_dict = dict()
#     for key, value in type_dist.items():
#         percentage_dict[key] = round((value / total) * 100, 1)
#     return percentage_dict

def get_types_distribution_for_maestro_benchmark(filename):
    type_dist = dict()
    with open(f"{filename}", 'r') as f:
        data = json.load(f)
    current_dist = get_unique_types_per_file(data)
    type_dist = get_total_type_distribution(current_dist, type_dist)
    total = sum(type_dist.values())
    percentage_dict = dict()
    for key, value in type_dist.items():
        percentage_dict[key] = round((value / total) * 100, 1)
    return percentage_dict

def plot_comparison(maestro_data, kg_data, our_data, kg_name):
    maestro_data = OrderedDict(sorted(maestro_data.items(), key=lambda item: item[1], reverse=True))
    kg_data = OrderedDict(sorted(kg_data.items(), key=lambda item: item[1], reverse=True))
    our_data = OrderedDict(sorted(our_data.items(), key=lambda item: item[1], reverse=True))

    labels = list()

    for item in kg_data.keys():
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
    values_kg = [kg_data.get(label, 0) for label in labels]
    values_our = [our_data.get(label, 0) for label in labels]

    # comment both for DBLP, comment second for dbpedia, uncomments both for yago
    labels = [label.replace(' ', '\n', 1) for label in labels]
    # labels = [label if len(label) < 7 or '\n' in label else label[:6]+'-\n'+label[6:] for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 25
    values_font_size = 18

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width, values_kg, bar_width, label='KG', color='steelblue')
    bar2 = ax.bar(x, values_our, bar_width, label='Chatty-Gen', color='tab:green')
    bar3 = ax.bar(x + bar_width, values_maestro, bar_width, label='Maestro', color='grey')

    for i, value in enumerate(values_kg):
        ax.text(i - bar_width, value + 0.1, str(int(round(value, 0))) if value > 0 else 'X', ha='center', va='bottom',  fontsize=values_font_size)

    for i, value in enumerate(values_our):
        ax.text(i, value + 0.1, str(int(round(value, 0))) if value > 0 else 'X', ha='center', va='bottom', fontsize=values_font_size)

    for i, value in enumerate(values_maestro):
        ax.text(i + bar_width, value + 0.1, str(int(round(value, 0))) if value > 0 else 'X', ha='center', va='bottom',  fontsize=values_font_size)



    # Set labels and title
    # ax.set_xlabel(f"Node Types for {kg_name.upper()}",  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Percentage', fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='center', fontsize=font_size)
    # ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.legend(loc='best', fontsize=values_font_size)

    fig.tight_layout()
    # output_file = f"../Figures/{kg_name}__maestro_diversity.pdf"
    output_file = f"../Final_Benchmarks/{kg_name}__maestro_diversity2.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')

def remove_small_percentages(kg_distribution, maestro_dist, our_distribution, limit=5):
    keys_to_remove = list()
    for key, value in kg_distribution.items():
        if round(value, 0) < limit:
            keys_to_remove.append(key)

    kg_distribution_output = dict()
    for key, value in kg_distribution.items():
        if key not in keys_to_remove:
            kg_distribution_output[key] = value

    maestro_output = dict()
    for key, value in maestro_dist.items():
        if key not in keys_to_remove:
            maestro_output[key] = value

    our_output = dict()
    for key, value in our_distribution.items():
      if key not in keys_to_remove:
          our_output[key] = value

    return kg_distribution_output, maestro_output, our_output


def remove_small_percentages_mag(kg_distribution, our_distribution, limit=5):
    keys_to_remove = list()
    for key, value in kg_distribution.items():
        if round(value, 0) < limit:
            keys_to_remove.append(key)

    kg_distribution_output = dict()
    for key, value in kg_distribution.items():
        if key not in keys_to_remove:
            kg_distribution_output[key] = value

    our_output = dict()
    for key, value in our_distribution.items():
      if key not in keys_to_remove:
          our_output[key] = value

    return kg_distribution_output, our_output

def plot_comparison_mag(kg_data, our_data, kg_name):
    kg_data = OrderedDict(sorted(kg_data.items(), key=lambda item: item[1], reverse=True))
    our_data = OrderedDict(sorted(our_data.items(), key=lambda item: item[1], reverse=True))

    labels = list()

    for item in kg_data.keys():
        if item not in labels:
            labels.append(item)

    for item in our_data.keys():
        if item not in labels:
            labels.append(item)

    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_kg = [kg_data.get(label, 0) for label in labels]
    values_our = [our_data.get(label, 0) for label in labels]

    # comment both for DBLP, comment second for dbpedia, uncomments both for yago
    labels = [label.replace(' ', '\n', 1) for label in labels]
    # labels = [label if len(label) < 7 or '\n' in label else label[:6]+'-\n'+label[6:] for label in labels]

    # Define bar width
    bar_width = 0.3
    font_size = 25
    values_font_size = 18

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width / 2, values_kg, bar_width, label='KG', color='steelblue')
    bar2 = ax.bar(x + bar_width / 2, values_our, bar_width, label='Chatty-Gen', color='tab:green')

    for i, value in enumerate(values_kg):
        ax.text(i - bar_width / 2, value + 0.1, str(int(round(value, 0))) if value > 0 else 'X', ha='center', va='bottom',  fontsize=values_font_size)

    for i, value in enumerate(values_our):
        ax.text(i + bar_width / 2, value + 0.1, str(int(round(value, 0))) if value > 0 else 'X', ha='center', va='bottom', fontsize=values_font_size)


    # Set labels and title
    # ax.set_xlabel(f"Node Types for {kg_name.upper()}",  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Percentage', fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='center', fontsize=font_size)
    # ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.legend(loc='best', fontsize=values_font_size)

    fig.tight_layout()
    # output_file = f"../Figures/{kg_name}__maestro_diversity.pdf"
    output_file = f"../Final_Benchmarks/{kg_name}__maestro_diversity2.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
    kg_name = 'makg'
    kg_distribution = get_types_for_kg(kg_name)
    maestro_dist = get_types_distribution_for_maestro_benchmark(maestro_directories[kg_name])
    our_approach_dist = get_type_distribution_our_approach(kg_distribution, our_files[kg_name])
    kg_distribution, maestro_dist, our_approach_dist = remove_small_percentages(kg_distribution, maestro_dist, our_approach_dist)
    plot_comparison(maestro_dist, kg_distribution, our_approach_dist, kg_name)

#     MAG
#     kg_name = 'makg'
#     kg_distribution = get_types_for_kg(kg_name)
#     our_approach_dist = get_type_distribution_our_approach(kg_distribution, our_files[kg_name])
#     kg_distribution, our_approach_dist = remove_small_percentages_mag(kg_distribution, our_approach_dist)
    # plot_comparison_mag(kg_distribution, our_approach_dist, kg_name)
